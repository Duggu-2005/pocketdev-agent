"""Git operations. Arguments are always passed as lists (never through a shell).

Safety rules enforced here:
- push only with `git push origin HEAD:refs/heads/<branch>`, never --force;
- refuse to push to the base branch, the repo default branch, main or master;
- the GitHub token persisted by actions/checkout is removed from the git config before any
  project code (tests/builds) runs, and passed per-command with `-c` only for fetch/push.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

BOT_NAME = "PocketDev AI"
BOT_EMAIL = "noreply@pocketdev.ai"
ALWAYS_PROTECTED = {"main", "master"}
BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,199}$")


class GitError(RuntimeError):
    pass


class PushRefused(GitError):
    pass


@dataclass
class GitRepo:
    root: Path
    github_token: str = ""
    server_url: str = "https://github.com"
    _auth_header: str = field(default="", repr=False)

    # ------------------------------------------------------------------ plumbing
    def _env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if not _sensitive_env(k)}
        env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "", "LC_ALL": "C"})
        return env

    def run(self, *args: str, check: bool = True, auth: bool = False, timeout: int = 300) -> str:
        cmd = ["git"]
        if auth and self._auth_header:
            cmd += ["-c", f"http.{self.server_url}/.extraheader={self._auth_header}"]
        cmd += list(args)
        p = subprocess.run(cmd, cwd=self.root, env=self._env(), capture_output=True, timeout=timeout)
        out = p.stdout.decode("utf-8", errors="replace")
        if check and p.returncode != 0:
            err = p.stderr.decode("utf-8", errors="replace").strip()
            raise GitError(f"git {args[0]} failed: {err[-500:]}")
        return out

    def ok(self, *args: str, auth: bool = False) -> bool:
        try:
            self.run(*args, auth=auth)
            return True
        except GitError:
            return False

    # ------------------------------------------------------------------ credentials
    def secure_credentials(self, temp_dir: str = "") -> None:
        """Take the token out of the git config so project code can't read it from .git/config."""
        key = f"http.{self.server_url}/.extraheader"
        existing = self.run("config", "--get-all", key, check=False).strip().splitlines()
        if self.github_token:
            basic = base64.b64encode(f"x-access-token:{self.github_token}".encode()).decode()
            self._auth_header = f"AUTHORIZATION: basic {basic}"
        elif existing:
            self._auth_header = existing[-1]
        self.run("config", "--local", "--unset-all", key, check=False)
        # actions/checkout v6+ keeps the credentials in a separate file under RUNNER_TEMP, included from
        # .git/config via includeIf. Drop the include and delete that file (only if it holds the header).
        includes = self.run("config", "--local", "--get-regexp", r"^includeif\..*\.path$", check=False)
        for line in includes.splitlines():
            name, _, value = line.partition(" ")
            path = Path(value.strip())
            in_temp = bool(temp_dir) and _is_under(path, Path(temp_dir))
            if in_temp or "credentials" in path.name.lower():
                self.run("config", "--local", "--unset-all", name, check=False)
                try:
                    if path.is_file() and "extraheader" in path.read_text(errors="ignore"):
                        path.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------------ branches
    def default_branches(self, event_path: str = "", ref_name: str = "") -> set[str]:
        names = set()
        head = self.run("symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD", check=False).strip()
        if head.startswith("origin/"):
            names.add(head[len("origin/"):])
        if event_path:
            try:
                data = json.loads(Path(event_path).read_text(encoding="utf-8"))
                db = (data.get("repository") or {}).get("default_branch")
                if db:
                    names.add(db)
            except (OSError, ValueError):
                pass
        if ref_name:
            names.add(ref_name)  # PocketDev dispatches the workflow on the default branch
        return names

    def remote_branch_exists(self, branch: str) -> bool:
        return self.ok("rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}^{{commit}}")

    def fetch_branch(self, branch: str) -> bool:
        return self.ok("fetch", "--no-tags", "--quiet", "origin",
                       f"+refs/heads/{branch}:refs/remotes/origin/{branch}", auth=True)

    def prepare_branch(self, branch: str, base: str, branch_is_new: bool) -> str:
        """Check out the feature branch. Returns 'created' or 'existing'."""
        validate_branch_name(branch)
        validate_branch_name(base)
        if not self.fetch_branch(base) and not self.remote_branch_exists(base):
            raise GitError(f"The base branch '{base}' was not found.")
        self.fetch_branch(branch)
        if self.remote_branch_exists(branch):
            self.run("checkout", "--quiet", "-B", branch, f"refs/remotes/origin/{branch}")
            return "existing"
        if not branch_is_new:
            raise GitError(f"The branch '{branch}' was not found on GitHub.")
        self.run("checkout", "--quiet", "-B", branch, f"refs/remotes/origin/{base}")
        return "created"

    # ------------------------------------------------------------------ commit / push
    def stage(self, paths: list[str]) -> list[str]:
        """Stage the given paths (new, edited or deleted). Skips gitignored and vanished-untracked paths."""
        if not paths:
            return []
        paths = sorted(set(paths))
        tracked = set(filter(None, self.run("ls-files", "-z", "--", *paths).split("\0")))
        ignored = set(filter(None, self.run("check-ignore", "-z", "--no-index", "--", *paths,
                                            check=False).split("\0")))
        keep = [p for p in paths if p in tracked or ((self.root / p).exists() and p not in ignored)]
        if keep:
            self.run("add", "-A", "--", *keep)
        return keep

    def staged_diff(self, max_chars: int = 60_000) -> str:
        return self.run("diff", "--cached", "--no-color", "--no-ext-diff")[:max_chars]

    def has_staged_changes(self) -> bool:
        p = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=self.root, env=self._env(),
                           capture_output=True, timeout=120)
        return p.returncode != 0

    def commit(self, paths: list[str], message: str) -> str | None:
        """Commit only the files the agents touched. Returns the new sha, or None if nothing changed."""
        self.stage(paths)
        if not self.has_staged_changes():
            return None
        self.run("-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}", "-c", "commit.gpgsign=false",
                 "commit", "--quiet", "--no-verify", "-m", message)
        return self.head_sha()

    def head_sha(self) -> str:
        return self.run("rev-parse", "HEAD").strip()

    def push(self, branch: str, protected: set[str]) -> None:
        ensure_push_allowed(branch, protected)
        current = self.run("rev-parse", "--abbrev-ref", "HEAD").strip()
        if current != branch:
            raise PushRefused(f"Refusing to push: checked-out branch is '{current}', expected '{branch}'.")
        self.run("push", "--no-verify", "--quiet", "origin", f"HEAD:refs/heads/{branch}", auth=True, timeout=300)

    # ------------------------------------------------------------------ diffs (from git, never from the model)
    def diff_range(self, base: str) -> str:
        return f"refs/remotes/origin/{base}...HEAD"

    def diff_stat(self, base: str) -> str:
        return self.run("diff", "--stat=100", "--no-color", "-M", self.diff_range(base))

    def diff_text(self, base: str, max_chars: int = 60_000) -> str:
        return self.run("diff", "--no-color", "--no-ext-diff", "-M", self.diff_range(base))[:max_chars]

    def files_changed(self, base: str) -> list[dict]:
        rng = self.diff_range(base)
        name_status = self.run("diff", "--name-status", "-z", "-M", rng)
        numstat = self.run("diff", "--numstat", "-z", "-M", rng)
        return parse_files_changed(name_status, numstat)


# ---------------------------------------------------------------------- pure helpers

def validate_branch_name(branch: str) -> None:
    if not branch or not BRANCH_RE.match(branch) or ".." in branch or branch.endswith((".lock", "/", ".")) \
            or "//" in branch or "@{" in branch:
        raise PushRefused(f"Invalid branch name: {branch!r}")


def ensure_push_allowed(branch: str, protected: set[str]) -> None:
    """HARD rule: never push to the base branch, the default branch, main or master."""
    validate_branch_name(branch)
    blocked = {b.strip().lower() for b in protected if b} | ALWAYS_PROTECTED
    if branch.strip().lower() in blocked or branch.lower().startswith("refs/"):
        raise PushRefused(f"Refusing to push to protected branch '{branch}'. PocketDev only pushes feature branches.")


def parse_files_changed(name_status_z: str, numstat_z: str) -> list[dict]:
    """Parse `git diff --name-status -z -M` and `git diff --numstat -z -M` output."""
    counts: dict[str, tuple[int, int]] = {}
    tokens = numstat_z.split("\0")
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.strip():
            i += 1
            continue
        parts = tok.split("\t")
        if len(parts) < 3:
            i += 1
            continue
        add, dele, path = parts[0].strip(), parts[1].strip(), parts[2]
        if path == "":  # rename/copy: next two tokens are old and new path
            path = tokens[i + 2] if i + 2 < len(tokens) else ""
            i += 3
        else:
            i += 1
        counts[path] = (int(add) if add.isdigit() else 0, int(dele) if dele.isdigit() else 0)

    out = []
    tokens = name_status_z.split("\0")
    i = 0
    while i < len(tokens):
        status = tokens[i].strip()
        if not status:
            i += 1
            continue
        code = status[0]
        if code in "RC":
            old, new = tokens[i + 1], tokens[i + 2]
            i += 3
            change = "renamed" if code == "R" else "added"
            path = new
            expl = f"Renamed from {old}" if code == "R" else f"Copied from {old}"
        else:
            path = tokens[i + 1]
            i += 2
            change = {"A": "added", "D": "deleted"}.get(code, "modified")
            expl = ""
        a, d = counts.get(path, (0, 0))
        out.append({"path": path, "change_type": change, "additions": a, "deletions": d, "explanation": expl})
    return out


_SENSITIVE_ENV = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|APIKEY|PRIVATE_KEY|CREDENTIAL)", re.I)


def _sensitive_env(name: str) -> bool:
    up = name.upper()
    return bool(_SENSITIVE_ENV.search(up)) or up.startswith(("POCKETDEV_", "ACTIONS_ID_TOKEN", "ACTIONS_RUNTIME",
                                                           "ACTIONS_CACHE_URL", "ACTIONS_RESULTS_URL"))


def sanitized_env(extra: dict | None = None) -> dict:
    """Environment for project commands (tests/builds): no tokens, keys or OIDC request credentials."""
    env = {k: v for k, v in os.environ.items() if not _sensitive_env(k)}
    env.update({"CI": "true", "GIT_TERMINAL_PROMPT": "0"})
    env.update(extra or {})
    return env


def _is_under(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)
    except ValueError:
        return False
