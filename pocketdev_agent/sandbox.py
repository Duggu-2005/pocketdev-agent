"""File access restricted to the repository working tree.

Every path the model gives us goes through Workspace.resolve(): relative only, no '..', no
escaping the repo through symlinks, nothing under .git/ or .github/workflows/, and no secret files.
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path, PurePosixPath

MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 500_000
MAX_READ_LINES = 400

# Directories never shown in listings/searches (noise, not security).
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".gradle", "build", "dist", ".idea",
             ".next", ".nuxt", "target", ".dart_tool", ".pytest_cache", ".mypy_cache", "Pods", ".tox"}

# Lower-cased basename globs the agents may never read or write.
SECRET_FILE_GLOBS = [
    ".env", ".env.*", "*.env", ".envrc", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.p8",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", "*.ppk", ".npmrc", ".pypirc", ".netrc", "_netrc",
    ".git-credentials", ".dockercfg", "credentials", "credentials.json", "*.credentials.json",
    "service-account*.json", "secrets.json", "secrets.yml", "secrets.yaml", "*.secret", "*.secrets",
    "local.properties", "keystore.properties", "*.tfvars", "*.tfstate", "htpasswd", ".htpasswd",
]
# Lower-cased directory prefixes (posix, relative to repo root) that are off-limits.
BLOCKED_DIR_PREFIXES = [".git/", ".github/workflows/", ".ssh/", ".aws/", ".gnupg/", ".docker/"]


class SandboxError(ValueError):
    """A path or operation the agent is not allowed to perform. Message is safe to show the model."""


def is_secret_name(name: str) -> bool:
    low = name.lower()
    return any(fnmatch.fnmatchcase(low, g) for g in SECRET_FILE_GLOBS)


def blocked_reason(rel_posix: str) -> str | None:
    low = rel_posix.lower().strip("/")
    if low in (".git", ".github/workflows") or any((low + "/").startswith(p) for p in BLOCKED_DIR_PREFIXES):
        return "This location is protected and can't be read or changed by the AI."
    for part in PurePosixPath(low).parts:
        if part == ".git":
            return "This location is protected and can't be read or changed by the AI."
    if is_secret_name(PurePosixPath(low).name):
        return "This looks like a secrets file, so the AI is not allowed to open or change it."
    return None


class Workspace:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(os.path.realpath(root))
        self.touched: set[str] = set()  # repo-relative posix paths written/deleted by the agent

    # ------------------------------------------------------------------ path rules
    def resolve(self, path: str, *, for_write: bool = False) -> tuple[Path, str]:
        """Return (absolute path, repo-relative posix path) or raise SandboxError."""
        if not isinstance(path, str) or (for_write and not path.strip()):
            raise SandboxError("A file path is required.")
        p = path.strip().replace("\\", "/")
        if "\x00" in p or len(p) > 500:
            raise SandboxError("Invalid path.")
        if p.startswith("/") or p.startswith("~") or re.match(r"^[A-Za-z]:", p):
            raise SandboxError("Use a path relative to the project folder (absolute paths are not allowed).")
        parts = [x for x in p.split("/") if x not in ("", ".")]
        if any(x == ".." for x in parts):
            raise SandboxError("Paths may not contain '..'.")
        rel = "/".join(parts)
        if not rel:
            if for_write:
                raise SandboxError("A file path is required.")
            return self.root, ""
        reason = blocked_reason(rel)
        if reason:
            raise SandboxError(reason)
        target = self.root.joinpath(*parts)
        # Resolve symlinks (for new files: the deepest existing parent) and make sure we stay inside.
        real = Path(os.path.realpath(target))
        if not _is_within(real, self.root):
            raise SandboxError("That path points outside the project folder.")
        real_rel = real.relative_to(self.root).as_posix() if real != self.root else ""
        if real_rel and real_rel != rel:
            reason = blocked_reason(real_rel)
            if reason:
                raise SandboxError(reason)
        if for_write and target.is_symlink():
            raise SandboxError("Writing through symbolic links is not allowed.")
        return target, rel

    # ------------------------------------------------------------------ operations
    def read_file(self, path: str, start_line: int = 1, max_lines: int = MAX_READ_LINES) -> str:
        target, rel = self.resolve(path)
        if not target.is_file():
            raise SandboxError(f"File not found: {rel}")
        size = target.stat().st_size
        with open(target, "rb") as fh:
            data = fh.read(MAX_READ_BYTES)
        if b"\x00" in data[:8192]:
            raise SandboxError(f"{rel} is a binary file ({size} bytes) and can't be shown as text.")
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        start = max(1, int(start_line or 1))
        count = max(1, min(int(max_lines or MAX_READ_LINES), MAX_READ_LINES))
        chunk = lines[start - 1:start - 1 + count]
        body = "\n".join(f"{i:>5}| {line}" for i, line in enumerate(chunk, start=start))
        end = start + len(chunk) - 1
        note = f"[{rel}: lines {start}-{end} of {len(lines)}"
        if size > MAX_READ_BYTES:
            note += f"; file is {size} bytes, only the first {MAX_READ_BYTES} were loaded"
        if end < len(lines):
            note += f"; call read_file again with start_line={end + 1} for more"
        return note + "]\n" + body

    def write_file(self, path: str, content: str) -> str:
        target, rel = self.resolve(path, for_write=True)
        data = (content or "").encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise SandboxError(f"Content is too large ({len(data)} bytes, max {MAX_WRITE_BYTES}).")
        if target.is_dir():
            raise SandboxError(f"{rel} is a folder.")
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        # The parent might have been a symlink created between checks; re-check after mkdir.
        if not _is_within(Path(os.path.realpath(target.parent)), self.root):
            raise SandboxError("That path points outside the project folder.")
        with open(target, "wb") as fh:
            fh.write(data)
        self.touched.add(rel)
        return f"{'Updated' if existed else 'Created'} {rel} ({len(data)} bytes)."

    def replace_in_file(self, path: str, old: str, new: str) -> str:
        target, rel = self.resolve(path, for_write=True)
        if not target.is_file():
            raise SandboxError(f"File not found: {rel}")
        if not old:
            raise SandboxError("old_text must not be empty.")
        text = target.read_text(encoding="utf-8", errors="replace")
        n = text.count(old)
        if n == 0:
            raise SandboxError("old_text was not found in the file. Read the file again and copy the exact text.")
        if n > 1:
            raise SandboxError(f"old_text appears {n} times; include more surrounding lines so it is unique.")
        return self.write_file(rel, text.replace(old, new, 1)).replace("Updated", "Edited")

    def delete_file(self, path: str) -> str:
        target, rel = self.resolve(path, for_write=True)
        if not target.is_file():
            raise SandboxError(f"File not found: {rel}")
        target.unlink()
        self.touched.add(rel)
        return f"Deleted {rel}."

    def list_dir(self, path: str = "") -> str:
        target, rel = self.resolve(path or "")
        if not target.is_dir():
            raise SandboxError(f"Folder not found: {rel or '.'}")
        entries = []
        for child in sorted(target.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            child_rel = f"{rel}/{child.name}" if rel else child.name
            if child.name in SKIP_DIRS or blocked_reason(child_rel):
                continue
            entries.append(child.name + ("/" if child.is_dir() else ""))
            if len(entries) >= 200:
                entries.append("... (more entries not shown)")
                break
        return f"[{rel or '.'}]\n" + ("\n".join(entries) if entries else "(empty)")

    def iter_files(self, limit: int = 5000):
        """Yield repo-relative posix paths of readable files (skipping noise and blocked files)."""
        n = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not os.path.islink(os.path.join(dirpath, d)))
            base = Path(dirpath).relative_to(self.root).as_posix()
            for fn in sorted(filenames):
                rel = fn if base == "." else f"{base}/{fn}"
                if blocked_reason(rel):
                    continue
                yield rel
                n += 1
                if n >= limit:
                    return

    def search_text(self, query: str, path: str = "", max_results: int = 50) -> str:
        if not query or len(query) > 200:
            raise SandboxError("Search text must be 1-200 characters.")
        base_target, base_rel = self.resolve(path or "")
        needle = query.lower()
        results = []
        for rel in self.iter_files():
            if base_rel and not (rel == base_rel or rel.startswith(base_rel + "/")):
                continue
            full = self.root / rel
            try:
                if full.is_symlink() or full.stat().st_size > MAX_READ_BYTES:
                    continue
                with open(full, "rb") as fh:
                    data = fh.read(MAX_READ_BYTES)
            except OSError:
                continue
            if b"\x00" in data[:4096]:
                continue
            for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1):
                if needle in line.lower():
                    results.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(results) >= max_results:
                        return "\n".join(results) + "\n... (more matches not shown)"
        return "\n".join(results) if results else "No matches."


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:  # different drives on Windows
        return False
