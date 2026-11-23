"""Tester: detects the project type and runs ALLOW-LISTED check commands only.

Commands are fixed argv lists built here (never from model output, never shell=True). They run with a
sanitized environment (no GitHub token, AI keys or OIDC credentials), a timeout, and truncated output.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pocketdev_agent.gitops import sanitized_env

OUTPUT_TAIL = 4000
INSTALL_TIMEOUT = 420
CHECK_TIMEOUT = 600
GRADLE_MIN_SECONDS = 600  # only try a Gradle build when at least this much time is left
NPM_PLACEHOLDER_TEST = "no test specified"


@dataclass
class CommandResult:
    ok: bool
    output: str
    timed_out: bool = False


@dataclass
class CheckResult:
    name: str
    ok: bool
    output: str = ""
    skipped: str = ""  # reason, when the check could not run (not counted as a failure)


def run_command(argv: list[str], cwd: Path, timeout: float, env: dict | None = None) -> CommandResult:
    """Run an allow-listed command; kill the whole process group on timeout."""
    kwargs = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(argv, cwd=cwd, env=env if env is not None else sanitized_env(),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    except OSError as e:
        return CommandResult(False, f"could not start {argv[0]}: {e.strerror}")
    try:
        out, _ = proc.communicate(timeout=max(1, timeout))
        return CommandResult(proc.returncode == 0, _tail(out))
    except subprocess.TimeoutExpired:
        _kill(proc)
        out, _ = proc.communicate()
        return CommandResult(False, _tail(out) + f"\n[stopped after {int(timeout)} seconds]", timed_out=True)


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        pass


def _tail(data: bytes | None) -> str:
    text = (data or b"").decode("utf-8", errors="replace")
    return text if len(text) <= OUTPUT_TAIL else "...\n" + text[-OUTPUT_TAIL:]


class Tester:
    def __init__(self, root: Path, *, time_left: Callable[[], float], temp_dir: str = "",
                 runner: Callable[..., CommandResult] = run_command, redact: Callable[[str], str] = lambda s: s,
                 on_progress: Callable[[str], None] = lambda s: None):
        self.root = Path(root)
        self.time_left = time_left
        self.temp_dir = Path(temp_dir) if temp_dir else Path(os.environ.get("RUNNER_TEMP") or self.root.parent)
        self.runner = runner
        self.redact = redact
        self.on_progress = on_progress
        self._install_keys: dict[str, str] = {}
        self._install_ok: dict[str, bool] = {}

    # ------------------------------------------------------------------ helpers
    def _timeout(self, cap: float) -> float:
        return max(0.0, min(cap, self.time_left() - 30))

    def _run(self, name: str, argv: list[str], cap: float, env: dict | None = None) -> CheckResult:
        t = self._timeout(cap)
        if t < 20:
            return CheckResult(name, True, skipped="not enough time left")
        self.on_progress(f"Running {name}")
        r = self.runner(argv, self.root, t, env)
        return CheckResult(name, r.ok, self.redact(r.output))

    def _manifest_key(self, *files: str) -> str:
        h = hashlib.sha256()
        for f in files:
            p = self.root / f
            if p.is_file():
                h.update(f.encode() + b"\0" + p.read_bytes())
        return h.hexdigest()

    def _install_once(self, kind: str, key: str, steps: list[tuple[list[str], dict | None]]) -> CheckResult | None:
        """Run install steps unless the same manifests were already installed. Returns a failure or None."""
        if self._install_keys.get(kind) == key:
            return None if self._install_ok.get(kind) else CheckResult(f"{kind} install", True,
                                                                       skipped="dependencies could not be installed")
        self._install_keys[kind] = key
        for argv, env in steps:
            r = self._run(f"{kind} install", argv, INSTALL_TIMEOUT, env)
            if r.skipped or not r.ok:
                self._install_ok[kind] = False
                # An install problem is usually not the AI's fault; report it but don't count it as a failure.
                return CheckResult(f"{kind} install", True, skipped="dependencies could not be installed")
        self._install_ok[kind] = True
        return None

    # ------------------------------------------------------------------ checks
    def run(self, touched: set[str] | list[str]) -> list[CheckResult]:
        touched = sorted(set(touched))
        results: list[CheckResult] = []
        results += self._syntax_checks(touched)
        results += self._python_tests()
        results += self._npm_checks()
        results += self._gradle_checks()
        return results

    def _syntax_checks(self, touched: list[str]) -> list[CheckResult]:
        out = []
        py = [p for p in touched if p.endswith(".py") and (self.root / p).is_file()]
        if py:
            out.append(self._run("Python syntax check", [sys.executable, "-m", "py_compile", *py], 120))
        bad = []
        for p in touched:
            if p.endswith(".json") and (self.root / p).is_file():
                try:
                    json.loads((self.root / p).read_text(encoding="utf-8"))
                except (ValueError, UnicodeDecodeError) as e:
                    bad.append(f"{p}: {e}")
        if any(p.endswith(".json") for p in touched):
            out.append(CheckResult("JSON check", not bad, "\n".join(bad)))
        return out

    def _has_python_tests(self) -> bool:
        for d in ("tests", "test"):
            if (self.root / d).is_dir() and any((self.root / d).rglob("*.py")):
                return True
        for p in list(self.root.glob("test_*.py")) + list(self.root.glob("*_test.py")):
            if p.is_file():
                return True
        return False

    def _python_tests(self) -> list[CheckResult]:
        if not self._has_python_tests():
            return []
        venv = self.temp_dir / "pocketdev-pyenv"
        vpy = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        manifests = [f for f in ("requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.py")
                     if (self.root / f).is_file()]
        steps: list[tuple[list[str], dict | None]] = []
        if not vpy.exists():
            steps.append(([sys.executable, "-m", "venv", str(venv)], None))
        pip = [str(vpy), "-m", "pip", "install", "--disable-pip-version-check", "--quiet"]
        steps.append((pip + ["pytest"], None))
        for req in ("requirements.txt", "requirements-dev.txt"):
            if (self.root / req).is_file():
                steps.append((pip + ["-r", req], None))
        if not any(r.startswith("requirements") for r in manifests) and \
                any(m in manifests for m in ("pyproject.toml", "setup.py")):
            steps.append((pip + ["-e", "."], None))
        failed = self._install_once("Python", self._manifest_key(*manifests), steps)
        if failed:
            return [failed]
        return [self._run("Python tests (pytest)", [str(vpy), "-m", "pytest", "-q", "-x", "--no-header",
                                                     "-p", "no:cacheprovider"], CHECK_TIMEOUT)]

    def _npm_checks(self) -> list[CheckResult]:
        pkg_file = self.root / "package.json"
        if not pkg_file.is_file():
            return []
        try:
            scripts = json.loads(pkg_file.read_text(encoding="utf-8")).get("scripts") or {}
        except (ValueError, UnicodeDecodeError, AttributeError) as e:
            return [CheckResult("package.json check", False, f"package.json is not valid JSON: {e}")]
        if not isinstance(scripts, dict):
            scripts = {}
        has_test = isinstance(scripts.get("test"), str) and NPM_PLACEHOLDER_TEST not in scripts["test"]
        has_build = isinstance(scripts.get("build"), str)
        if not (has_test or has_build):
            return []
        npm = shutil.which("npm")
        if not npm:
            return [CheckResult("npm checks", True, skipped="npm is not installed")]
        if (self.root / "package-lock.json").is_file():
            install = [npm, "ci", "--no-audit", "--no-fund"]
        elif (self.root / "yarn.lock").is_file() or (self.root / "pnpm-lock.yaml").is_file():
            return [CheckResult("npm checks", True, skipped="yarn/pnpm projects are not checked yet")]
        else:
            install = [npm, "install", "--no-audit", "--no-fund", "--no-package-lock"]
        failed = self._install_once("npm", self._manifest_key("package.json", "package-lock.json"),
                                    [(install, None)])
        if failed:
            return [failed]
        out = []
        if has_test:
            out.append(self._run("npm test", [npm, "test", "--silent"], CHECK_TIMEOUT))
        if has_build:
            out.append(self._run("npm run build", [npm, "run", "build", "--silent"], CHECK_TIMEOUT))
        return out

    def _gradle_checks(self) -> list[CheckResult]:
        if not (self.root / "gradlew").is_file():
            return []
        is_android = any((self.root / d).is_file() for d in ("app/build.gradle", "app/build.gradle.kts"))
        if not is_android:
            return []
        if self.time_left() < GRADLE_MIN_SECONDS:
            return [CheckResult("Android build", True, skipped="not enough time left for an Android build")]
        if os.name != "posix":
            return [CheckResult("Android build", True, skipped="Android builds only run on Linux runners")]
        return [self._run("Android build (assembleDebug)",
                          ["sh", "./gradlew", "assembleDebug", "--no-daemon", "--console=plain", "-q"],
                          min(1200, self.time_left() - 120))]


def failures_text(results: list[CheckResult]) -> str:
    parts = []
    for r in results:
        if not r.ok and not r.skipped:
            parts.append(f"### {r.name} failed\n<check_output>\n{r.output.strip()[-OUTPUT_TAIL:]}\n</check_output>")
    return "\n\n".join(parts)


def results_summary(results: list[CheckResult]) -> list[str]:
    lines = []
    for r in results:
        if r.skipped:
            lines.append(f"{r.name}: skipped ({r.skipped})")
        else:
            lines.append(f"{r.name}: {'passed' if r.ok else 'failed'}")
    return lines
