import json
import sys

from pocketdev_agent.checks import CommandResult, Tester, failures_text, run_command

Tester.__test__ = False


def test_python_syntax_check_and_json(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "bad.py").write_text("def f(:\n")
    (tmp_path / "data.json").write_text("{bad")
    t = Tester(tmp_path, time_left=lambda: 1000)
    results = {r.name: r for r in t.run({"ok.py", "bad.py", "data.json", "gone.py"})}
    assert not results["Python syntax check"].ok
    assert not results["JSON check"].ok
    assert "bad.py" in failures_text(list(results.values()))


def test_npm_and_pytest_use_fixed_allowlisted_commands(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "jest", "build": "vite build",
                                                                   "postinstall": "curl evil"}}))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    calls = []

    def runner(argv, cwd, timeout, env=None):
        calls.append(argv)
        return CommandResult(True, "ok")

    t = Tester(tmp_path, time_left=lambda: 1000, temp_dir=str(tmp_path / "tmp"), runner=runner)
    import pocketdev_agent.checks as checks
    orig = checks.shutil.which
    checks.shutil.which = lambda name: "/usr/bin/npm"
    try:
        t.run(set())
        t.run(set())  # second round: dependencies are not reinstalled
    finally:
        checks.shutil.which = orig
    flat = [" ".join(map(str, c)) for c in calls]
    assert sum("npm install" in c for c in flat) == 1
    assert sum("pip install" in c and "pytest" in c for c in flat) == 1
    assert sum(c.endswith("npm test --silent") for c in flat) == 2
    assert sum(c.endswith("npm run build --silent") for c in flat) == 2
    assert sum("-m pytest" in c for c in flat) == 2
    assert not any("curl" in c for c in flat)


def test_skips_when_no_time(tmp_path):
    (tmp_path / "a.py").write_text("x=1\n")
    t = Tester(tmp_path, time_left=lambda: 10)
    [r] = t.run({"a.py"})
    assert r.ok and r.skipped


def test_run_command_timeout_and_tail(tmp_path):
    r = run_command([sys.executable, "-c", "import time; print('x'*10000, flush=True); time.sleep(30)"],
                    tmp_path, timeout=2)
    assert not r.ok and r.timed_out and len(r.output) < 4200
    r = run_command(["definitely-not-a-command-xyz"], tmp_path, timeout=5)
    assert not r.ok
