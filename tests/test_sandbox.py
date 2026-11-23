import os

import pytest

from pocketdev_agent.sandbox import SandboxError, Workspace
from pocketdev_agent.tools import make_tools


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("print('x')\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text("on: push\n")
    (root / ".env").write_text("KEY=1\n")
    (tmp_path / "outside.txt").write_text("outside secret\n")
    return Workspace(root)


@pytest.mark.parametrize("path", ["../outside.txt", "src/../../outside.txt", "/etc/passwd", "C:/Windows/win.ini",
                                  "~/.ssh/id_rsa", "..\\outside.txt", "a/\x00b"])
def test_escapes_rejected(ws, path):
    with pytest.raises(SandboxError):
        ws.read_file(path)
    with pytest.raises(SandboxError):
        ws.write_file(path, "x")


@pytest.mark.parametrize("path", [".git/config", ".git", "sub/.git/HEAD", ".github/workflows/ci.yml",
                                  ".github/workflows/new.yml", ".GITHUB/Workflows/x.yml"])
def test_protected_locations(ws, path):
    with pytest.raises(SandboxError):
        ws.write_file(path, "x")
    with pytest.raises(SandboxError):
        ws.read_file(path)


@pytest.mark.parametrize("name", [".env", ".env.local", ".env.production", "prod.env", "server.pem", "id_rsa",
                                  "id_rsa.pub", "id_ed25519", "config/secrets.json", "release.keystore", "app.jks",
                                  ".npmrc", "local.properties", "certs/tls.key", ".ENV"])
def test_secret_files_refused(ws, name):
    with pytest.raises(SandboxError):
        ws.read_file(name)
    with pytest.raises(SandboxError):
        ws.write_file(name, "x")


def test_secret_files_hidden_from_listing_and_search(ws):
    assert ".env" not in ws.list_dir("")
    assert ".git/" not in ws.list_dir("")
    assert "KEY=1" not in ws.search_text("KEY")


def test_symlink_escape_rejected(ws, tmp_path):
    try:
        os.symlink(tmp_path / "outside.txt", ws.root / "link.txt")
    except OSError:
        pytest.skip("this OS user may not create symlinks (Windows without developer mode)")
    os.symlink(tmp_path, ws.root / "linkdir")
    os.symlink(ws.root / ".git" / "config", ws.root / "gitcfg")
    for p in ("link.txt", "linkdir/outside.txt", "gitcfg"):
        with pytest.raises(SandboxError):
            ws.read_file(p)
    with pytest.raises(SandboxError):
        ws.write_file("linkdir/new.txt", "x")


def test_normal_operations_and_touched(ws):
    assert "Created" in ws.write_file("src/new/util.py", "def f():\n    return 1\n")
    assert "Edited" in ws.replace_in_file("src/main.py", "print('x')", "print('y')")
    assert "print('y')" in ws.read_file("./src/main.py")
    with pytest.raises(SandboxError):
        ws.replace_in_file("src/main.py", "not there", "z")
    assert "src/main.py:1" in ws.search_text("PRINT")
    ws.delete_file("src/new/util.py")
    assert ws.touched == {"src/new/util.py", "src/main.py"}


def test_read_paging_and_size_cap(ws):
    ws.write_file("big.txt", "\n".join(f"line {i}" for i in range(1000)))
    out = ws.read_file("big.txt")
    assert "lines 1-400 of 1000" in out and "start_line=401" in out
    assert "line 999" in ws.read_file("big.txt", start_line=900)
    with pytest.raises(SandboxError):
        ws.write_file("huge.txt", "x" * 600_000)


def test_tools_wrap_sandbox(ws):
    tools = {t.name: t for t in make_tools(ws)}
    assert set(tools) == {"read_file", "list_dir", "search_text", "write_file", "replace_in_file", "delete_file"}
    ro = {t.name for t in make_tools(ws, read_only=True)}
    assert ro == {"read_file", "list_dir", "search_text"}
    with pytest.raises(SandboxError):
        tools["read_file"].invoke({"path": ".env"})
