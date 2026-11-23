import os

import pytest

from conftest import git
from pocketdev_agent.gitops import (GitError, GitRepo, PushRefused, ensure_push_allowed, parse_files_changed,
                                    sanitized_env)


@pytest.mark.parametrize("branch", ["main", "MAIN", "master", "develop", "release", "-f", "a..b", "refs/heads/x",
                                    "", "feature/x.lock", "x@{1}"])
def test_push_guard_refuses(branch):
    with pytest.raises(PushRefused):
        ensure_push_allowed(branch, {"develop", "release"})


def test_push_guard_allows_feature_branch():
    ensure_push_allowed("feature/login-page", {"main", "develop"})


def test_push_refuses_base_and_default_branch_on_real_repo(repo):
    g = GitRepo(repo)
    assert "main" in g.default_branches()
    with pytest.raises(PushRefused):
        g.push("main", {"main"})
    git(repo, "checkout", "-b", "trunk")
    with pytest.raises(PushRefused):
        g.push("trunk", {"trunk"})  # base branch
    # wrong checked-out branch is refused too
    with pytest.raises(PushRefused):
        g.push("feature/other", {"main"})


def test_prepare_branch_commit_push_and_files_changed(repo, tmp_path):
    g = GitRepo(repo)
    assert g.prepare_branch("feature/x", "main", True) == "created"
    (repo / "app.py").write_text("def hello():\n    return 'hello'\n\ndef bye():\n    return 'bye'\n")
    (repo / "new.txt").write_text("a\nb\n")
    (repo / "old.txt").unlink()
    git(repo, "mv", "README.md", "DOCS.md")
    git(repo, "reset", "-q")  # unstage; commit() stages only the paths we pass
    (repo / "logo.bin").write_bytes(b"\x00\x01\x02")
    sha = g.commit(["app.py", "new.txt", "old.txt", "README.md", "DOCS.md", "logo.bin", "ghost.txt"], "test")
    assert sha and len(sha) == 40
    files = {f["path"]: f for f in g.files_changed("main")}
    assert files["app.py"]["change_type"] == "modified"
    assert (files["app.py"]["additions"], files["app.py"]["deletions"]) == (4, 1)
    assert files["new.txt"] == {"path": "new.txt", "change_type": "added", "additions": 2, "deletions": 0,
                                "explanation": ""}
    assert files["old.txt"]["change_type"] == "deleted" and files["old.txt"]["deletions"] == 5
    assert files["DOCS.md"]["change_type"] == "renamed" and "README.md" in files["DOCS.md"]["explanation"]
    assert files["logo.bin"]["additions"] == 0  # binary: '-' in numstat
    g.push("feature/x", {"main"})
    origin = tmp_path / "origin.git"
    assert git(origin, "rev-parse", "refs/heads/feature/x").strip() == sha
    assert git(origin, "rev-parse", "refs/heads/main").strip() != sha
    # follow-up run continues on the existing remote branch
    git(repo, "checkout", "-q", "main")
    assert g.prepare_branch("feature/x", "main", False) == "existing"
    assert g.head_sha() == sha
    assert g.commit(["app.py"], "nothing new") is None


def test_prepare_branch_missing(repo):
    g = GitRepo(repo)
    with pytest.raises(GitError):
        g.prepare_branch("feature/nope", "main", False)
    with pytest.raises(GitError):
        g.prepare_branch("feature/nope", "no-such-base", True)


def test_parse_files_changed_z_format():
    ns = "M\0a.py\0A\0b c.txt\0D\0gone.txt\0R087\0old/x.py\0new/x.py\0C100\0a.py\0copy.py\0T\0link\0"
    num = "3\t1\ta.py\0" "2\t0\tb c.txt\0" "0\t4\tgone.txt\0" "1\t1\t\0old/x.py\0new/x.py\0" "0\t0\tcopy.py\0-\t-\tlink\0"
    out = {f["path"]: f for f in parse_files_changed(ns, num)}
    assert out["a.py"]["change_type"] == "modified" and out["a.py"]["additions"] == 3
    assert out["b c.txt"]["change_type"] == "added"
    assert out["gone.txt"]["deletions"] == 4
    assert out["new/x.py"]["change_type"] == "renamed" and out["new/x.py"]["additions"] == 1
    assert out["copy.py"]["change_type"] == "added"
    assert out["link"]["change_type"] == "modified" and out["link"]["additions"] == 0


def test_secure_credentials_strips_persisted_token(repo, tmp_path):
    key = "http.https://github.com/.extraheader"
    git(repo, "config", "--local", key, "AUTHORIZATION: basic c2VjcmV0")
    cred = tmp_path / "runner_temp" / "git-credentials-123.config"
    cred.parent.mkdir()
    cred.write_text(f"[http \"https://github.com/\"]\n\textraheader = AUTHORIZATION: basic c2VjcmV0\n")
    git(repo, "config", "--local", f"includeIf.gitdir:{repo.as_posix()}/.git.path", str(cred))
    g = GitRepo(repo, github_token="ghs_exampletoken")
    g.secure_credentials(str(tmp_path / "runner_temp"))
    cfg = (repo / ".git" / "config").read_text()
    assert "extraheader" not in cfg and "git-credentials" not in cfg
    assert not cred.exists()
    assert g._auth_header.startswith("AUTHORIZATION: basic ")


def test_sanitized_env_drops_secrets(monkeypatch):
    for k in ("GITHUB_TOKEN", "POCKETDEV_GEMINI_API_KEY", "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
              "ACTIONS_ID_TOKEN_REQUEST_URL", "MY_PASSWORD", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(k, "x")
    env = sanitized_env()
    assert not any(k in env for k in ("GITHUB_TOKEN", "POCKETDEV_GEMINI_API_KEY", "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
                                      "ACTIONS_ID_TOKEN_REQUEST_URL", "MY_PASSWORD", "AWS_SECRET_ACCESS_KEY"))
    assert env["CI"] == "true" and "PATH" in {k.upper() for k in env}
