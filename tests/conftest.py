import json
import subprocess
from pathlib import Path

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from server_rules import validate_event, validate_plan, validate_result

GIT_ID = ["-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]


def git(cwd, *args):
    return subprocess.run(["git", *GIT_ID, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    """A working clone with a local bare 'origin' (no network). Default branch: main."""
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    work = tmp_path / "work"
    git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    seed.mkdir()
    git(seed, "init", "-b", "main")
    (seed / "app.py").write_text("def hello():\n    return 'hi'\n")
    (seed / "README.md").write_text("# Demo\nA tiny demo app.\n")
    (seed / ".env").write_text("SECRET=supersecretvalue123\n")
    (seed / "old.txt").write_text("old\n" * 5)
    git(seed, "add", "-A")
    git(seed, "commit", "-m", "init")
    git(seed, "remote", "add", "origin", str(origin))
    git(seed, "push", "origin", "main")
    git(tmp_path, "clone", "--quiet", str(origin), str(work))
    return work


class FakeToolModel(GenericFakeChatModel):
    """Scripted chat model that 'supports' tool binding (returns itself)."""

    def bind_tools(self, tools, **kwargs):
        return self


class StaticTokens:
    def __init__(self):
        self.calls = 0

    def get(self, force=False):
        self.calls += 1
        return "test-oidc-token"


class FakeServer:
    """In-memory PocketDev API for /agent/tasks/{id}... endpoints. Validates payloads like the server."""

    def __init__(self, task):
        self.task = task
        self.events, self.plans, self.results = [], [], []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-oidc-token"
        path = request.url.path
        tid = self.task["id"]
        base = f"/api/v1/agent/tasks/{tid}"
        if request.method == "GET" and path == base:
            return httpx.Response(200, json=self.task)
        body = json.loads(request.content or b"{}")
        if path == base + "/events":
            validate_event(body)
            self.events.append(body)
            return httpx.Response(204)
        if path == base + "/plan":
            validate_plan(body)
            if self.task["status"] != "planning":
                return httpx.Response(409, json={"error": {"code": "conflict", "message": "moved on"}})
            self.plans.append(body)
            self.task["status"] = "awaiting_approval"
            return httpx.Response(204)
        if path == base + "/result":
            validate_result(body)
            if self.task["status"] != "building":
                return httpx.Response(409, json={"error": {"code": "conflict", "message": "moved on"}})
            self.results.append(body)
            self.task["status"] = "completed" if body["status"] == "completed" else "failed"
            return httpx.Response(204)
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "nope"}})


def make_task(phase="build", **kw):
    task = {
        "id": "6f1c2d3e-0000-4000-8000-000000000001", "phase": phase,
        "status": "planning" if phase == "plan" else "building",
        "repo_full_name": "octo/demo", "prompt": "Add a greeting function", "base_branch": "main",
        "branch": "feature/greeting", "branch_is_new": True,
        "plan": None if phase == "plan" else {"summary": "Add greet()", "steps": ["Add greet"],
                                               "files": [{"path": "app.py", "action": "modify", "reason": "x"}]},
        "questions": [], "answers": {}, "note": "", "followup_prompt": None,
        "limits": {"max_steps": 40, "max_tokens": 400000, "max_minutes": 25, "max_fix_attempts": 3},
    }
    task.update(kw)
    return task


@pytest.fixture
def fake_server():
    def make(task):
        return FakeServer(task)
    return make
