import json

import httpx
from langchain_core.messages import AIMessage

from conftest import FakeToolModel, StaticTokens, git, make_task
from pocketdev_agent.api import ApiClient
from pocketdev_agent.config import NO_KEY_MESSAGE, Settings
from pocketdev_agent.runner import run


def settings_for(repo, phase):
    return Settings(api_base="https://api.example/api/v1", task_id="6f1c2d3e-0000-4000-8000-000000000001",
                    phase=phase, repo_dir=repo, github_token="ghs_faketoken1234567890",
                    temp_dir=str(repo.parent / "runner_temp"))


def api_for(server):
    return ApiClient("https://api.example/api/v1", server.task["id"], StaticTokens(),
                     transport=httpx.MockTransport(server.handler), sleep=lambda s: None)


def tool_call(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


def fake(*replies):
    return lambda settings: [FakeToolModel(messages=iter(replies))]


def test_plan_phase(repo, fake_server):
    server = fake_server(make_task("plan", prompt="Make a login page"))
    plan = {
        "summary": "Add a simple login page.", "steps": ["Create the page", "Add checks"],
        "files": [{"path": "login.html", "action": "create", "reason": "The new page"}],
        "questions": [{"id": "q1", "text": "Which colours do you like?", "kind": "palette", "options": [
            {"id": "o1", "label": "Ocean", "description": "Calm blues", "colors": ["#0EA5E9", "#0F172A", "#F8FAFC"]},
            {"id": "o2", "label": "Forest", "colors": ["#16A34A", "#14532D", "#F0FDF4"]},
            {"id": "o3", "label": "Sunset", "colors": ["#F97316", "#7C2D12", "#FFF7ED"]}]}],
    }
    models = fake(
        AIMessage("", tool_calls=[tool_call("list_dir", {"path": ""}, 1), tool_call("read_file", {"path": ".env"}, 2)]),
        AIMessage("", tool_calls=[tool_call("write_file", {"path": "x.txt", "content": "no"}, 3)]),  # not allowed
        AIMessage("Here is the plan:\n```json\n" + json.dumps(plan) + "\n```"),
    )
    assert run(settings_for(repo, "plan"), api_for(server), models) == 0
    assert len(server.plans) == 1
    body = server.plans[0]
    assert body["plan"]["summary"] == "Add a simple login page."
    assert body["questions"][0]["kind"] == "palette" and len(body["questions"][0]["options"]) == 3
    assert not (repo / "x.txt").exists()  # planner has read-only tools
    assert server.task["status"] == "awaiting_approval"
    assert any(e["agent"] == "planner" for e in server.events)
    assert not any("SECRET" in e["message"] for e in server.events)


def test_build_phase_end_to_end_with_fix_loop(repo, fake_server, tmp_path):
    server = fake_server(make_task("build"))
    new_app = "def hello():\n    return 'hi'\n\n\ndef greet(name):\n    return f'Hello, {name}!'\n"
    models = fake(
        # coder round 1: read, then write broken code, try forbidden paths
        AIMessage("", tool_calls=[tool_call("read_file", {"path": "app.py"}, 1)]),
        AIMessage("", tool_calls=[tool_call("write_file", {"path": "app.py", "content": "def greet(:\n"}, 2),
                                  tool_call("write_file", {"path": "../escape.txt", "content": "x"}, 3),
                                  tool_call("read_file", {"path": ".env"}, 4),
                                  tool_call("write_file", {"path": ".github/workflows/x.yml", "content": "x"}, 5)]),
        AIMessage("Added greet()."),
        # tester fails (syntax) -> coder round 2 fixes it
        AIMessage("", tool_calls=[tool_call("write_file", {"path": "app.py", "content": new_app}, 6)]),
        AIMessage("Fixed the syntax error."),
        # reviewer
        AIMessage('{"approved": true, "issues": []}'),
        # summarizer
        AIMessage('{"overview": "Added a **greet** function.", "files": {"app.py": "Adds greet(name)."}}'),
    )
    assert run(settings_for(repo, "build"), api_for(server), models) == 0
    assert len(server.results) == 1
    res = server.results[0]
    assert res["status"] == "completed", res
    assert res["files_changed"] == [{"path": "app.py", "change_type": "modified", "additions": 4, "deletions": 0,
                                     "explanation": "Adds greet(name)."}]
    assert "greet" in res["summary"] and "`app.py`" in res["summary"]
    origin = tmp_path / "origin.git"
    assert git(origin, "rev-parse", "refs/heads/feature/greeting").strip() == res["commit_sha"]
    assert "greet" not in git(origin, "show", "main:app.py")  # main untouched
    assert "PocketDev AI" in git(origin, "log", "-1", "--format=%an", "feature/greeting")
    assert not (tmp_path / "escape.txt").exists()
    assert not (repo / ".github").exists()
    msgs = " | ".join(e["message"] for e in server.events)
    assert "Some checks failed" in msgs and "All checks passed" in msgs
    assert "supersecretvalue" not in msgs and "ghs_faketoken" not in msgs


def test_build_no_changes_reports_failure(repo, fake_server):
    server = fake_server(make_task("build"))
    models = fake(AIMessage("I looked but nothing needs changing."))
    assert run(settings_for(repo, "build"), api_for(server), models) == 1
    assert server.results[0]["status"] == "failed"
    assert "didn't change any files" in server.results[0]["error"]


def test_build_refuses_branch_equal_to_base(repo, fake_server, tmp_path):
    server = fake_server(make_task("build", branch="main"))
    models = fake(AIMessage("", tool_calls=[tool_call("write_file", {"path": "a.txt", "content": "x"}, 1)]),
                  AIMessage("done"))
    assert run(settings_for(repo, "build"), api_for(server), models) == 1
    assert server.results[0]["status"] == "failed"
    assert "never to the main branch" in server.results[0]["error"]
    assert git(tmp_path / "origin.git", "rev-parse", "main").strip() == git(repo, "rev-parse", "origin/main").strip()


def test_secret_in_change_blocks_push(repo, fake_server, tmp_path):
    server = fake_server(make_task("build"))
    leaked = "API_KEY = 'AIza" + "x" * 35 + "'\n"
    models = fake(
        AIMessage("", tool_calls=[tool_call("write_file", {"path": "config.py", "content": leaked}, 1)]),
        AIMessage("done"),
        AIMessage('{"approved": true, "issues": []}'),
        # reviewer's secret scan sends it back; coder ignores it
        AIMessage("I kept it."),
        AIMessage('{"approved": true, "issues": []}'),  # second (final) review
    )
    assert run(settings_for(repo, "build"), api_for(server), models) == 1
    assert server.results[0]["status"] == "failed"
    assert "secret" in server.results[0]["error"]
    assert "feature/greeting" not in git(tmp_path / "origin.git", "branch", "--list")


def test_no_api_key(repo, fake_server):
    server = fake_server(make_task("build"))
    assert run(settings_for(repo, "build"), api_for(server), lambda s: []) == 1
    assert server.results[0]["error"] == NO_KEY_MESSAGE
    plan_server = fake_server(make_task("plan"))
    assert run(settings_for(repo, "plan"), api_for(plan_server), lambda s: []) == 1
    assert plan_server.events[-1] == {"type": "error", "agent": "planner", "message": NO_KEY_MESSAGE}


def test_cancelled_task_stops(repo, fake_server):
    server = fake_server(make_task("build", status="cancelled"))
    assert run(settings_for(repo, "build"), api_for(server), fake()) == 0
    assert not server.results and not server.events


def test_cancelled_mid_run_stops_without_result(repo, fake_server):
    server = fake_server(make_task("build"))

    def cancel_then_reply():
        server.task["status"] = "cancelled"
        yield AIMessage("", tool_calls=[tool_call("write_file", {"path": "a.txt", "content": "x"}, 1)])
        yield AIMessage("done")

    models = lambda s: [FakeToolModel(messages=cancel_then_reply())]
    assert run(settings_for(repo, "build"), api_for(server), models) == 0
    assert not server.results


def test_step_limit_wraps_up_honestly(repo, fake_server):
    task = make_task("build")
    task["limits"]["max_steps"] = 2
    server = fake_server(task)
    replies = [AIMessage("", tool_calls=[tool_call("write_file", {"path": f"f{i}.txt", "content": "x\n"}, i)])
               for i in range(5)]
    models = fake(*replies, AIMessage('{"overview": "Added files.", "files": {}}'))
    assert run(settings_for(repo, "build"), api_for(server), models) == 0
    res = server.results[0]
    assert res["status"] == "completed"
    assert len(res["files_changed"]) == 2
    assert "stopped early" in res["summary"]


def test_followup_continues_existing_branch(repo, fake_server, tmp_path):
    git(repo, "checkout", "-q", "-b", "feature/greeting")
    (repo / "first.txt").write_text("first\n")
    git(repo, "add", "first.txt")
    git(repo, "commit", "-q", "-m", "first")
    git(repo, "push", "-q", "origin", "feature/greeting")
    git(repo, "checkout", "-q", "main")
    server = fake_server(make_task("build", branch_is_new=False, followup_prompt="Also add second.txt"))
    models = fake(AIMessage("", tool_calls=[tool_call("write_file", {"path": "second.txt", "content": "2\n"}, 1)]),
                  AIMessage("done"), AIMessage('{"approved": true}'), AIMessage("not json at all"),
                  AIMessage("still not json"))
    assert run(settings_for(repo, "build"), api_for(server), models) == 0
    res = server.results[0]
    assert {f["path"] for f in res["files_changed"]} == {"first.txt", "second.txt"}
    assert "Changed 2 file(s)" in res["summary"]  # deterministic fallback summary from git stats
    log = git(tmp_path / "origin.git", "log", "--format=%s", "feature/greeting")
    assert "first" in log and "PocketDev: Also add second.txt" in log
