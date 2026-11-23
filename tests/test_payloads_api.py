import json

import httpx
import pytest

from pocketdev_agent.api import ApiClient, ApiError, OIDCTokenProvider, TaskCancelled, TaskConflict
from pocketdev_agent.limits import Budget
from pocketdev_agent.payloads import extract_json, normalize_plan, normalize_questions, result_payload
from pocketdev_agent.redact import Redactor
from server_rules import validate_plan, validate_result


# ---------------------------------------------------------------------- payloads

def test_messy_plan_is_normalized_to_server_rules():
    raw = {
        "summary": "x" * 5000,
        "steps": ["s" * 900] * 50 + [None, ""],
        "files": [{"path": "./a.py", "action": "edit", "reason": "r" * 900}, {"path": "a.py"}, {"nope": 1}]
                 + [{"path": f"f{i}.py", "action": "create"} for i in range(150)],
        "questions": [
            {"id": "q 1!", "text": "Colours?", "kind": "palette", "options": [
                {"id": "o1", "label": "Ocean", "colors": ["#0EA5E9", "#0F172A", "blue", "#FFF"]},
                {"id": "o1", "label": "Forest", "colors": "#16A34A,#14532D"},
                {"id": "o3", "label": "No colours", "colors": []},
                {"label": "L" * 200, "colors": ["#000000"] * 12}]},
            {"id": "q2", "text": "Only one option", "kind": "choice", "options": [{"label": "A"}]},
            {"id": "q3", "text": "Style?", "kind": "weird", "options": [{"label": "A", "colors": ["#000000"]},
                                                                      {"label": "B"}]},
            {"id": "q4", "text": "Dark?", "kind": "theme", "options": [{"label": "Dark", "colors": ["#000000"]},
                                                                      {"label": "Light", "colors": ["#FFFFFF"]}]},
            {"id": "q5", "text": "Too many", "kind": "choice", "options": [{"label": "A"}, {"label": "B"}]},
        ],
    }
    plan = normalize_plan(raw)
    questions = normalize_questions(raw["questions"])
    validate_plan({"plan": plan, "questions": questions})
    assert plan["files"][0] == {"path": "a.py", "action": "modify", "reason": "r" * 499 + "…"}
    assert len(questions) == 3  # q2 dropped (1 option), q5 over the 0-3 limit
    pal = questions[0]
    assert pal["id"] == "q_1_" and [o["label"] for o in pal["options"]][:2] == ["Ocean", "Forest"]
    assert pal["options"][0]["colors"] == ["#0EA5E9", "#0F172A"]
    assert questions[1]["kind"] == "choice" and questions[1]["options"][0]["colors"] == []


def test_result_payload_caps():
    files = [{"path": f"f{i}", "change_type": "weird", "additions": -5, "deletions": 10**9, "explanation": "e" * 5000}
             for i in range(600)]
    body = result_payload("completed", "s" * 30000, files, "ABC1234")
    validate_result(body)
    assert len(body["files_changed"]) == 500 and body["commit_sha"] == "abc1234" and body["error"] is None
    failed = result_payload("failed", error="")
    validate_result(failed)
    assert failed["error"] and failed["commit_sha"] == ""
    assert result_payload("completed", commit_sha="not-a-sha")["commit_sha"] == ""


def test_extract_json():
    assert extract_json('Sure!\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    assert extract_json('text {"x": 2} more') == {"x": 2}
    assert extract_json("{not json} then {\"ok\": true}") == {"ok": True}
    assert extract_json("nothing") is None


def test_redactor():
    r = Redactor(["my-secret-gemini-key-123"])
    out = r("key=my-secret-gemini-key-123 tok=ghp_" + "a" * 36 + " AIza" + "b" * 35)
    assert "my-secret" not in out and "ghp_" not in out and "AIza" not in out


def test_budget_limits():
    t = [0.0]
    b = Budget.from_limits({"max_steps": 3, "max_tokens": 1000, "max_minutes": 5, "max_fix_attempts": 1},
                           clock=lambda: t[0])
    assert b.exhausted() is None
    b.charge_step(3)
    assert b.exhausted() == "steps"
    b.steps = 0

    class Msg:
        usage_metadata = {"total_tokens": 1200}
        content = ""
    b.record_model_call(10, Msg())
    assert b.exhausted() == "tokens"
    b.tokens = 0
    t[0] = 5 * 60  # past the deadline minus the wrap-up reserve
    assert b.exhausted() == "time"


# ---------------------------------------------------------------------- OIDC + API client

def test_oidc_token_fetched_with_audience_and_refreshed_after_4_minutes():
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"value": f"jwt-{len(seen)}"})

    now = [1000.0]
    p = OIDCTokenProvider("https://token.example/req?api-version=2.0", "req-token", "pocketdev",
                          transport=httpx.MockTransport(handler), clock=lambda: now[0])
    assert p.get() == "jwt-1"
    assert seen[0].url.params["audience"] == "pocketdev"
    assert seen[0].url.params["api-version"] == "2.0"
    assert seen[0].headers["Authorization"] == "bearer req-token"
    now[0] += 200
    assert p.get() == "jwt-1"  # still fresh
    now[0] += 41
    assert p.get() == "jwt-2"  # older than 240 s -> re-fetched
    assert p.get(force=True) == "jwt-3"


def test_oidc_requires_env():
    with pytest.raises(RuntimeError):
        OIDCTokenProvider("", "", "pocketdev")


class Tokens:
    def __init__(self):
        self.n = 0
        self.forced = 0

    def get(self, force=False):
        if force:
            self.forced += 1
            self.n += 1
        return f"t{self.n}"


def client(handler, tokens=None):
    return ApiClient("https://api.example/api/v1", "abc", tokens or Tokens(),
                     transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_api_retries_5xx_and_network_errors():
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"id": "abc", "status": "building"})

    assert client(handler).get_task()["status"] == "building"
    assert len(calls) == 3
    assert calls[0].url.path == "/api/v1/agent/tasks/abc"


def test_api_refreshes_token_once_on_401():
    tokens = Tokens()

    def handler(req):
        if req.headers["Authorization"] == "Bearer t0":
            return httpx.Response(401)
        return httpx.Response(204)

    client(handler, tokens).post_plan({"summary": "s", "steps": [], "files": []}, [])
    assert tokens.forced == 1


def test_api_409_raises_conflict_without_retry():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(409, json={"error": {"code": "conflict", "message": "moved on"}})

    with pytest.raises(TaskConflict):
        client(handler).post_result({"status": "failed"})
    assert len(calls) == 1


def test_api_cancelled_and_4xx():
    with pytest.raises(TaskCancelled):
        client(lambda r: httpx.Response(200, json={"status": "cancelled"})).get_task()
    with pytest.raises(ApiError):
        client(lambda r: httpx.Response(400, json={"error": {"message": "bad"}})).post_plan({}, [])


def test_events_are_best_effort_and_capped():
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(500)

    client(handler).post_event("log", "coder", "m" * 5000)  # does not raise
    assert len(bodies[0]["message"]) <= 2000
