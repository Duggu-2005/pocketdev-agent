"""Mirror of the server's validators in backend/server/tasks/agent_views.py (kept in sync by hand)."""
import re

HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,50}$")
SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")


def _str(v, maxlen, blank=False):
    assert isinstance(v, str), v
    assert len(v) <= maxlen, (len(v), maxlen)
    if not blank:
        assert v.strip(), "blank"


def validate_event(body):
    assert set(body) == {"type", "agent", "message"}
    assert body["type"] in {"status", "step", "log", "error"}
    assert body["agent"] in {"planner", "coder", "tester", "reviewer", "summarizer", "system"}
    _str(body["message"], 2000)


def validate_plan(body):
    plan = body["plan"]
    _str(plan["summary"], 3000)
    assert isinstance(plan["steps"], list) and len(plan["steps"]) <= 30
    for s in plan["steps"]:
        _str(s, 500)
    assert isinstance(plan["files"], list) and len(plan["files"]) <= 100
    for f in plan["files"]:
        _str(f["path"], 500)
        assert f["action"] in {"create", "modify", "delete"}
        _str(f.get("reason", ""), 500, blank=True)
    qs = body.get("questions", [])
    assert isinstance(qs, list) and len(qs) <= 5
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert ID_RE.match(q["id"])
        _str(q["text"], 300)
        assert q["kind"] in {"palette", "theme", "choice"}
        assert 2 <= len(q["options"]) <= 6
        assert len({o["id"] for o in q["options"]}) == len(q["options"])
        for o in q["options"]:
            assert ID_RE.match(o["id"])
            _str(o["label"], 80)
            _str(o.get("description", ""), 300, blank=True)
            assert len(o.get("colors", [])) <= 8
            assert all(HEX_COLOR.match(c) for c in o.get("colors", []))


def validate_result(body):
    assert body["status"] in {"completed", "failed"}
    _str(body.get("summary", ""), 20_000, blank=True)
    fc = body.get("files_changed", [])
    assert len(fc) <= 500
    for f in fc:
        _str(f["path"], 500)
        assert f["change_type"] in {"added", "modified", "deleted", "renamed"}
        assert isinstance(f["additions"], int) and 0 <= f["additions"] <= 1_000_000
        assert isinstance(f["deletions"], int) and 0 <= f["deletions"] <= 1_000_000
        _str(f.get("explanation", ""), 1000, blank=True)
    sha = body.get("commit_sha", "")
    assert sha == "" or SHA_RE.match(sha)
    err = body.get("error")
    assert err is None or (isinstance(err, str) and len(err) <= 2000)
