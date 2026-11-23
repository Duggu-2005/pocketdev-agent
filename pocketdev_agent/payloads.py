"""Build payloads that always pass the server's validators (backend/server/tasks/agent_views.py)."""
from __future__ import annotations

import json
import re
from typing import Any

HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,50}$")
SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")

EVENT_TYPES = {"status", "step", "log", "error"}
AGENTS = {"planner", "coder", "tester", "reviewer", "summarizer", "system"}
CHANGE_TYPES = {"added", "modified", "deleted", "renamed"}
QUESTION_KINDS = {"palette", "theme", "choice"}
FILE_ACTIONS = {"create", "modify", "delete"}

MAX_EVENT = 2000
MAX_SUMMARY = 20_000
MAX_ERROR = 2000
MAX_QUESTIONS = 3  # server allows 5; product rule is 0-3


def clip(text: Any, limit: int) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\x00", "").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)].rstrip() + "…"


def _short_id(value: Any, fallback: str) -> str:
    s = str(value or "").strip()
    s = re.sub(r"[^A-Za-z0-9_-]", "_", s)[:50]
    return s if ID_RE.match(s) else fallback


# ---------------------------------------------------------------------- plan

def normalize_plan(raw: Any, fallback_summary: str = "") -> dict:
    raw = raw if isinstance(raw, dict) else {}
    summary = clip(raw.get("summary"), 3000) or clip(fallback_summary, 3000) or "Plan for your request."
    steps = [clip(s, 500) for s in (raw.get("steps") or []) if isinstance(s, (str, int, float)) and str(s).strip()]
    files = []
    seen = set()
    for f in raw.get("files") or []:
        if not isinstance(f, dict):
            continue
        path = str(f.get("path", "")).strip().replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        path = clip(path, 500)
        if not path or path in seen:
            continue
        seen.add(path)
        action = str(f.get("action", "modify")).lower().strip()
        files.append({"path": path, "action": action if action in FILE_ACTIONS else "modify",
                      "reason": clip(f.get("reason"), 500)})
    return {"summary": summary, "steps": steps[:30], "files": files[:100]}


def normalize_questions(raw: Any) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    q_ids: set[str] = set()
    for qi, q in enumerate(raw, start=1):
        if not isinstance(q, dict) or len(out) >= MAX_QUESTIONS:
            continue
        kind = str(q.get("kind", "choice")).lower().strip()
        if kind not in QUESTION_KINDS:
            kind = "choice"
        text = clip(q.get("text") or q.get("question"), 300)
        if not text:
            continue
        qid = _short_id(q.get("id"), f"q{qi}")
        while qid in q_ids:
            qid = f"q{qi}_{len(q_ids)}"
        options = []
        o_ids: set[str] = set()
        for oi, o in enumerate(q.get("options") or [], start=1):
            if not isinstance(o, dict) or len(options) >= 6:
                continue
            label = clip(o.get("label"), 80)
            if not label:
                continue
            colors = o.get("colors")
            if isinstance(colors, str):  # tolerate "#aaa111,#bbb222"
                colors = colors.split(",")
            colors = [c.strip() for c in (colors or []) if isinstance(c, str) and HEX_COLOR.match(c.strip())][:8]
            if kind in ("palette", "theme") and not colors:
                continue  # colour options without colours are useless cards
            if kind == "choice":
                colors = []
            oid = _short_id(o.get("id"), f"o{oi}")
            while oid in o_ids:
                oid = f"o{oi}_{len(o_ids)}"
            o_ids.add(oid)
            options.append({"id": oid, "label": label, "description": clip(o.get("description"), 300),
                            "colors": colors})
        if len(options) < 2:
            continue
        q_ids.add(qid)
        out.append({"id": qid, "text": text, "kind": kind, "options": options})
    return out


# ---------------------------------------------------------------------- result

def normalize_files_changed(files: list[dict]) -> list[dict]:
    out = []
    for f in files[:500]:
        ct = f.get("change_type")
        out.append({
            "path": clip(f.get("path"), 500),
            "change_type": ct if ct in CHANGE_TYPES else "modified",
            "additions": max(0, min(int(f.get("additions") or 0), 1_000_000)),
            "deletions": max(0, min(int(f.get("deletions") or 0), 1_000_000)),
            "explanation": clip(f.get("explanation"), 1000),
        })
    return [f for f in out if f["path"]]


def result_payload(status: str, summary: str = "", files_changed: list[dict] | None = None,
                   commit_sha: str = "", error: str | None = None) -> dict:
    assert status in ("completed", "failed")
    sha = (commit_sha or "").strip().lower()
    return {
        "status": status,
        "summary": clip(summary, MAX_SUMMARY),
        "files_changed": normalize_files_changed(files_changed or []),
        "commit_sha": sha if SHA_RE.match(sha) else "",
        "error": (clip(error, MAX_ERROR) or "The AI couldn't finish this task.") if status == "failed" else None,
    }


# ---------------------------------------------------------------------- model output parsing

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (fenced or bare)."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)]
    candidates.append(text)
    dec = json.JSONDecoder()
    for cand in candidates:
        idx = cand.find("{")
        while idx != -1:
            try:
                obj, _ = dec.raw_decode(cand, idx)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            idx = cand.find("{", idx + 1)
    return None
