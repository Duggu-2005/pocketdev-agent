"""Shared agent machinery: run context, model calls with budget accounting, the tool-calling loop."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable

from pocketdev_agent.api import ApiClient
from pocketdev_agent.checks import Tester
from pocketdev_agent.config import Settings
from pocketdev_agent.gitops import GitRepo
from pocketdev_agent.limits import Budget
from pocketdev_agent.redact import Redactor
from pocketdev_agent.sandbox import SandboxError, Workspace
from pocketdev_agent.tools import READ_ONLY_TOOLS, describe_call

log = logging.getLogger(__name__)

MAX_TOOL_OUTPUT = 24_000
REPO_CONTEXT_CHARS = 30_000
KEY_FILES = ["README.md", "README", "readme.md", "README.rst", "package.json", "requirements.txt", "pyproject.toml",
             "setup.py", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
             "app/build.gradle", "app/build.gradle.kts", "pubspec.yaml", "go.mod", "Cargo.toml", "composer.json",
             "Gemfile", "index.html", "vite.config.js", "vite.config.ts", "next.config.js", "manage.py"]


class ModelError(RuntimeError):
    """All configured AI providers failed."""


@dataclass
class Context:
    settings: Settings
    api: ApiClient
    ws: Workspace
    git: GitRepo
    budget: Budget
    models: list
    tools: list
    tester: Tester | None
    task: dict
    redact: Redactor
    protected_branches: set[str] = field(default_factory=set)

    def event(self, agent: str, message: str, type_: str = "step") -> None:
        log.info("[%s] %s", agent, self.redact(message))
        self.api.post_event(type_, agent, self.redact(message))

    def check_cancelled(self) -> None:
        self.api.check_not_cancelled()


def message_chars(messages: list[BaseMessage]) -> int:
    return sum(len(str(m.content)) for m in messages)


def call_model(ctx: Context, runnable: Runnable, messages: list[BaseMessage]) -> AIMessage:
    try:
        msg = runnable.invoke(messages)
    except Exception as e:  # noqa: BLE001 - every provider in the fallback chain failed
        log.warning("model call failed: %s", ctx.redact(f"{type(e).__name__}: {e}")[:500])
        raise ModelError("The AI service did not respond (it may be out of free quota). "
                         "Please try again later, or add another AI key in the PocketDev app.") from e
    ctx.budget.record_model_call(message_chars(messages), msg)
    return msg


def wrap_tool_output(name: str, text: str) -> str:
    if len(text) > MAX_TOOL_OUTPUT:
        text = text[:MAX_TOOL_OUTPUT] + "\n... (output cut)"
    return f'<tool_result tool="{name}">\n{text}\n</tool_result>'


def run_tool_loop(ctx: Context, runnable: Runnable, messages: list[BaseMessage], agent: str,
                  tools: list, max_rounds: int) -> tuple[str, str | None]:
    """Call the model, execute its tool calls, repeat until it answers without tools.

    Returns (final text, limit reason or None). Every tool call costs one step of the budget.
    """
    by_name = {t.name: t for t in tools}
    text = ""
    for _ in range(max_rounds):
        reason = ctx.budget.exhausted()
        if reason:
            return text, reason
        ai = call_model(ctx, runnable, messages)
        messages.append(ai)
        text = ai.text or text
        if not ai.tool_calls:
            return text, None
        for i, tc in enumerate(ai.tool_calls):
            name, args = tc.get("name", ""), tc.get("args") or {}
            call_id = tc.get("id") or f"call_{ctx.budget.steps}_{i}"
            if ctx.budget.exhausted() == "steps":
                out = "Step limit reached. Stop calling tools and give your final answer now."
            elif name not in by_name:
                out = f"Error: unknown tool '{name}'. Available: {', '.join(by_name)}."
            else:
                ctx.budget.charge_step()
                ctx.event(agent, describe_call(name, args), "log" if name in READ_ONLY_TOOLS else "step")
                try:
                    out = str(by_name[name].invoke(args))
                except SandboxError as e:
                    out = f"Error: {e}"
                except Exception as e:  # noqa: BLE001 - bad arguments etc.; tell the model, keep going
                    out = f"Error: {type(e).__name__}: {str(e)[:300]}"
            messages.append(ToolMessage(content=wrap_tool_output(name, ctx.redact(out)), tool_call_id=call_id,
                                        name=name))
    return text, "steps"


def ask_json(ctx: Context, runnable: Runnable, messages: list[BaseMessage]) -> dict | None:
    """One model call expecting JSON; one retry with a reminder if the reply is not JSON."""
    from pocketdev_agent.payloads import extract_json
    from langchain_core.messages import HumanMessage

    for attempt in range(2):
        if ctx.budget.exhausted():
            return None
        ai = call_model(ctx, runnable, messages)
        data = extract_json(ai.text)
        if data is not None:
            return data
        messages = messages + [ai, HumanMessage("Reply with ONLY the JSON object, nothing else.")]
    return None


def repo_context(ws: Workspace, max_files: int = 300) -> str:
    """A compact, size-capped view of the repo for the planner (never the whole repo)."""
    files = list(ws.iter_files(limit=max_files + 1))
    tree = "\n".join(files[:max_files]) + ("\n... (more files not shown)" if len(files) > max_files else "")
    parts = [f"File tree ({min(len(files), max_files)} files shown):\n{tree or '(empty repository)'}"]
    budget = REPO_CONTEXT_CHARS - len(parts[0])
    key = [".pocketdev.md"] + KEY_FILES
    for rel in key:
        if budget <= 500:
            break
        try:
            target, rel_ok = ws.resolve(rel)
        except SandboxError:
            continue
        if not target.is_file():
            continue
        cap = 6000 if rel == ".pocketdev.md" else 3000
        try:
            with open(target, "rb") as fh:
                data = fh.read(cap)
        except OSError:
            continue
        if b"\x00" in data:
            continue
        snippet = data.decode("utf-8", errors="replace")[:min(cap, budget)]
        parts.append(f"--- {rel_ok} ---\n{snippet}")
        budget -= len(snippet)
    return "\n\n".join(parts)


def describe_answers(task: dict) -> str:
    """Turn {"q1": "o2"} into readable text with the chosen option's label and colours."""
    answers = task.get("answers") or {}
    lines = []
    for q in task.get("questions") or []:
        chosen = answers.get(q.get("id"))
        if not chosen:
            continue
        opt = next((o for o in q.get("options", []) if o.get("id") == chosen), None)
        if opt:
            colors = f" colours: {', '.join(opt.get('colors') or [])}" if opt.get("colors") else ""
            lines.append(f"- {q.get('text')} -> {opt.get('label')}"
                         f"{' (' + opt['description'] + ')' if opt.get('description') else ''}{colors}")
        else:
            lines.append(f"- {q.get('text')} -> {str(chosen)[:200]}")
    return "\n".join(lines)


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1)
