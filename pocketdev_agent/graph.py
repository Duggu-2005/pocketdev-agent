"""The LangGraph agent graph.

    START --(phase=plan)--> planner --> END
    START --(phase=build)-> coder --> tester --(checks failed & fix attempts left)--> coder
                                         \\--> reviewer --(issues, first review only)--> coder
                                                   \\--> commit --(nothing to commit / blocked)--> END
                                                            \\--> summarizer --> publish --> END
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from pocketdev_agent import prompts
from pocketdev_agent.agents import (Context, ask_json, call_model, describe_answers, repo_context, run_tool_loop,
                                    to_json)
from pocketdev_agent.checks import failures_text, results_summary
from pocketdev_agent.gitops import GitError
from pocketdev_agent.limits import LIMIT_TEXT
from pocketdev_agent.models import chain
from pocketdev_agent.payloads import clip, normalize_plan, normalize_questions
from pocketdev_agent.redact import find_secrets
from pocketdev_agent.tools import READ_ONLY_TOOLS

log = logging.getLogger(__name__)

PLANNER_TOOL_ROUNDS = 8
CODER_TOOL_ROUNDS = 40


class State(TypedDict, total=False):
    phase: str
    plan: dict
    questions: list
    fix_attempts: int
    review_rounds: int
    feedback: str
    coder_changed: bool
    checks: list
    checks_ok: bool
    next: str
    limit: str
    review_notes: list
    commit_sha: str
    summary: str
    files_changed: list
    status: str
    error: str


def build_graph(ctx: Context):
    read_tools = [t for t in ctx.tools if t.name in READ_ONLY_TOOLS]

    # ------------------------------------------------------------------ planner
    def planner(state: State) -> dict:
        ctx.check_cancelled()
        task = ctx.task
        ctx.event("planner", "Looking at your project", "status")
        human = (f"PocketDev task from the project owner:\n{task.get('prompt', '')}\n\n"
                 f"Base branch: {task.get('base_branch')}\n\n"
                 f"<repo_context>\n{ctx.redact(repo_context(ctx.ws))}\n</repo_context>")
        messages = [SystemMessage(prompts.PLANNER), HumanMessage(human)]
        text, limit = run_tool_loop(ctx, chain(ctx.models, read_tools), messages, "planner", read_tools,
                                    PLANNER_TOOL_ROUNDS)
        from pocketdev_agent.payloads import extract_json
        data = extract_json(text)
        if data is None:
            # Ask once more, without tools, for the final JSON.
            data = ask_json(ctx, chain(ctx.models), messages + [HumanMessage(
                "Now write the plan. Reply with ONLY the JSON object described in the instructions.")])
        if data is None:
            raise RuntimeError("The AI could not write a plan for this task. Please try again or rephrase it.")
        plan = normalize_plan(data, fallback_summary=task.get("prompt", ""))
        questions = normalize_questions(data.get("questions"))
        ctx.event("planner", f"Plan ready with {len(plan['steps'])} steps", "step")
        return {"plan": plan, "questions": questions}

    # ------------------------------------------------------------------ coder
    def coder(state: State) -> dict:
        ctx.check_cancelled()
        reason = ctx.budget.exhausted()
        if reason:
            return {"limit": reason, "coder_changed": False}
        task = ctx.task
        feedback = state.get("feedback") or ""
        ctx.event("coder", "Fixing problems found during checks" if feedback else "Writing the code", "status")
        parts = [f"PocketDev task from the project owner:\n{task.get('prompt', '')}"]
        if task.get("followup_prompt"):
            parts.append("This is a FOLLOW-UP on work already on this branch. New request from the owner:\n"
                         f"{task['followup_prompt']}")
        if task.get("plan"):
            parts.append(f"Approved plan:\n{to_json(task['plan'])}")
        answers = describe_answers(task)
        if answers:
            parts.append(f"The owner's design choices:\n{answers}")
        if task.get("note"):
            parts.append(f"Extra note from the owner:\n{task['note']}")
        if ctx.ws.touched:
            parts.append("Files you already changed in this run: " + ", ".join(sorted(ctx.ws.touched)))
        if feedback:
            parts.append(f"Please fix these problems:\n{feedback}")
        messages = [SystemMessage(prompts.CODER), HumanMessage("\n\n".join(parts))]
        before = _snapshot(ctx)
        text, limit = run_tool_loop(ctx, chain(ctx.models, ctx.tools), messages, "coder", ctx.tools,
                                    CODER_TOOL_ROUNDS)
        changed = _snapshot(ctx) != before
        if text:
            ctx.event("coder", clip(text, 500), "log")
        out: dict = {"coder_changed": changed, "feedback": ""}
        if limit:
            out["limit"] = limit
        return out

    # ------------------------------------------------------------------ tester
    def tester(state: State) -> dict:
        ctx.check_cancelled()
        if ctx.tester is None or not ctx.ws.touched:
            return {"checks": [], "checks_ok": True, "next": "reviewer"}
        ctx.event("tester", "Checking that everything still works", "status")
        results = ctx.tester.run(ctx.ws.touched)
        ok = all(r.ok or r.skipped for r in results)
        summary = results_summary(results)
        if results:
            ctx.event("tester", ("All checks passed" if ok else "Some checks failed") + ": " + "; ".join(summary),
                      "step")
        attempts = state.get("fix_attempts", 0)
        can_fix = (not ok and state.get("coder_changed", True) and attempts < ctx.budget.max_fix_attempts
                   and not state.get("limit") and not ctx.budget.exhausted())
        out = {"checks": summary, "checks_ok": ok, "next": "coder" if can_fix else "reviewer"}
        if can_fix:
            out["fix_attempts"] = attempts + 1
            out["feedback"] = failures_text(results)
        return out

    # ------------------------------------------------------------------ reviewer
    def reviewer(state: State) -> dict:
        ctx.check_cancelled()
        staged = ctx.git.stage(sorted(ctx.ws.touched))
        if not staged:
            return {"next": "commit"}
        ctx.event("reviewer", "Reviewing the changes", "status")
        diff = ctx.git.staged_diff()
        issues = []
        added = "\n".join(line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
        if find_secrets(added):
            issues.append("The change contains something that looks like a password, token or private key. "
                          "Remove it and read it from configuration instead.")
        if not ctx.budget.exhausted() and not state.get("limit"):
            data = ask_json(ctx, chain(ctx.models), [
                SystemMessage(prompts.REVIEWER),
                HumanMessage(f"Task: {ctx.task.get('followup_prompt') or ctx.task.get('prompt', '')}\n\n"
                             f"<diff>\n{ctx.redact(diff)}\n</diff>")])
            if data and not data.get("approved", True):
                issues += [clip(i, 500) for i in (data.get("issues") or []) if isinstance(i, str)][:10]
        rounds = state.get("review_rounds", 0)
        if issues and rounds == 0 and not ctx.budget.exhausted() and not state.get("limit"):
            ctx.event("reviewer", f"Found {len(issues)} thing(s) to fix", "step")
            return {"review_rounds": 1, "next": "coder",
                    "feedback": "Code review found:\n" + "\n".join(f"- {i}" for i in issues)}
        if issues:
            ctx.event("reviewer", "Review notes: " + "; ".join(issues)[:1500], "log")
        else:
            ctx.event("reviewer", "Review passed", "step")
        return {"review_rounds": rounds + (1 if issues else 0), "review_notes": issues, "next": "commit"}

    # ------------------------------------------------------------------ commit
    def commit(state: State) -> dict:
        ctx.check_cancelled()
        staged = ctx.git.stage(sorted(ctx.ws.touched))
        if staged:
            added = "\n".join(line for line in ctx.git.staged_diff(max_chars=2_000_000).splitlines()
                              if line.startswith("+") and not line.startswith("+++"))
            if find_secrets(added):
                return {"status": "failed", "error": "The change contained something that looks like a password or "
                                                     "secret key, so it was not saved. Please try again."}
        title = clip((ctx.task.get("followup_prompt") or ctx.task.get("prompt") or "update").splitlines()[0], 60)
        sha = ctx.git.commit(staged, f"PocketDev: {title}\n\nMade by PocketDev AI agents.") if staged else None
        if not sha:
            msg = "The AI didn't change any files. Try describing the change in more detail."
            if state.get("limit"):
                msg = f"The AI stopped because {LIMIT_TEXT[state['limit']]} before changing any files."
            return {"status": "failed", "error": msg}
        ctx.event("system", "Saved the changes", "step")
        return {"commit_sha": sha}

    # ------------------------------------------------------------------ summarizer
    def summarizer(state: State) -> dict:
        ctx.event("summarizer", "Writing a summary of what changed", "status")
        base = ctx.task["base_branch"]
        files = ctx.git.files_changed(base)  # from git, not from the model
        stat = ctx.git.diff_stat(base)
        overview, explanations = "", {}
        if not ctx.budget.exhausted() or ctx.budget.exhausted() == "steps":
            diff = ctx.git.diff_text(base, max_chars=50_000)
            try:
                data = ask_json(ctx, chain(ctx.models), [
                    SystemMessage(prompts.SUMMARIZER),
                    HumanMessage(f"Task: {ctx.task.get('prompt', '')}\n\n<diff>\n{stat}\n\n{ctx.redact(diff)}\n</diff>")])
            except Exception as e:  # noqa: BLE001 - summary falls back to the plain git stats
                log.warning("summarizer model failed: %s", type(e).__name__)
                data = None
            if data:
                overview = clip(data.get("overview"), 4000) if isinstance(data.get("overview"), str) else ""
                if isinstance(data.get("files"), dict):
                    explanations = {str(k): clip(v, 1000) for k, v in data["files"].items() if isinstance(v, str)}
        for f in files:
            if f["path"] in explanations:
                f["explanation"] = (f["explanation"] + ". " if f["explanation"] else "") + explanations[f["path"]]
        return {"files_changed": files, "summary": build_summary(overview, files, state)}

    # ------------------------------------------------------------------ publish
    def publish(state: State) -> dict:
        ctx.check_cancelled()
        branch = ctx.task["branch"]
        ctx.event("system", f"Uploading the changes to branch {branch}", "status")
        try:
            ctx.git.push(branch, ctx.protected_branches)
        except GitError as e:
            log.warning("push failed: %s", ctx.redact(str(e)))
            return {"status": "failed", "error": "Could not upload the changes to GitHub. "
                                                 "Someone may have changed the branch meanwhile; please try again."}
        return {"status": "completed"}

    # ------------------------------------------------------------------ wiring
    g = StateGraph(State)
    for name, fn in [("planner", planner), ("coder", coder), ("tester", tester), ("reviewer", reviewer),
                     ("commit", commit), ("summarizer", summarizer), ("publish", publish)]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, lambda s: "planner" if s.get("phase") == "plan" else "coder",
                            ["planner", "coder"])
    g.add_edge("planner", END)
    g.add_edge("coder", "tester")
    g.add_conditional_edges("tester", lambda s: s.get("next", "reviewer"), ["coder", "reviewer"])
    g.add_conditional_edges("reviewer", lambda s: s.get("next", "commit"), ["coder", "commit"])
    g.add_conditional_edges("commit", lambda s: END if s.get("status") == "failed" else "summarizer",
                            ["summarizer", END])
    g.add_edge("summarizer", "publish")
    g.add_edge("publish", END)
    return g.compile()


def _snapshot(ctx: Context) -> tuple:
    """Cheap fingerprint of the touched files, to know whether a coder round changed anything."""
    out = []
    for rel in sorted(ctx.ws.touched):
        p = ctx.ws.root / rel
        try:
            st = p.stat()
            out.append((rel, st.st_size, st.st_mtime_ns))
        except OSError:
            out.append((rel, None, None))
    return tuple(out)


def build_summary(overview: str, files: list[dict], state: State) -> str:
    adds = sum(f["additions"] for f in files)
    dels = sum(f["deletions"] for f in files)
    lines = ["## What changed", overview or f"Changed {len(files)} file(s): {adds} line(s) added, "
                                            f"{dels} line(s) removed.", "", "## Files"]
    for f in files[:200]:
        expl = f" — {f['explanation']}" if f.get("explanation") else ""
        lines.append(f"- `{f['path']}` ({f['change_type']}, +{f['additions']} −{f['deletions']}){expl}")
    if len(files) > 200:
        lines.append(f"- …and {len(files) - 200} more")
    if state.get("checks"):
        lines += ["", "## Checks"] + [f"- {c}" for c in state["checks"]]
    notes = []
    if state.get("checks") and not state.get("checks_ok", True):
        notes.append("Some checks still fail. You can ask for a follow-up to fix them.")
    if state.get("review_notes"):
        notes.append("The reviewer still has notes: " + "; ".join(state["review_notes"])[:1500])
    if state.get("limit"):
        notes.append(f"The AI stopped early because {LIMIT_TEXT[state['limit']]}. The work may be incomplete.")
    if notes:
        lines += ["", "## Heads-up"] + [f"- {n}" for n in notes]
    return "\n".join(lines)
