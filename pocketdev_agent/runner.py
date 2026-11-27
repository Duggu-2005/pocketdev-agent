"""Entry point: `python -m pocketdev_agent`."""
from __future__ import annotations

import dataclasses
import logging
import sys
import time
from typing import Callable

from langgraph.errors import GraphRecursionError

from pocketdev_agent.agents import Context, ModelError
from pocketdev_agent.api import ApiClient, ApiError, OIDCTokenProvider, TaskCancelled, TaskConflict
from pocketdev_agent.checks import Tester
from pocketdev_agent.config import NO_KEY_MESSAGE, Settings
from pocketdev_agent.gitops import ALWAYS_PROTECTED, GitError, GitRepo, PushRefused
from pocketdev_agent.graph import build_graph
from pocketdev_agent.limits import Budget
from pocketdev_agent.models import build_models, resolve_providers
from pocketdev_agent.payloads import result_payload
from pocketdev_agent.redact import RedactingFilter, Redactor
from pocketdev_agent.sandbox import Workspace
from pocketdev_agent.tools import make_tools

log = logging.getLogger("pocketdev_agent")

GENERIC_ERROR = "Something went wrong while the AI was working on your task. Please try again."
EXPECTED_STATUS = {"plan": "planning", "build": "building"}


def friendly_error(e: BaseException) -> str:
    if isinstance(e, ModelError):
        return str(e)
    if isinstance(e, PushRefused):
        return "PocketDev only saves work to a separate feature branch, never to the main branch. Pick another branch name."
    if isinstance(e, GitError) and str(e).startswith("The "):  # our own friendly git messages
        return str(e)
    if isinstance(e, GraphRecursionError):
        return "The AI reached the maximum number of steps for one task."
    if isinstance(e, RuntimeError) and str(e).startswith("The AI "):
        return str(e)
    return GENERIC_ERROR


def run(settings: Settings, api: ApiClient, model_factory: Callable[[Settings], list] = build_models, *,
        redactor: Redactor | None = None, tester_factory: Callable[..., Tester] | None = None,
        clock: Callable[[], float] = time.monotonic) -> int:
    started = clock()
    redact = redactor or Redactor(settings.secret_values())
    try:
        task = api.get_task()
    except TaskCancelled:
        log.info("Task was cancelled; nothing to do.")
        return 0
    phase = settings.phase or str(task.get("phase", ""))
    if phase not in EXPECTED_STATUS:
        log.error("Unknown phase %r", phase)
        return 2
    if task.get("status") != EXPECTED_STATUS[phase]:
        log.info("Task is %s, not %s; it has moved on. Stopping.", task.get("status"), EXPECTED_STATUS[phase])
        return 0

    def fail(message: str) -> int:
        try:
            if phase == "build":
                api.post_result(result_payload("failed", error=message))
            else:  # the API has no failure endpoint for planning; tell the user through an error event
                api.post_event("error", "planner", message)
        except (TaskConflict, TaskCancelled):
            return 0
        except Exception as e:  # noqa: BLE001
            log.error("could not report failure: %s", redact(str(e)))
        return 1

    providers, warning = resolve_providers(settings.providers, task)
    if warning:
        log.warning("%s", warning)
        api.post_event("log", "system", warning)
    settings = dataclasses.replace(settings, providers=providers)
    models = model_factory(settings)
    if not models:
        return fail(NO_KEY_MESSAGE)

    budget = Budget.from_limits(task.get("limits"), clock=clock, started=started)
    ws = Workspace(settings.repo_dir)
    git = GitRepo(ws.root, github_token=settings.github_token, server_url=settings.server_url)
    try:
        protected: set[str] = set()
        if phase == "build":
            git.secure_credentials(settings.temp_dir)  # before any project code can run
            protected = {task.get("base_branch") or ""} | git.default_branches(settings.event_path, settings.ref_name) \
                | ALWAYS_PROTECTED
            from pocketdev_agent.gitops import ensure_push_allowed
            ensure_push_allowed(task.get("branch") or "", protected)
            api.post_event("status", "system", "Getting your branch ready")
            how = git.prepare_branch(task["branch"], task["base_branch"], bool(task.get("branch_is_new")))
            log.info("branch %s (%s)", task["branch"], how)
        tester = None
        if phase == "build":
            make_tester = tester_factory or Tester
            tester = make_tester(ws.root, time_left=budget.time_left, temp_dir=settings.temp_dir, redact=redact,
                                 on_progress=lambda m: api.post_event("log", "tester", redact(m)))
        ctx = Context(settings=settings, api=api, ws=ws, git=git, budget=budget, models=models,
                      tools=make_tools(ws, read_only=(phase == "plan")), tester=tester, task=task, redact=redact,
                      protected_branches=protected)
        graph = build_graph(ctx)
        recursion = 12 + 4 * (budget.max_fix_attempts + 3)
        final = graph.invoke({"phase": phase, "fix_attempts": 0, "review_rounds": 0}, {"recursion_limit": recursion})
        log.info("usage: %d tool steps, %d model calls, ~%d tokens, %.0fs", budget.steps, budget.model_calls,
                 budget.tokens, budget.elapsed())
        if phase == "plan":
            api.post_plan(final["plan"], final.get("questions") or [])
            return 0
        status = final.get("status") or "failed"
        api.post_result(result_payload(
            "completed" if status == "completed" else "failed",
            summary=redact(final.get("summary", "")),
            files_changed=final.get("files_changed") or [],
            commit_sha=final.get("commit_sha", "") if status == "completed" else "",
            error=redact(final.get("error") or GENERIC_ERROR) if status != "completed" else None,
        ))
        return 0 if status == "completed" else 1
    except (TaskCancelled, TaskConflict):
        log.info("Task was cancelled or moved on. Stopping.")
        return 0
    except Exception as e:  # noqa: BLE001
        log.error("task failed: %s", redact(f"{type(e).__name__}: {e}"))
        return fail(friendly_error(e))


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    redactor = Redactor(settings.secret_values())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.addFilter(RedactingFilter(redactor))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    for noisy in ("httpx", "httpcore", "langchain", "google", "openai", "anthropic", "groq"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        tokens = OIDCTokenProvider(settings.oidc_request_url, settings.oidc_request_token, settings.audience)
        api = ApiClient(settings.api_base, settings.task_id, tokens)
    except RuntimeError as e:
        log.error("%s", e)
        return 2
    try:
        return run(settings, api, redactor=redactor)
    except ApiError as e:
        log.error("PocketDev API error: %s", redactor(str(e)))
        return 3
    finally:
        api.close()
