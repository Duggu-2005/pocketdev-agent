"""PocketDev API client (agent runner callbacks), authenticated with GitHub Actions OIDC tokens."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable
from urllib.parse import quote

import httpx

from pocketdev_agent.payloads import clip

log = logging.getLogger(__name__)

TOKEN_MAX_AGE = 240  # seconds; GitHub OIDC tokens live ~5-10 minutes
RETRY_STATUSES = {429, 500, 502, 503, 504}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status


class TaskConflict(Exception):
    """409 from the API: the task moved on (e.g. cancelled). Stop without retrying."""


class TaskCancelled(Exception):
    """The task was cancelled by the user. Stop immediately."""


class OIDCTokenProvider:
    """Fetches a GitHub Actions OIDC token and re-fetches it before it gets old."""

    def __init__(self, request_url: str, request_token: str, audience: str, *,
                 transport: httpx.BaseTransport | None = None, clock: Callable[[], float] = time.monotonic,
                 max_age: float = TOKEN_MAX_AGE):
        if not request_url or not request_token:
            raise RuntimeError("GitHub Actions OIDC is not available. The workflow needs 'id-token: write'.")
        self._url = request_url
        self._request_token = request_token
        self._audience = audience
        self._transport = transport
        self._clock = clock
        self._max_age = max_age
        self._token: str | None = None
        self._fetched_at = 0.0
        self.fetch_count = 0

    def get(self, force: bool = False) -> str:
        if force or not self._token or self._clock() - self._fetched_at >= self._max_age:
            self._token = self._fetch()
            self._fetched_at = self._clock()
        return self._token

    def _fetch(self) -> str:
        sep = "&" if "?" in self._url else "?"
        url = f"{self._url}{sep}audience={quote(self._audience, safe='')}"
        last: Exception | None = None
        for attempt in range(4):
            try:
                with httpx.Client(transport=self._transport, timeout=20) as c:
                    r = c.get(url, headers={"Authorization": f"bearer {self._request_token}",
                                            "Accept": "application/json"})
                if r.status_code == 200:
                    value = r.json().get("value")
                    if value:
                        self.fetch_count += 1
                        return value
                    last = RuntimeError("OIDC response had no token")
                elif r.status_code in RETRY_STATUSES:
                    last = RuntimeError(f"OIDC token request failed ({r.status_code})")
                else:
                    raise RuntimeError(f"OIDC token request failed ({r.status_code})")
            except httpx.TransportError as e:
                last = e
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Could not get a GitHub OIDC token: {last}")


class ApiClient:
    def __init__(self, api_base: str, task_id: str, tokens: Any, *, transport: httpx.BaseTransport | None = None,
                 max_attempts: int = 5, sleep: Callable[[float], None] = time.sleep):
        self.base = f"{api_base.rstrip('/')}/agent/tasks/{quote(task_id, safe='')}"
        self.tokens = tokens
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._client = httpx.Client(transport=transport, timeout=30,
                                    headers={"User-Agent": "pocketdev-agent", "Accept": "application/json"})

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, json: Any = None) -> httpx.Response:
        url = self.base + path
        refreshed = False
        attempt = 0
        while True:
            attempt += 1
            try:
                r = self._client.request(method, url, json=json,
                                         headers={"Authorization": f"Bearer {self.tokens.get(force=False)}"})
            except httpx.TransportError as e:
                if attempt >= self.max_attempts:
                    raise ApiError(0, f"network error: {type(e).__name__}") from e
                self._sleep(min(2 ** attempt, 30))
                continue
            if r.status_code == 401 and not refreshed:
                refreshed = True
                self.tokens.get(force=True)  # token expired / rejected: get a fresh one once
                attempt -= 1
                continue
            if r.status_code == 409:
                raise TaskConflict(_error_message(r))
            if r.status_code in RETRY_STATUSES and attempt < self.max_attempts:
                delay = min(2 ** attempt, 30)
                if r.status_code == 429:
                    try:
                        delay = min(max(float(r.headers.get("Retry-After", delay)), 1), 60)
                    except ValueError:
                        pass
                self._sleep(delay)
                continue
            if r.status_code >= 400:
                raise ApiError(r.status_code, _error_message(r))
            return r

    # ------------------------------------------------------------------ endpoints
    def get_task(self) -> dict:
        task = self._request("GET", "").json()
        if task.get("status") == "cancelled":
            raise TaskCancelled()
        return task

    def check_not_cancelled(self) -> None:
        self.get_task()

    def post_event(self, type_: str, agent: str, message: str) -> None:
        """Best effort: progress events must never crash the run."""
        try:
            self._request("POST", "/events", {"type": type_, "agent": agent, "message": clip(message, 2000) or "..."})
        except TaskConflict:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("could not send event: %s", e)

    def post_plan(self, plan: dict, questions: list) -> None:
        self._request("POST", "/plan", {"plan": plan, "questions": questions})

    def post_result(self, payload: dict) -> None:
        self._request("POST", "/result", payload)


def _error_message(r: httpx.Response) -> str:
    try:
        return str(r.json().get("error", {}).get("message", ""))[:300]
    except Exception:  # noqa: BLE001
        return r.text[:300]
