"""Hard per-task limits: steps (tool calls), tokens, wall-clock minutes."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_LIMITS = {"max_steps": 40, "max_tokens": 400_000, "max_minutes": 25, "max_fix_attempts": 3}

# Time kept in reserve at the end so we can still commit, push, summarise and report.
WRAP_UP_SECONDS = 150


def clamp_limits(raw: dict | None) -> dict:
    out = dict(DEFAULT_LIMITS)
    for k, default in DEFAULT_LIMITS.items():
        try:
            v = int((raw or {}).get(k, default))
        except (TypeError, ValueError):
            v = default
        out[k] = max(0 if k == "max_fix_attempts" else 1, min(v, default * 5))
    return out


@dataclass
class Budget:
    max_steps: int
    max_tokens: int
    max_minutes: int
    max_fix_attempts: int
    clock: Callable[[], float] = time.monotonic
    started: float = field(default=0.0)
    steps: int = 0
    tokens: int = 0
    model_calls: int = 0

    @classmethod
    def from_limits(cls, limits: dict | None, clock: Callable[[], float] = time.monotonic,
                    started: float | None = None) -> "Budget":
        lim = clamp_limits(limits)
        return cls(lim["max_steps"], lim["max_tokens"], lim["max_minutes"], lim["max_fix_attempts"],
                   clock=clock, started=clock() if started is None else started)

    # --------------------------------------------------------------- time
    def elapsed(self) -> float:
        return self.clock() - self.started

    def time_left(self) -> float:
        """Seconds left before we must start wrapping up."""
        return self.max_minutes * 60 - WRAP_UP_SECONDS - self.elapsed()

    # --------------------------------------------------------------- accounting
    def charge_step(self, n: int = 1) -> None:
        self.steps += n

    def record_model_call(self, prompt_chars: int, message: Any) -> None:
        self.model_calls += 1
        usage = getattr(message, "usage_metadata", None) or {}
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        if not total:
            out_chars = len(str(getattr(message, "content", "")))
            for tc in getattr(message, "tool_calls", None) or []:
                out_chars += len(str(tc.get("args", "")))
            total = (prompt_chars + out_chars) // 4 + 1  # rough approximation: ~4 chars per token
        self.tokens += int(total)

    def exhausted(self) -> str | None:
        """Name of the first limit reached, or None."""
        if self.steps >= self.max_steps:
            return "steps"
        if self.tokens >= self.max_tokens:
            return "tokens"
        if self.time_left() <= 0:
            return "time"
        return None


LIMIT_TEXT = {
    "steps": "it reached the maximum number of steps for one task",
    "tokens": "it used up the AI budget (tokens) for one task",
    "time": "it ran out of time for one task",
}
