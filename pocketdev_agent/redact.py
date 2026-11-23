"""Scrub secrets from any text that leaves the runner (events, logs, prompts, results)."""
from __future__ import annotations

import base64
import logging
import re

# Generic token shapes (also used by the reviewer's secret scan on added lines).
SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{32,}"),
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
    re.compile(r"(?i)authorization:\s*(?:basic|bearer)\s+[A-Za-z0-9+/=._\-]{8,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWTs (OIDC tokens)
]

MASK = "[redacted]"


class Redactor:
    def __init__(self, secrets: list[str] | None = None):
        self._values: list[str] = []
        for s in secrets or []:
            self.add(s)

    def add(self, secret: str) -> None:
        if not secret or len(secret) < 6:
            return
        variants = {secret, base64.b64encode(f"x-access-token:{secret}".encode()).decode()}
        for v in variants:
            if v not in self._values:
                self._values.append(v)
        self._values.sort(key=len, reverse=True)

    def __call__(self, text: str | None) -> str:
        if not text:
            return text or ""
        for v in self._values:
            text = text.replace(v, MASK)
        for pat in SECRET_PATTERNS:
            text = pat.sub(MASK, text)
        return text


class RedactingFilter(logging.Filter):
    def __init__(self, redactor: Redactor):
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = self.redactor(record.getMessage())
            record.args = ()
        except Exception:  # never break logging
            pass
        return True


def find_secrets(text: str) -> list[str]:
    """Return short descriptions of secret-looking strings in text (no secret values)."""
    found = []
    for pat in SECRET_PATTERNS[:-2]:  # skip the auth-header / JWT shapes: noisy in code
        if pat.search(text):
            found.append(pat.pattern[:30])
    return found
