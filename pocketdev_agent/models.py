"""Bring-your-own-key chat models with LangChain fallbacks (Groq -> Gemini -> OpenRouter -> Anthropic)."""
from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from pocketdev_agent.config import DEFAULT_OPENROUTER_MODEL, OPENROUTER_BASE_URL, ProviderConfig, Settings

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 180
MAX_OUTPUT_TOKENS = 16_000


def build_model(p: ProviderConfig) -> BaseChatModel:
    if p.name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=p.model, api_key=p.api_key, temperature=0.2, max_retries=2,
                                      timeout=REQUEST_TIMEOUT, max_output_tokens=MAX_OUTPUT_TOKENS)
    if p.name == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=p.model, api_key=p.api_key, temperature=0.2, max_retries=2,
                        timeout=REQUEST_TIMEOUT, max_tokens=MAX_OUTPUT_TOKENS)
    if p.name == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=p.model, api_key=p.api_key, base_url=OPENROUTER_BASE_URL, temperature=0.2,
                          max_retries=2, timeout=REQUEST_TIMEOUT, max_tokens=MAX_OUTPUT_TOKENS,
                          default_headers={"X-Title": "PocketDev AI"})
    if p.name == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=p.model, api_key=p.api_key, max_retries=2, timeout=REQUEST_TIMEOUT,
                             max_tokens=MAX_OUTPUT_TOKENS)
    raise ValueError(f"unknown provider {p.name}")


MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,120}$")
INVALID_MODEL_WARNING = "The saved OpenRouter model setting is not valid, so the default model is used."


def resolve_providers(providers: Sequence[ProviderConfig], task: dict | None) -> tuple[list[ProviderConfig], str | None]:
    """Apply the per-repo OpenRouter model from the task (`task["models"]["openrouter"]`, optional).

    A valid id overrides POCKETDEV_OPENROUTER_MODEL / the default. Whenever the OpenRouter model is not the
    default, "openrouter/free" is added right after it as an extra fallback (never duplicated).
    Returns (providers, warning or None). Invalid values are ignored with a warning.
    """
    warning = None
    chosen: str | None = None
    models: Any = (task or {}).get("models")
    raw = models.get("openrouter") if isinstance(models, dict) else None
    if raw is not None and raw != "":
        if isinstance(raw, str) and MODEL_ID_RE.match(raw.strip()):
            chosen = raw.strip()
        else:
            warning = INVALID_MODEL_WARNING
    out: list[ProviderConfig] = []
    for p in providers:
        if p.name != "openrouter":
            out.append(p)
            continue
        p = replace(p, model=chosen) if chosen else p
        out.append(p)
        if p.model != DEFAULT_OPENROUTER_MODEL:
            out.append(replace(p, model=DEFAULT_OPENROUTER_MODEL))
    return out, warning


def build_models(settings: Settings) -> list[BaseChatModel]:
    models = []
    for p in settings.providers:
        try:
            models.append(build_model(p))
            log.info("AI provider available: %s (%s)", p.name, p.model)
        except Exception as e:  # noqa: BLE001 - a broken provider must not block the others
            log.warning("could not set up provider %s: %s", p.name, type(e).__name__)
    return models


def chain(models: Sequence[BaseChatModel], tools: Sequence | None = None) -> Runnable:
    """Primary model with the others as fallbacks (each one bound to the same tools)."""
    if not models:
        raise ValueError("no models")
    bound = [m.bind_tools(list(tools)) if tools else m for m in models]
    return bound[0].with_fallbacks(bound[1:]) if len(bound) > 1 else bound[0]
