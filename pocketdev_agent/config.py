"""Settings read from the environment (set by the PocketDev workflow file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Default model ids. Checked against the providers' docs on 2026-11-19:
#   Gemini:     https://aistudio.google.com/docs/models + /docs/pricing  ("gemini-3.8-flash": stable, free tier;
#               gemini-2.5-* are deprecated and get re-routed on 2026-12-01)
#   Groq:       https://console.groq.com/docs/models + /docs/tool-use  ("openai/gpt-oss-120b": production, tool use)
#   OpenRouter: https://openrouter.ai/api/v1/models  ("openrouter/free": router over free models, supports tools;
#               individual ":free" model ids churn often, so the router is the stable default)
#   Anthropic:  https://platform.claude.com/docs/en/about-claude/models/overview  ("claude-sonnet-5-5")
# Override any of them with the POCKETDEV_*_MODEL environment variables.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Provider order = fallback order (Groq free tier first, Claude optional paid last).
PROVIDERS = (
    ("groq", "POCKETDEV_GROQ_API_KEY", "POCKETDEV_GROQ_MODEL", DEFAULT_GROQ_MODEL),
    ("gemini", "POCKETDEV_GEMINI_API_KEY", "POCKETDEV_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    ("openrouter", "POCKETDEV_OPENROUTER_API_KEY", "POCKETDEV_OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
    ("anthropic", "POCKETDEV_ANTHROPIC_API_KEY", "POCKETDEV_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
)

# Values of these variables are scrubbed from every log line, event and command output.
SECRET_ENV_VARS = (
    "GITHUB_TOKEN", "GH_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN", "ACTIONS_RUNTIME_TOKEN",
    *(p[1] for p in PROVIDERS),
)

NO_KEY_MESSAGE = "Add an AI key for this project in the PocketDev app."


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    model: str


@dataclass
class Settings:
    api_base: str
    task_id: str
    phase: str
    audience: str = "pocketdev"
    repo_dir: Path = field(default_factory=Path.cwd)
    github_token: str = ""
    oidc_request_url: str = ""
    oidc_request_token: str = ""
    providers: list[ProviderConfig] = field(default_factory=list)
    server_url: str = "https://github.com"
    event_path: str = ""
    ref_name: str = ""
    temp_dir: str = ""

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = dict(os.environ if env is None else env)
        missing = [k for k in ("POCKETDEV_API_BASE", "POCKETDEV_TASK_ID") if not env.get(k)]
        if missing:
            raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")
        providers = []
        for name, key_var, model_var, default_model in PROVIDERS:
            key = (env.get(key_var) or "").strip()
            if key:
                providers.append(ProviderConfig(name, key, (env.get(model_var) or "").strip() or default_model))
        return cls(
            api_base=env["POCKETDEV_API_BASE"].rstrip("/"),
            task_id=env["POCKETDEV_TASK_ID"].strip(),
            phase=(env.get("POCKETDEV_PHASE") or "").strip().lower(),
            audience=(env.get("POCKETDEV_OIDC_AUDIENCE") or "pocketdev").strip(),
            repo_dir=Path(env.get("GITHUB_WORKSPACE") or os.getcwd()),
            github_token=env.get("GITHUB_TOKEN", ""),
            oidc_request_url=env.get("ACTIONS_ID_TOKEN_REQUEST_URL", ""),
            oidc_request_token=env.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", ""),
            providers=providers,
            server_url=(env.get("GITHUB_SERVER_URL") or "https://github.com").rstrip("/"),
            event_path=env.get("GITHUB_EVENT_PATH", ""),
            ref_name=env.get("GITHUB_REF_NAME", ""),
            temp_dir=env.get("RUNNER_TEMP", ""),
        )

    def secret_values(self) -> list[str]:
        vals = [self.github_token, self.oidc_request_token, *(p.api_key for p in self.providers)]
        return [v for v in vals if v]
