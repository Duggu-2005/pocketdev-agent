# pocketdev-agent

The PocketDev AI agent runner. It runs inside a GitHub Actions workflow **in the user's own repository**
(installed by PocketDev as `.github/workflows/pocketdev.yml`), never on PocketDev's servers.

- **plan phase**: the Planner looks at the repo and sends a short plan + optional design questions to the PocketDev API.
- **build phase**: Coder ⇄ Tester (fix loop) → Reviewer → commit → Summarizer (from the real `git diff`) →
  push to the feature branch (never the base/default branch, never force) → result sent to the API.

Built with LangGraph (agent graph) and LangChain chat models (Gemini → Groq → OpenRouter → Claude fallbacks,
bring-your-own-key).

## Install

The workflow installs it pinned to a commit:

```
pip install "pocketdev-agent @ git+https://github.com/Duggu-2005/pocketdev-agent@<commit-sha>"
python -m pocketdev_agent
```

Requires Python 3.13+ (the workflow uses `ubuntu-latest` with Python 3.13).

## Environment

| Variable | Meaning |
|---|---|
| `POCKETDEV_API_BASE` | API base URL, e.g. `https://pocketdev-ai.antideploy.app/api/v1` |
| `POCKETDEV_OIDC_AUDIENCE` | OIDC audience (default `pocketdev`) |
| `POCKETDEV_TASK_ID`, `POCKETDEV_PHASE` | Task UUID and `plan` / `build` |
| `GITHUB_TOKEN` | Used only for `git fetch` / `git push` of the feature branch |
| `ACTIONS_ID_TOKEN_REQUEST_URL`, `ACTIONS_ID_TOKEN_REQUEST_TOKEN` | Provided by Actions when the job has `id-token: write` |
| `POCKETDEV_GEMINI_API_KEY`, `POCKETDEV_GROQ_API_KEY`, `POCKETDEV_OPENROUTER_API_KEY`, `POCKETDEV_ANTHROPIC_API_KEY` | AI keys (any may be empty; at least one needed) |
| `POCKETDEV_GEMINI_MODEL`, `POCKETDEV_GROQ_MODEL`, `POCKETDEV_OPENROUTER_MODEL`, `POCKETDEV_ANTHROPIC_MODEL` | Optional model overrides (defaults in `pocketdev_agent/config.py`) |

## Development

```
python -m venv .venv
.venv/Scripts/python -m pip install langgraph==1.2.23 langchain-core==1.8.3 langchain-google-genai==4.5.1 \
    langchain-groq==1.2.1 langchain-openai==1.8.3 langchain-anthropic==1.7.11 httpx==0.28.1 pytest==9.2.0
.venv/Scripts/python -m pytest -q
```

Tests are fully offline: fake chat models, a fake PocketDev API (`httpx.MockTransport`) and a local bare git
repo as `origin`.
