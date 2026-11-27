import httpx
import pytest
from langchain_core.messages import AIMessage

from conftest import FakeToolModel, StaticTokens, make_task
from pocketdev_agent.api import ApiClient
from pocketdev_agent.config import DEFAULT_OPENROUTER_MODEL, ProviderConfig, Settings
from pocketdev_agent.models import INVALID_MODEL_WARNING, build_models, chain, resolve_providers
from pocketdev_agent.runner import run

GROQ = ProviderConfig("groq", "gk", "openai/gpt-oss-120b")
OR_DEFAULT = ProviderConfig("openrouter", "ok", DEFAULT_OPENROUTER_MODEL)
ANTH = ProviderConfig("anthropic", "ak", "claude-sonnet-5-5")


def names(providers):
    return [(p.name, p.model) for p in providers]


def test_override_applied_with_free_fallback_after_it():
    out, warn = resolve_providers([GROQ, OR_DEFAULT, ANTH], {"models": {"openrouter": "recursive/flint-v1.0:free"}})
    assert warn is None
    assert names(out) == [("groq", "openai/gpt-oss-120b"), ("openrouter", "recursive/flint-v1.0:free"),
                          ("openrouter", DEFAULT_OPENROUTER_MODEL), ("anthropic", "claude-sonnet-5-5")]
    assert out[1].api_key == out[2].api_key == "ok"


def test_chosen_equal_to_default_is_not_duplicated():
    out, warn = resolve_providers([GROQ, OR_DEFAULT], {"models": {"openrouter": DEFAULT_OPENROUTER_MODEL}})
    assert warn is None and names(out) == [("groq", "openai/gpt-oss-120b"), ("openrouter", DEFAULT_OPENROUTER_MODEL)]


def test_env_model_override_also_gets_free_fallback():
    env_or = ProviderConfig("openrouter", "ok", "some/env-model")
    out, _ = resolve_providers([env_or], {})
    assert names(out) == [("openrouter", "some/env-model"), ("openrouter", DEFAULT_OPENROUTER_MODEL)]


@pytest.mark.parametrize("bad", ["bad model id!", "x" * 121, "a;rm -rf", 123, ["a"], {"x": 1}, "  "])
def test_invalid_id_ignored_with_warning(bad):
    out, warn = resolve_providers([GROQ, OR_DEFAULT], {"models": {"openrouter": bad}})
    assert names(out) == [("groq", "openai/gpt-oss-120b"), ("openrouter", DEFAULT_OPENROUTER_MODEL)]
    assert warn == INVALID_MODEL_WARNING


@pytest.mark.parametrize("task", [{}, {"models": None}, {"models": {}}, {"models": {"openrouter": None}},
                                  {"models": {"openrouter": ""}}, {"models": "weird"}, None])
def test_missing_models_key_keeps_current_behaviour(task):
    out, warn = resolve_providers([GROQ, OR_DEFAULT, ANTH], task)
    assert warn is None and names(out) == names([GROQ, OR_DEFAULT, ANTH])


def test_no_openrouter_key_means_override_is_unused():
    out, warn = resolve_providers([GROQ], {"models": {"openrouter": "recursive/flint-v1.0:free"}})
    assert warn is None and names(out) == [("groq", "openai/gpt-oss-120b")]


def test_real_chain_uses_chosen_then_free_model():
    s = Settings(api_base="x", task_id="t", phase="build",
                 providers=resolve_providers([GROQ, OR_DEFAULT], {"models": {"openrouter": "vendor/model:free"}})[0])
    models = build_models(s)
    assert [getattr(m, "model_name", None) for m in models] == ["openai/gpt-oss-120b", "vendor/model:free",
                                                                DEFAULT_OPENROUTER_MODEL]
    r = chain(models)
    assert len(r.fallbacks) == 2


def test_chain_falls_back_when_chosen_model_fails():
    class Broken(FakeToolModel):
        def _generate(self, *a, **k):
            raise RuntimeError("404 model not found")

    r = chain([Broken(messages=iter([])), FakeToolModel(messages=iter([AIMessage("from free model")]))])
    assert r.invoke("hi").text == "from free model"


def _run_with(repo, fake_server, models_field, seen):
    task = make_task("plan")
    if models_field is not ...:
        task["models"] = models_field
    server = fake_server(task)
    api = ApiClient("https://api.example/api/v1", task["id"], StaticTokens(),
                    transport=httpx.MockTransport(server.handler), sleep=lambda s: None)
    settings = Settings(api_base="https://api.example/api/v1", task_id=task["id"], phase="plan", repo_dir=repo,
                        providers=[GROQ, OR_DEFAULT])

    def factory(s):
        seen.extend(names(s.providers))
        return [FakeToolModel(messages=iter([AIMessage('{"summary": "Plan.", "steps": ["a"], "files": []}')]))]

    assert run(settings, api, factory) == 0
    return server


def test_runner_passes_override_to_model_factory(repo, fake_server):
    seen = []
    _run_with(repo, fake_server, {"openrouter": "vendor/model:free"}, seen)
    assert seen == [("groq", "openai/gpt-oss-120b"), ("openrouter", "vendor/model:free"),
                    ("openrouter", DEFAULT_OPENROUTER_MODEL)]


def test_runner_invalid_override_warns_and_continues(repo, fake_server):
    seen = []
    server = _run_with(repo, fake_server, {"openrouter": "not valid!!"}, seen)
    assert seen == names([GROQ, OR_DEFAULT])
    assert {"type": "log", "agent": "system", "message": INVALID_MODEL_WARNING} in server.events
    assert server.plans


def test_runner_without_models_key(repo, fake_server):
    seen = []
    server = _run_with(repo, fake_server, ..., seen)
    assert seen == names([GROQ, OR_DEFAULT]) and server.plans
