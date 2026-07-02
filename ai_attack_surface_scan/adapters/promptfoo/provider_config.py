"""Build the promptfoo JSON config from a recon Target (TOOL_API.md §2, §6).

The target is an HTTP provider; the request body + response transform are derived
from `ai_interface_type` (same matrix as garak). The redteam.provider (grader +
generator) is forced to the local Ollama so there is zero external egress.

Config is emitted as JSON (promptfoo accepts JSON configs), so the shared spine
needs no YAML dependency.
"""
from __future__ import annotations

# Env var the target auth key is injected through (never written into the config).
TARGET_KEY_ENV = "REDAMON_TARGET_KEY"

# ai_interface_type -> (body dict with '{{prompt}}', transformResponse expr).
# '<MODEL>' is replaced with the resolved model id.
_TEMPLATES: dict[str, tuple[dict, str]] = {
    "llm-chat": (
        {"model": "<MODEL>", "messages": [{"role": "user", "content": "{{prompt}}"}]},
        "json.choices[0].message.content"),
    "llm-completion": (
        {"model": "<MODEL>", "prompt": "{{prompt}}"},
        "json.choices[0].text"),
    "ollama-chat": (
        {"model": "<MODEL>", "messages": [{"role": "user", "content": "{{prompt}}"}],
         "stream": False},
        "json.message.content"),
    "ollama-generate": (
        {"model": "<MODEL>", "prompt": "{{prompt}}", "stream": False},
        "json.response"),
    "anthropic": (
        {"model": "<MODEL>", "max_tokens": 512,
         "messages": [{"role": "user", "content": "{{prompt}}"}]},
        "json.content[0].text"),
}


def _interface(target, path: str) -> str:
    """Resolve the template family from the explicit type, else from the path."""
    it = (getattr(target, "ai_interface_type", None) or "").lower()
    if it in ("llm-chat", "chat"):
        return "llm-chat"
    if it in ("llm-completion", "completion"):
        return "llm-completion"
    p = (path or "").lower()
    if "/api/chat" in p:
        return "ollama-chat"
    if "/api/generate" in p:
        return "ollama-generate"
    if "/v1/messages" in p:
        return "anthropic"
    if "/completions" in p and "/chat/" not in p:
        return "llm-completion"
    return "llm-chat"


def _resolve_model(target, model: str | None) -> str:
    ids = getattr(target, "ai_model_ids", None)
    first = ids[0] if isinstance(ids, list) and ids else (ids if isinstance(ids, str) else None)
    return model or first or getattr(target, "ai_model_family_guess", None) or "default"


def _sub_model(obj, model: str):
    if isinstance(obj, dict):
        return {k: _sub_model(v, model) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sub_model(v, model) for v in obj]
    if obj == "<MODEL>":
        return model
    return obj


def build_target_provider(target, model: str | None = None,
                          auth_header: str | None = None,
                          auth_scheme: str | None = None) -> dict:
    """The HTTP provider dict pointing at the in-scope target."""
    baseurl = (getattr(target, "baseurl", "") or "").rstrip("/")
    path = getattr(target, "path", "/") or "/"
    if not path.startswith("/"):
        path = "/" + path
    url = baseurl + path

    family = _interface(target, path)
    body_tmpl, transform = _TEMPLATES[family]
    body = _sub_model(body_tmpl, _resolve_model(target, model))

    headers = {"Content-Type": "application/json"}
    if auth_header:
        scheme = (auth_scheme + " ") if auth_scheme else ""
        headers[auth_header] = f"{scheme}{{{{env.{TARGET_KEY_ENV}}}}}"

    return {
        "id": "https",
        "label": "redamon-target",
        "config": {
            "url": url,
            "method": "POST",
            "headers": headers,
            "body": body,
            "transformResponse": transform,
        },
    }


def build_config(target, *, plugins: list[str], strategies: list[str],
                 num_tests: int, judge_base_url: str, judge_model: str,
                 purpose: str = "A general-purpose chat assistant.",
                 model: str | None = None, auth_header: str | None = None,
                 auth_scheme: str | None = None,
                 grader_provider: str = "local-ollama",
                 grader_model: str | None = None,
                 grader_base_url: str | None = None,
                 grader_has_key: bool = False) -> dict:
    """Full promptfoo config: target provider + redteam (plugins/strategies) +
    grader provider.

    The grader provider is built by `adapters.grader.build_grader_provider`.
    For `local-ollama` (default) it points at the local Ollama /v1 shim with a
    synthetic key (zero egress, historical behaviour). For an external backend
    (openai / anthropic / openai-compatible) it points at the hosted/self-hosted
    model; the real key is injected ENV-only by the adapter, never written here.
    `grader_model` falls back to `judge_model` so the default local path is
    unchanged when no explicit grader model is given.
    """
    from adapters.grader import build_grader_provider

    resolved_grader_model = (grader_model or judge_model or "").strip()
    # local-ollama uses the on-demand Ollama URL the orchestrator supplied; an
    # explicit grader_base_url overrides it (e.g. a self-hosted openai-compatible).
    local_base = (grader_base_url or judge_base_url or "").rstrip("/") or None

    grader = build_grader_provider(
        grader_provider, resolved_grader_model, local_base, has_key=grader_has_key)
    return {
        "description": "redamon ai-attack-surface promptfoo run",
        "targets": [build_target_provider(target, model, auth_header, auth_scheme)],
        "redteam": {
            "purpose": purpose,
            "numTests": int(num_tests),
            "plugins": [{"id": p} for p in plugins],
            "strategies": [{"id": s} for s in strategies],
            "provider": grader,
        },
    }
