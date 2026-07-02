"""Grader/judge provider builder (TOOL_API.md §2, §6; AI Gauntlet grader fidelity).

Until 5.1.1 every AI Gauntlet tool graded on the on-demand local Ollama (zero
egress). This module lets the promptfoo grader instead use a closed/stronger
model (OpenAI / Anthropic / OpenAI-compatible), opt-in, with the grader key
delivered ENV-only by the orchestrator (never in the on-disk config file).

`build_grader_provider` returns the `redteam.provider` dict promptfoo consumes.
Backend selection:
  local-ollama        -> openai:chat:<model> against the local Ollama /v1 shim
                         (current behaviour; key is a synthetic sk-noop).
  openai              -> hosted OpenAI openai:chat:<model> (key via OPENAI_API_KEY).
  anthropic           -> hosted Anthropic anthropic:chat:<model> (key via ANTHROPIC_API_KEY).
  openai-compatible   -> openai:chat:<model> against a self-hosted baseUrl (key via OPENAI_API_KEY).

`grader_env` maps a resolved grader provider to the environment variables the
promptfoo subprocess needs so the grader can authenticate, derived from the
ENV-only GRADER_API_KEY the orchestrator injects. Payload generation stays
offline in every case (PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION=true is set
by the adapter, not here) — only the grader egresses.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("ai-attack-surface")

# Providers whose grader traffic leaves the container. Used to gate the egress
# strip + provenance flag in the promptfoo adapter.
EXTERNAL_PROVIDERS = {"openai", "anthropic", "openai-compatible"}


def is_external(grader_provider: str | None) -> bool:
    return (grader_provider or "local-ollama") in EXTERNAL_PROVIDERS


def _normalize(provider: str | None) -> str:
    return (provider or "local-ollama").strip().lower() or "local-ollama"


def build_grader_provider(grader_provider: str | None, grader_model: str | None,
                          grader_base_url: str | None, has_key: bool = False) -> dict:
    """Return the promptfoo `redteam.provider` dict for the chosen grader backend.

    `grader_model` is the model id the grader serves (e.g. gpt-4o,
    claude-opus-4-6, qwen2.5:7b). `grader_base_url` is only used by
    openai-compatible (self-hosted) and local-ollama. `has_key` is informational
    here (the key is injected by the adapter from ENV); it only guards the
    hosted OpenAI/Anthropic path so a misconfigured external run fails loudly
    rather than silently using a no-op key.
    """
    provider = _normalize(grader_provider)
    model = (grader_model or "").strip() or "default"

    if provider == "local-ollama":
        base = (grader_base_url or "").rstrip("/")
        return {
            "id": f"openai:chat:{model}",
            "config": {"apiBaseUrl": base + "/v1", "apiKey": "sk-noop"},
        }

    if provider == "openai":
        if not has_key:
            logger.warning("promptfoo openai grader has no key; grading will fail")
        return {"id": f"openai:chat:{model}"}

    if provider == "anthropic":
        if not has_key:
            logger.warning("promptfoo anthropic grader has no key; grading will fail")
        return {"id": f"anthropic:chat:{model}"}

    if provider == "openai-compatible":
        base = (grader_base_url or "").rstrip("/")
        return {
            "id": f"openai:chat:{model}",
            "config": {"apiBaseUrl": base + "/v1"},
        }

    # Unknown -> fall back to local Ollama so a bad value degrades, not aborts.
    logger.warning(f"promptfoo: unknown grader provider '{grader_provider}'; "
                   "falling back to local-ollama")
    base = (grader_base_url or "").rstrip("/")
    return {
        "id": f"openai:chat:{model}",
        "config": {"apiBaseUrl": base + "/v1", "apiKey": "sk-noop"},
    }
