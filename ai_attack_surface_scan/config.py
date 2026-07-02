"""Run configuration for an AI Attack Surface job.

Mirrors the partial-recon config pattern: the orchestrator writes a JSON config
file and passes its path via env (AI_ATTACK_CONFIG). Everything also has an env
fallback so the container can be driven directly for testing.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("ai-attack-surface")


@dataclass
class Bounds:
    """Shared run bounds (the §3 'Run bounds' block). Stubbed enforcement in
    Step 2; the real safety layer fills in over later steps."""
    trials: int = 1
    asr_threshold: float = 0.3
    judge_model: str = ""
    max_turns: int = 4
    # RNG seed for tools that support it (garak --seed, pyrit) — part of the
    # reproducibility envelope. Default 0 (deterministic baseline).
    seed: int = 0
    # How many requests a tool fires at the target concurrently (garak
    # --parallel_attempts; promptfoo eval concurrency). Default 2 — low enough
    # that a single-CPU victim's queue doesn't back up past the request timeout,
    # high enough to make progress. Raise it for a fast/GPU target.
    parallelism: int = 2
    # Hard wall-clock budget (seconds) for the whole tool invocation; when
    # exceeded the subprocess is killed (proc.run_streamed). Default 36000 = 10h
    # — big multi-family garak sweeps and slow probes (atkgen) need room. The UI
    # exposes it in minutes.
    timeout: int = 36000
    # Hard guardrail floor — categories blocked before any payload leaves the
    # container, regardless of other settings (§10). Read-only floor.
    hard_blocked_categories: list[str] = field(
        default_factory=lambda: ["csam", "cbrn", "bioweapon"]
    )


@dataclass
class RunConfig:
    project_id: str
    user_id: str
    tool: str = "skeleton"
    run_id: str = ""
    # Explicit node selection from the UI picker (§2): list of
    # {"baseurl": ..., "path": ..., "method": ...}. Empty => load all applicable
    # AI endpoints (skeleton convenience; the UI always sends an explicit set).
    targets: list[dict] = field(default_factory=list)
    bounds: Bounds = field(default_factory=Bounds)
    # RoE confirmation gate — a launch is a confirmed action (§10).
    roe_confirmed: bool = False
    dry_run: bool = False
    # Local Ollama judge endpoint (set by the orchestrator at spawn); the tools
    # point their judge/grader at it for zero external egress.
    judge_base_url: str = ""
    # The model id the TARGET serves (for the tool's request body). Falls back to
    # recon's ai_model_ids / ai_model_family_guess when empty.
    target_model: str = ""
    # Target authentication (shared across all tools; the tool adapter applies it
    # when building its request to the target). Three modes resolve to these:
    #   none   -> api_key="", auth_header=""
    #   bearer -> api_key=<token>, auth_header="Authorization", auth_scheme="Bearer"
    #   custom -> api_key=<value>, auth_header=<name>, auth_scheme="" (or a scheme)
    api_key: str = ""           # the secret value (-> garak REST_API_KEY / $KEY)
    auth_header: str = ""       # header name to carry the key, e.g. Authorization
    auth_scheme: str = ""       # scheme prefix, e.g. "Bearer" (empty for raw key)
    # Optional per-tool probe/plugin override (e.g. garak probe families). Empty
    # => the adapter's default catalog.
    probes: list[str] = field(default_factory=list)
    # Free-text description of what the target app does. Shared across tools that
    # generate/grade from app context (giskard description, promptfoo purpose,
    # pyrit objective framing). Empty => a generic default per tool.
    target_purpose: str = ""
    # promptfoo: payload-mutation strategies (base64/rot13/leetspeak/...). Empty
    # => the adapter derives from chips / defaults to ["basic"].
    strategies: list[str] = field(default_factory=list)
    # pyrit: optional custom attack objective (the harmful goal). Empty => the
    # selected attack's built-in objectives.
    objective: str = ""
    # Grader/judge provider. "local-ollama" (default) keeps the historical
    # zero-egress behaviour: the grader runs on the on-demand local Ollama model
    # named in bounds.judge_model. Any other value ("openai" / "anthropic" /
    # "openai-compatible") routes the grader to a closed model whose key is
    # injected ENV-only as GRADER_API_KEY (or ANTHROPIC_API_KEY) by the
    # orchestrator — it never travels in this config file. The non-secret grader
    # fields below are persisted (needed to build the grader's provider block).
    grader_provider: str = "local-ollama"
    grader_model: str = ""      # when empty, falls back to bounds.judge_model
    grader_base_url: str = ""   # openai-compatible base; ignored for hosted providers


def load_config() -> RunConfig:
    """Build a RunConfig from the JSON config file (if any) + env fallbacks."""
    project_id = os.environ.get("PROJECT_ID", "")
    user_id = os.environ.get("USER_ID", "")
    run_id = os.environ.get("AI_ATTACK_RUN_ID", "")
    tool = os.environ.get("AI_ATTACK_TOOL", "skeleton")

    data: dict = {}
    # Inline JSON (AI_ATTACK_CONFIG_JSON) takes precedence over a file path
    # (AI_ATTACK_CONFIG); both are optional and fall back to env scalars.
    inline = os.environ.get("AI_ATTACK_CONFIG_JSON", "")
    cfg_path = os.environ.get("AI_ATTACK_CONFIG", "")
    if inline:
        try:
            data = json.loads(inline)
            logger.info("Loaded run config from AI_ATTACK_CONFIG_JSON")
        except Exception as e:
            logger.warning(f"Failed to parse AI_ATTACK_CONFIG_JSON: {e}")
    elif cfg_path and os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as fh:
                data = json.load(fh)
            logger.info(f"Loaded run config from {cfg_path}")
        except Exception as e:  # malformed config must not crash the spawn
            logger.warning(f"Failed to read AI_ATTACK_CONFIG {cfg_path}: {e}")

    bounds_data = data.get("bounds", {}) or {}
    bounds = Bounds(
        trials=int(bounds_data.get("trials", 1)),
        asr_threshold=float(bounds_data.get("asr_threshold", 0.3)),
        judge_model=str(bounds_data.get("judge_model", "")),
        max_turns=int(bounds_data.get("max_turns", 4)),
        seed=int(bounds_data.get("seed", 0) or 0),
        # Clamp to a sane range so a stray value can't hammer or stall the target.
        parallelism=max(1, min(16, int(bounds_data.get("parallelism", 2) or 2))),
        # Clamp 1 min .. 24h.
        timeout=max(60, min(86400, int(bounds_data.get("timeout", 36000) or 36000))),
    )

    return RunConfig(
        project_id=data.get("project_id", project_id),
        user_id=data.get("user_id", user_id),
        tool=data.get("tool", tool),
        run_id=data.get("run_id", run_id),
        targets=data.get("targets", []) or [],
        bounds=bounds,
        roe_confirmed=bool(data.get("roe_confirmed", False)),
        dry_run=bool(data.get("dry_run", False)),
        judge_base_url=str(data.get("judge_base_url", "") or ""),
        target_model=str(data.get("target_model", "") or ""),
        api_key=str(data.get("api_key", "") or ""),
        auth_header=str(data.get("auth_header", "") or ""),
        auth_scheme=str(data.get("auth_scheme", "") or ""),
        probes=data.get("probes", []) or [],
        target_purpose=str(data.get("target_purpose", "") or ""),
        strategies=data.get("strategies", []) or [],
        objective=str(data.get("objective", "") or ""),
        # Grader routing. The orchestrator places these at the top level of the
        # run config (mirroring judge_base_url); bounds is the secondary source
        # since the UI / API sends them inside `bounds`. The grader API key is
        # ENV-only (GRADER_API_KEY) and never appears here.
        grader_provider=str(data.get("grader_provider", bounds_data.get("grader_provider", "local-ollama")) or "local-ollama"),
        grader_model=str(data.get("grader_model", bounds_data.get("grader_model", "")) or ""),
        grader_base_url=str(data.get("grader_base_url", bounds_data.get("grader_base_url", "")) or ""),
    )
