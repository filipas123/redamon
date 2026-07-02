# Plan: Promptfoo External-Grader Fidelity (zero-egress opt-out)

## Context / problem

The AI Gauntlet grader/judge is architecturally locked to a **local Ollama open-weight
model** across all four tools, and the promptfoo default judge was *downgraded* in 5.1.1
from `qwen2.5:7b` to `qwen2.5:0.5b`. A 0.5B model grading jailbreak/toxicity success is a
genuine false-ASR source that undermines professional report accuracy.

Evidence (grounded in code):
- `ai_attack_surface_scan/adapters/promptfoo/provider_config.py:122` — grader emitted as
  `openai:chat:<judge_model>` with `apiBaseUrl` pinned to local Ollama `/v1`.
- `ai_attack_surface_scan/adapters/promptfoo/adapter.py:168` — `OPENAI_API_KEY` stripped
  from the subprocess env + `_OFFLINE_ENV` (remote-generation/telemetry/update disabled).
- `recon_orchestrator/container_manager.py:1378` — `local_llm_manager.ensure_up(judge_model)`
  sets `run_config["judge_base_url"]` to the local Ollama URL; scan container is
  `network_mode="host"` (it already live-fetches HuggingFace datasets → egress exists).
- `webapp/src/app/ai-attack-surface/page.tsx:44` — judge default `qwen2.5:0.5b`.
- Sibling tools share the lock: `garak/runner.py:46-50`, `giskard/giskard_run.py:74`
  (`f"ollama/{model}"`), `pyrit/pyrit_run.py:68`.

"Stable" verdict: v5.2.0 (2026-06-30) is active and test-backed; the issue is grader
*fidelity*, not platform instability.

## Goal

Let the **promptfoo** grader use a closed/stronger model (OpenAI / Anthropic /
OpenAI-compatible) instead of the locked local Ollama, so ASR reporting is defensible.
Zero-egress stays the **default**; external grading is an explicit, RoE-gated, recorded
opt-in. garak/giskard/pyrit are a documented follow-on (same pattern).

## Decisions (locked)

1. **Scope**: promptfoo only now; garak/giskard/pyrit as a sequenced follow-on (each
   hardcodes `ollama/<model>`).
2. **Grader key source**: reuse the existing `UserLlmProvider` table
   (`providerType` openai/anthropic/openai-compatible + `apiKey` + `baseUrl`,
   server-stored). Resolved server-side in the webapp route; passed to the scan container
   as **ENV only** — never in the JSON config file on `/tmp/redamon`, never through the
   browser.
3. **Egress consent**: per-launch "Allow external grader" checkbox, RoE-gated (launch
   blocked unless checked), UI egress warning, and `ai_grader_egress=true` + grader
   provider/model stamped on each `Vulnerability` + evidence line.
4. **Grader backends**: `local-ollama` (default, unchanged) | `openai` | `anthropic` |
   `openai-compatible`.
5. **Sequencing vs heuristic**: ship the closed-model grader first (ground truth +
   provenance). A local gold-corpus + ML heuristic grader is a **follow-on phase**, gated
   on a measured agreement test vs the closed judge — not a substitute (it can't bootstrap
   its own labels; accuracy is bound by label fidelity, not disk capacity).

## Data flow (after change)

UI `page.tsx` → `useAiAttackSurface.launch({..., grader_*})` →
`POST /api/ai-attack-surface/[projectId]/start` (webapp resolves grader key from
`UserLlmProvider`) → orchestrator `/ai-attack-surface/[projectId]/start` →
`start_ai_attack_surface` (skips `local_llm_manager.ensure_up` when grader != local;
passes grader key as ENV) → scan container (`AI_ATTACK_CONFIG` + grader key ENV) →
`config.load_config` → `main.run_tool` → `promptfoo adapter.run` →
`provider_config.build_config` emits the right `redteam.provider` → `adapter._invoke`
conditionally stops stripping the grader key (keeps payload-generation offline).

## Tasks

### 1. Shared grader-provider abstraction (scan container)
- `ai_attack_surface_scan/config.py`: add to `RunConfig`: `grader_provider`
  (`"local-ollama"` default), `grader_model`, `grader_base_url`. `grader_api_key` is
  **env-only** (`GRADER_API_KEY`); never persisted in `RunConfig`/config file. Extend
  `load_config()` to read `bounds.grader_*`.
- New `adapters/grader.py`: one helper `build_grader_provider(grader_provider, grader_model,
  grader_base_url, has_key)` returning the promptfoo `redteam.provider` dict per backend:
  - `local-ollama` → `{"id":"openai:chat:<model>","config":{"apiBaseUrl":<base>/v1,"apiKey":"sk-noop"}}` (current behavior).
  - `openai` → `{"id":"openai:chat:<model>"}` (no apiBaseUrl; key via `OPENAI_API_KEY` env).
  - `anthropic` → `{"id":"anthropic:chat:<model>"}` (key via `ANTHROPIC_API_KEY` env).
  - `openai-compatible` → `{"id":"openai:chat:<model>","config":{"apiBaseUrl":<base>/v1}}` (key via `OPENAI_API_KEY` env).

### 2. promptfoo adapter wiring
- `adapters/promptfoo/provider_config.py`: `build_config` takes grader fields and uses
  `build_grader_provider(...)` instead of the hardcoded local block.
- `adapters/promptfoo/adapter.py`:
  - `run()` reads `grader_*` from `bounds`/config; passes to `build_config`.
  - `_invoke()`: egress guard becomes conditional — when `grader_provider == "local-ollama"`
    keep current behavior (strip `OPENAI_API_KEY`, set `_OFFLINE_ENV`). When external,
    inject the grader key env (`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`) from `GRADER_API_KEY`
    and **keep** `PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION=true` (payloads stay local;
    only the grader egresses). Target key handling unchanged.
  - Stamp provenance on each `Finding`: `ai_grader_provider`, `ai_grader_model`,
    `ai_grader_egress` (bool), plus a line in `evidence`.

### 3. Findings + graph schema provenance
- `normalizer.py`: add `ai_grader_provider`, `ai_grader_model`, `ai_grader_egress` to
  `Finding` and `_props()` (COALESCE so local-only runs stay null).
- `readmes/GRAPH.SCHEMA.md`: document the three new `Vulnerability` properties.

### 4. Orchestrator
- `recon_orchestrator/container_manager.py:start_ai_attack_surface`: when
  `bounds.grader_provider != "local-ollama"`, skip `local_llm_manager.ensure_up` / lease;
  set `run_config["grader_base_url"]` from the resolved provider; pass `GRADER_API_KEY`
  (and `ANTHROPIC_API_KEY` when anthropic) into the scan container `environment`.
- `recon_orchestrator/api.py` (the start route): accept + forward `bounds.grader_*`;
  enforce RoE when an external grader is requested (the launch is already RoE-gated; the
  consent flag below is the new gate).

### 5. Webapp (server-side key resolution)
- `webapp/src/app/api/ai-attack-surface/[projectId]/start/route.ts`: accept
  `body.grader_provider`, `body.grader_provider_id` (the `UserLlmProvider.id`). When
  external, `prisma.userLlmProvider.findUnique` → resolve `apiKey`/`baseUrl`/
  `providerType`; send `grader_provider`/`grader_model`/`grader_base_url` to the
  orchestrator and the **key out of band via a server-only channel** (orchestrator env or
  a one-shot header — match whatever `orchestratorFetch` already does for secrets; do NOT
  put the key in the JSON body that the browser sent). Reject if the provider isn't owned
  by `project.userId`.
- Add a route test: external grader without a valid owned provider → 400; key never
  echoed back in the response.

### 6. UI
- `webapp/src/lib/aiAttackSurface.ts`: extend `AiAttackRunState`/launch payload with
  `grader_provider`, `grader_model`, `grader_provider_id`, `grader_consent`.
- `webapp/src/app/ai-attack-surface/page.tsx`: in the Run-bounds block add a "Grader"
  selector (`local-ollama` | provider dropdown sourced from the user's `UserLlmProvider`s
  filtered to openai/anthropic/openai-compatible). When non-local: show the egress warning
  + a consent checkbox ("Allow external grader — attack transcripts will be sent to
  <provider>. Required for launch."). Block `canLaunch` unless consent checked. Default
  stays `local-ollama`.
- Update the existing `useAiAttackSurface` launch type + the route-forwarded fields.

## Follow-on phase (documented, not built now)

Local gold-corpus + ML heuristic grader, Dockerized on the operator storage (e.g. 2TB
RAID — solves capacity, not the inference-throughput that fidelity actually needs):
- Use the closed-model grader (phase 1) to label a held-out gold corpus offline.
- Train/heuristic-grade locally; keep egress off for repeat grading.
- **Gate**: ship only after a measured agreement test vs the closed judge on a held-out
  set clears a chosen threshold (e.g. false-ASR delta < X). Never the sole judge until then.
- The "Epistemological Data" method stays **out of scope** until a concrete spec (input,
  signal, scoring) is provided; it cannot be planned or claimed against this codebase yet.

## Risks / mitigations

- **Reopens the marketed zero-egress guarantee** → opt-in + per-launch consent + provenance;
  default unchanged so existing zero-egress runs are byte-identical.
- **Grader sees attack transcripts** (data-handling) → consent text states it; recorded on
  findings for report disclosure.
- **Provider/model id conventions** → `build_grader_provider` validates the model string
  fits promptfoo's `<provider>:chat:<model>` form; fail-soft (warn + skip) on mismatch.
- **Key leakage** → key is ENV-only, server-resolved, never in config file/response.

## Validation

- Extend `ai_attack_surface_scan/adapters/promptfoo/tests/test_promptfoo_adapter.py`:
  - `build_config` emits the correct `redteam.provider` per backend
    (`local-ollama`/`openai`/`anthropic`/`openai-compatible`).
  - `_invoke` strips keys + sets offline env **only** when `grader_provider=="local-ollama"`;
    injects grader key env otherwise; `PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION` stays
    `true` in both.
  - Findings carry `ai_grader_provider`/`ai_grader_model`/`ai_grader_egress`.
  - Existing zero-egress path (local-ollama) unchanged → regression-green.
- `ai_attack_surface_scan/tests/test_main.py`: promptfoo dispatch passes new grader fields.
- Webapp route test: external grader without an owned `UserLlmProvider` → 400; key not in
  response body.
- Live: optional `PROMPTFOO_LIVE` smoke against a real provider (gated, opt-in) confirming
  grading runs end-to-end with a closed model.
- Graph schema doc updated; new properties queryable via the NL→Cypher agent prompt.

## Open questions (for implementation agent)

- Exact channel for passing the server-resolved grader key to the orchestrator without it
  landing in the forwarded JSON body — confirm against how `orchestratorFetch` / the
  `ORCHESTRATOR_API_KEY` is handled today (prefer an orchestrator-side env injection the
  scan container inherits, over a header the webapp route sends).
- Whether the egress consent should also be recorded as a project-level audit event
  (recommended) — confirm the audit surface used elsewhere.
