'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import type { AiAttackRunState, AiFinding, AiTarget } from '@/lib/aiAttackSurface'

export interface AiAttackLogLine {
  log: string
  phase?: string | null
  phaseNumber?: number | null
  isPhaseStart?: boolean
  level: string
  timestamp: string
}

export interface LaunchTarget {
  baseurl: string
  path: string
  method?: string
  interface_type?: string   // custom targets carry their own shape
  model?: string
  custom?: boolean
}

export interface LaunchPayload {
  tool: string
  targets: LaunchTarget[]
  bounds: {
    trials?: number
    asr_threshold?: number
    judge_model?: string
    max_turns?: number
    seed?: number
    parallelism?: number
    timeout?: number
    // Grader fidelity (promptfoo). Default local-ollama (zero egress). An
    // external grader egresses the grader's verdict traffic to a closed model.
    grader_provider?: string          // local-ollama | openai | anthropic | openai-compatible
    grader_provider_id?: string       // a stored UserLlmProvider.id (key resolved server-side)
    grader_model?: string             // the grader model id
    grader_consent?: boolean          // operator opt-in (RoE-gated)
  }
  roe_confirmed: boolean
  dry_run?: boolean
  probes?: string[]
  strategies?: string[]   // promptfoo: payload-mutation strategies
  objective?: string      // pyrit: optional custom attack objective
  target_model?: string
  // Free-text app description (shared): improves giskard/promptfoo/pyrit relevance.
  target_purpose?: string
  // Target auth (shared): resolved from the UI auth mode.
  api_key?: string
  auth_header?: string
  auth_scheme?: string
}

export function useAiAttackSurface(projectId: string | null) {
  const [targets, setTargets] = useState<AiTarget[]>([])
  const [findings, setFindings] = useState<AiFinding[]>([])
  const [run, setRun] = useState<AiAttackRunState | null>(null)
  const [logs, setLogs] = useState<AiAttackLogLine[]>([])
  const [phase, setPhase] = useState<{ name: string | null; num: number | null }>({ name: null, num: null })
  const [launching, setLaunching] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [loadingTargets, setLoadingTargets] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const esRef = useRef<EventSource | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopStream = useCallback(() => {
    esRef.current?.close()
    esRef.current = null
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const loadTargets = useCallback(async () => {
    if (!projectId) return
    setLoadingTargets(true)
    try {
      const r = await fetch(`/api/ai-attack-surface/${projectId}/targets`)
      const d = await r.json()
      setTargets(d.targets || [])
    } catch {
      setTargets([])
    } finally {
      setLoadingTargets(false)
    }
  }, [projectId])

  const loadFindings = useCallback(async () => {
    if (!projectId) return
    try {
      const r = await fetch(`/api/ai-attack-surface/${projectId}/findings`)
      const d = await r.json()
      setFindings(d.findings || [])
    } catch {
      /* soft */
    }
  }, [projectId])

  const pollStatus = useCallback(async (runId: string) => {
    if (!projectId) return
    try {
      const r = await fetch(`/api/ai-attack-surface/${projectId}/${runId}/status`)
      const d: AiAttackRunState = await r.json()
      setRun(d)
      if (d.status === 'completed' || d.status === 'error') {
        stopStream()
        loadFindings()
      }
    } catch {
      /* soft */
    }
  }, [projectId, stopStream, loadFindings])

  // Open the SSE log stream + status polling for a run. Reusable by launch AND
  // by reconnect-on-mount (so a page refresh re-attaches to an in-flight scan).
  const attachStream = useCallback((runId: string) => {
    if (!projectId) return
    stopStream()
    const es = new EventSource(`/api/ai-attack-surface/${projectId}/${runId}/logs`)
    esRef.current = es
    // The server replays the full log history on (re)connect; reset our view on
    // open so a refresh/reconnect restores everything without duplicating lines.
    es.onopen = () => { setLogs([]); setPhase({ name: null, num: null }) }
    es.addEventListener('log', (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data) as AiAttackLogLine
        setLogs((prev) => [...prev, data])
        if (data.isPhaseStart && data.phase) {
          setPhase({ name: data.phase, num: data.phaseNumber ?? null })
        }
      } catch {
        /* ignore malformed line */
      }
    })
    es.onerror = () => es.close()
    pollRef.current = setInterval(() => pollStatus(runId), 3000)
  }, [projectId, stopStream, pollStatus])

  // On mount / project change, re-attach to any still-running scan so the page
  // is stateful across reloads (status + phase + live logs all restored).
  const reconnect = useCallback(async () => {
    if (!projectId || esRef.current) return
    try {
      const r = await fetch(`/api/ai-attack-surface/${projectId}/all`)
      const d = await r.json()
      const active = (d.runs || []).find(
        (x: AiAttackRunState) => x.status === 'running' || x.status === 'starting')
      // Re-check after the await: a launch during the fetch may have attached a
      // stream already (avoid a double-attach race).
      if (active && !esRef.current) {
        setRun(active)
        attachStream(active.run_id)
      }
    } catch {
      /* soft */
    }
  }, [projectId, attachStream])

  const launch = useCallback(async (payload: LaunchPayload): Promise<AiAttackRunState | null> => {
    if (!projectId) return null
    setLaunching(true)
    setStopping(false)
    setError(null)
    setLogs([])
    setFindings([])
    setPhase({ name: null, num: null })
    stopStream()
    try {
      const r = await fetch(`/api/ai-attack-surface/${projectId}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || 'Failed to launch')
      setRun(d)
      attachStream(d.run_id)
      return d
    } catch (e) {
      const m = e instanceof Error ? e.message : 'Launch failed'
      setError(m)
      return null
    } finally {
      setLaunching(false)
    }
  }, [projectId, attachStream, stopStream])

  const stop = useCallback(async () => {
    if (!projectId || !run?.run_id) return
    // React immediately: flag stopping + freeze the log stream so the operator
    // sees instant feedback instead of waiting on the backend kill.
    setStopping(true)
    stopStream()
    try {
      const r = await fetch(`/api/ai-attack-surface/${projectId}/${run.run_id}/stop`, { method: 'POST' })
      const d: AiAttackRunState = await r.json()
      if (d && d.status) setRun(d)            // use the authoritative stopped state
      else setRun((prev) => (prev ? { ...prev, status: 'idle' } : prev))
    } catch {
      // Even if the call fails, drop out of the running state locally.
      setRun((prev) => (prev ? { ...prev, status: 'idle' } : prev))
    } finally {
      setStopping(false)
    }
    loadFindings()
  }, [projectId, run, stopStream, loadFindings])

  useEffect(() => () => stopStream(), [stopStream])

  // Re-attach to an in-flight scan on mount / project change (statefulness).
  useEffect(() => { reconnect() }, [reconnect])

  return {
    targets, findings, run, logs, phase, launching, stopping, loadingTargets, error,
    loadTargets, loadFindings, launch, stop, reconnect,
  }
}
