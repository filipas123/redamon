import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'
import { orchestratorFetch } from '@/lib/orchestrator'

const RECON_ORCHESTRATOR_URL = process.env.RECON_ORCHESTRATOR_URL || 'http://localhost:8010'
const WEBAPP_URL = process.env.WEBAPP_URL || 'http://localhost:3000'

interface RouteParams {
  params: Promise<{ projectId: string }>
}

// External-grader provider types we accept for grading. Maps a UserLlmProvider
// providerType to the grader backend the promptfoo adapter understands.
const GRADER_BACKEND_BY_TYPE: Record<string, string> = {
  openai: 'openai',
  anthropic: 'anthropic',
  openai_compatible: 'openai-compatible',
  openrouter: 'openai-compatible',
}

// Resolve an external grader from the operator's stored UserLlmProvider.
// Returns { provider, model, baseUrl, backend } or null if not external-capable
// / not owned by this project's user. The API key is read here server-side and
// travels to the orchestrator only via the X-Grader-Key header — it never
// enters the JSON body (which gets persisted to /tmp/redamon) nor the response.
async function resolveExternalGrader(userId: string, providerId: string, model?: string) {
  const provider = await prisma.userLlmProvider.findFirst({
    where: { id: providerId, userId },
    select: { id: true, providerType: true, apiKey: true, baseUrl: true, modelIdentifier: true },
  })
  if (!provider) return null
  const backend = GRADER_BACKEND_BY_TYPE[provider.providerType]
  if (!backend) return null
  const graderModel = (model || provider.modelIdentifier || '').trim()
  const baseUrl = (provider.baseUrl || '').trim()
  return { provider, backend, graderModel, baseUrl }
}

// POST /api/ai-attack-surface/{projectId}/start
// Launch one AI Attack Surface tool against the selected AI nodes.
export async function POST(request: NextRequest, { params }: RouteParams) {
  try {
    const { projectId } = await params
    const body = await request.json()

    const project = await prisma.project.findUnique({
      where: { id: projectId },
      select: { id: true, userId: true },
    })
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 })
    }

    // Grader routing. Default is local-ollama (zero egress). An external grader
    // is opt-in: the UI sends grader_provider + grader_provider_id (a stored
    // UserLlmProvider) + grader_consent. We resolve the key server-side here and
    // forward it ONLY via the X-Grader-Key header — never in the JSON body.
    const bounds = { ...(body.bounds || {}) }
    const requestedProvider = String(bounds.grader_provider || 'local-ollama')
    const providerId = String(bounds.grader_provider_id || '')
    let graderKey = ''

    if (requestedProvider !== 'local-ollama') {
      if (!providerId) {
        return NextResponse.json(
          { error: 'External grader requires a stored provider (grader_provider_id)' },
          { status: 400 },
        )
      }
      const resolved = await resolveExternalGrader(
        project.userId, providerId, bounds.grader_model as string | undefined,
      )
      if (!resolved) {
        return NextResponse.json(
          { error: 'Selected grader provider not found or not supported for grading' },
          { status: 400 },
        )
      }
      graderKey = resolved.provider.apiKey || ''
      if (!graderKey) {
        return NextResponse.json(
          { error: 'Selected grader provider has no API key configured' },
          { status: 400 },
        )
      }
      // Stamp the resolved non-secret routing into bounds for the orchestrator.
      // grader_provider_id is webapp-only (it resolved the key) — drop it so the
      // orchestrator / scan container never sees an opaque id it can't use.
      bounds.grader_provider = resolved.backend
      bounds.grader_model = resolved.graderModel
      bounds.grader_base_url = resolved.baseUrl
      bounds.grader_consent = !!bounds.grader_consent
      delete bounds.grader_provider_id
    }

    const orchHeaders: Record<string, string> = { 'Content-Type': 'application/json' }
    if (graderKey) orchHeaders['X-Grader-Key'] = graderKey

    const response = await orchestratorFetch(`${RECON_ORCHESTRATOR_URL}/ai-attack-surface/${projectId}/start`, {
      method: 'POST',
      headers: orchHeaders,
      body: JSON.stringify({
        project_id: projectId,
        user_id: project.userId,
        webapp_api_url: WEBAPP_URL,
        tool: body.tool || 'garak',
        targets: body.targets || [],
        bounds,
        roe_confirmed: body.roe_confirmed ?? false,
        dry_run: body.dry_run ?? false,
        probes: body.probes || [],
        strategies: body.strategies || [],
        objective: body.objective || '',
        target_model: body.target_model || '',
        target_purpose: body.target_purpose || '',
        api_key: body.api_key || '',
        auth_header: body.auth_header || '',
        auth_scheme: body.auth_scheme || '',
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      return NextResponse.json(
        { error: errorData.detail || 'Failed to start AI Gauntlet scan' },
        { status: response.status },
      )
    }

    // Never echo the grader key back; the orchestrator response carries no key.
    return NextResponse.json(await response.json())
  } catch (error) {
    console.error('Error starting AI Attack Surface scan:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 },
    )
  }
}
