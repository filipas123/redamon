import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

interface RouteParams {
  params: Promise<{ projectId: string }>
}

// External grader provider types we expose for grading (mirrors the start route).
const GRADER_BACKEND_BY_TYPE: Record<string, string> = {
  openai: 'openai',
  anthropic: 'anthropic',
  openai_compatible: 'openai-compatible',
  openrouter: 'openai-compatible',
}

function maskSecret(value: string): string {
  if (!value || value.length <= 4) return value ? '••••' : ''
  return '••••••••' + value.slice(-4)
}

// GET /api/ai-attack-surface/{projectId}/graders
// Lists the project owner's stored LLM providers that can serve as an external
// grader (openai / anthropic / openai-compatible / openrouter). Keys are masked
// — the actual key is resolved server-side at launch time and travels to the
// orchestrator only via the X-Grader-Key header. Lets the UI populate the
// grader dropdown without ever holding a key in the browser.
export async function GET(_request: NextRequest, { params }: RouteParams) {
  const { projectId } = await params
  const project = await prisma.project.findUnique({
    where: { id: projectId },
    select: { id: true, userId: true },
  })
  if (!project) {
    return NextResponse.json({ error: 'Project not found' }, { status: 404 })
  }

  const providers = await prisma.userLlmProvider.findMany({
    where: { userId: project.userId },
    select: { id: true, providerType: true, name: true, modelIdentifier: true, baseUrl: true, apiKey: true },
  })
  const graders = providers
    .filter((p) => GRADER_BACKEND_BY_TYPE[p.providerType])
    .map((p) => ({
      id: p.id,
      name: p.name,
      backend: GRADER_BACKEND_BY_TYPE[p.providerType],
      providerType: p.providerType,
      modelIdentifier: p.modelIdentifier,
      baseUrl: p.baseUrl,
      hasKey: !!p.apiKey,
      apiKeyMasked: maskSecret(p.apiKey),
    }))
  return NextResponse.json({ graders, count: graders.length })
}
