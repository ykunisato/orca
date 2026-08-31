import type {
  AgentSessionAcquisition,
  StructuredAgentSessionAcquireInput
} from '../native-chat/agent-session-wire/structured-agent-session-adapter'
import {
  AgentSessionAcquisitionRefusal,
  AgentSessionPreSpawnError
} from '../native-chat/agent-session-wire/structured-agent-session-adapter'
import {
  closeFailedCodexAcquisition,
  stopSupersededCodexAcquisition
} from './codex-structured-acquisition-lifecycle'
import { createCodexJournalTranslator } from './codex-structured-journal-translation'
import {
  openCodexAppServerConnection,
  type CodexAppServerServerRequest
} from './codex-app-server-connection'
import { codexProcessIdentity, codexProviderHandleLink } from './codex-structured-owner-identity'
import { buildCodexStructuredChildEnvironment } from './codex-structured-child-environment'
import { openCodexThread } from './codex-structured-thread-open'
import {
  closeCodexPublishedSession,
  handleCodexSessionExit
} from './codex-structured-session-close'
import {
  reportedCodexThreadOptions,
  restoredCodexSessionOptions,
  validateCodexStructuredSessionInitialOptions
} from './codex-structured-session-options'
import type { CodexAcquisitionRegistry } from './codex-structured-session-state'
import {
  codexSessionLifecycle,
  mintCodexAcquisitionGeneration,
  type CodexAcquisitionAttempt,
  type CodexSession,
  type CodexStructuredSessionAdapterDeps
} from './codex-structured-session-state'
import type { CodexStructuredTurnCancellation } from './codex-structured-turn-cancellation'

export type CodexSessionAcquireContext = {
  deps: CodexStructuredSessionAdapterDeps
  sessions: Map<string, CodexSession>
  acquisitions: CodexAcquisitionRegistry
  turnCancellation: CodexStructuredTurnCancellation
  deliver: (
    acquisition: CodexAcquisitionAttempt['window'],
    sessionId: string,
    event: () => void,
    retainedBytes?: number
  ) => void
  handleNotification: (sessionId: string, method: string, params: unknown) => void
  handleServerRequest: (sessionId: string, request: CodexAppServerServerRequest) => void
  handleUnhandledFrame: (sessionId: string, kind: string, params: unknown) => void
  clearNotificationRetries: (
    sessionId: string,
    connection: CodexAcquisitionAttempt['window']['connection']
  ) => void
  retryNotifications: (
    sessionId: string,
    connection: CodexAcquisitionAttempt['window']['connection']
  ) => void
  forceCloseUnexpected: (
    sessionId: string,
    fence: number,
    generation: string,
    reason: Error
  ) => Promise<boolean>
}

export async function acquireCodexStructuredSession(
  context: CodexSessionAcquireContext,
  input: StructuredAgentSessionAcquireInput
): Promise<AgentSessionAcquisition> {
  const { deps, acquisitions, sessions, turnCancellation } = context
  const sessionId = input.identity.sessionId
  const { previousAttempt, attempt } = acquisitions.start(sessionId)
  const acquisition = attempt.window
  let unbindReadingControl: (() => void) | undefined
  let primaryThreadId =
    input.identity.providerHandle.kind === 'codex' ? input.identity.providerHandle.threadId : null
  const translator = input.events
    ? createCodexJournalTranslator({
        sink: input.events,
        primaryThreadId: () => primaryThreadId,
        bindPromptItemId: (journalItemId, threadId, promptKey) =>
          acquisition.prompts.bindJournalItemId(journalItemId, threadId, promptKey)
      })
    : null
  const open = deps.openConnection ?? openCodexAppServerConnection
  try {
    await stopSupersededCodexAcquisition({
      sessionId,
      registry: acquisitions,
      replacement: attempt,
      previous: previousAttempt
    })
    acquisitions.assertCurrent(sessionId, attempt)
    if (!(await closeCodexPublishedSession(sessions, sessionId, deps.onEvent))) {
      throw new Error(`codex app-server for session ${sessionId} could not be stopped`)
    }
    acquisitions.assertCurrent(sessionId, attempt)
    const launch = await deps
      .resolveLaunch({ identity: input.identity })
      .catch((error: unknown) => {
        throw new AgentSessionPreSpawnError(error)
      })
    acquisitions.assertCurrent(sessionId, attempt)
    const connection = await open(
      {
        command: launch.command,
        args: launch.args,
        cwd: launch.cwd,
        env: buildCodexStructuredChildEnvironment(launch, input.spawnToken)
      },
      {
        onNotification: (method, params) =>
          context.deliver(
            acquisition,
            sessionId,
            () => context.handleNotification(sessionId, method, params),
            Buffer.byteLength(JSON.stringify(params ?? null), 'utf8')
          ),
        onServerRequest: (request) =>
          context.deliver(
            acquisition,
            sessionId,
            () => context.handleServerRequest(sessionId, request),
            Buffer.byteLength(JSON.stringify(request), 'utf8')
          ),
        onUnhandledFrame: (kind, payload) =>
          context.deliver(
            acquisition,
            sessionId,
            () => context.handleUnhandledFrame(sessionId, kind, payload),
            Buffer.byteLength(JSON.stringify({ kind, payload }), 'utf8')
          ),
        onExit: (error) => {
          acquisition.prompts.clear()
          handleCodexSessionExit({
            sessions,
            sessionId,
            connection: acquisition.connection,
            error,
            ...(deps.onEvent ? { onEvent: deps.onEvent } : {})
          })
          context.clearNotificationRetries(sessionId, acquisition.connection)
        }
      }
    )
    acquisition.connection = connection
    if (connection.pauseReading && connection.resumeReading) {
      unbindReadingControl = input.events?.bindReadingControl?.({
        pauseReading: connection.pauseReading,
        resumeReading: () => {
          connection.resumeReading?.()
          context.retryNotifications(sessionId, connection)
        }
      })
    }
    acquisitions.assertCurrent(sessionId, attempt)
    const opened = await openCodexThread(connection, launch, deps.requestTimeoutMs)
    acquisitions.assertCurrent(sessionId, attempt)
    primaryThreadId = opened.threadId
    const restoreAdmission = translator?.restoreThread(opened.threadId, opened.thread ?? {})
    if (restoreAdmission && !restoreAdmission.accepted) {
      throw new AgentSessionAcquisitionRefusal(
        'Codex thread history exceeds the bounded restore queue; history was not partially imported.'
      )
    }
    const process = await codexProcessIdentity(
      { ...input, pid: connection.pid },
      deps.readProcessStartTime
    )
    acquisitions.assertCurrent(sessionId, attempt)
    const acquisitionGeneration = mintCodexAcquisitionGeneration(deps)
    const acquired: AgentSessionAcquisition = {
      process,
      link: codexProviderHandleLink({
        threadId: opened.threadId,
        resumed: launch.resumeThreadId !== null,
        fence: input.fence,
        linkId: deps.mintLinkId?.(),
        observedAt: deps.now?.() ?? Date.now()
      }),
      acquisitionGeneration
    }
    if (connection.closed) {
      throw new Error(`codex app-server for session ${sessionId} exited while being acquired`)
    }
    const session: CodexSession = {
      connection,
      ...codexSessionLifecycle(input.fence, acquisitionGeneration),
      threadId: opened.threadId,
      historyPath: opened.historyPath,
      prompts: acquisition.prompts,
      options: restoredCodexSessionOptions(input.options),
      reportedOptions: reportedCodexThreadOptions(opened),
      turnIdWaiters: [],
      translator,
      ...(unbindReadingControl ? { unbindReadingControl } : {}),
      forceCloseUnexpected: (reason) =>
        context.forceCloseUnexpected(sessionId, input.fence, acquisitionGeneration, reason)
    }
    if (input.validateOptions && input.options) {
      await validateCodexStructuredSessionInitialOptions(
        session,
        input.options,
        deps.requestTimeoutMs
      )
    }
    if (connection.closed) {
      throw new Error(`codex app-server for session ${sessionId} exited while being acquired`)
    }
    acquisitions.publishIfCurrent(sessionId, attempt, () => {
      turnCancellation.register(session)
      sessions.set(sessionId, session)
    })
    for (const event of acquisition.drain()) {
      event()
    }
    return acquired
  } catch (error) {
    if (sessions.get(sessionId)?.connection !== acquisition.connection) {
      return closeFailedCodexAcquisition({
        sessionId,
        registry: acquisitions,
        attempt,
        cause: error,
        dispose: () => {
          unbindReadingControl?.()
          translator?.dispose()
        }
      })
    }
    acquisitions.deleteIfCurrent(sessionId, attempt)
    throw error
  } finally {
    attempt.finish()
  }
}
