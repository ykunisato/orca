import type {
  AgentJournalMessageItem,
  AgentSessionJournalIdentity
} from '../../shared/agent-session-journal-types'
import type {
  AgentSessionAcquisition,
  AgentSessionDispatchOutcome,
  StructuredAgentSessionAcquireInput,
  StructuredAgentSessionAdapter,
  StructuredAgentSessionSetOptionInput
} from '../native-chat/agent-session-wire/structured-agent-session-adapter'
import type { CodexJournalTranslationAdmission } from './codex-structured-journal-translation'
import { supportsCodexStructuredLocation } from './codex-structured-location-support'
import {
  closeAllCodexSessions,
  closeCodexPublishedSession,
  closeCodexSession
} from './codex-structured-session-close'
import {
  CodexAcquisitionRegistry,
  type CodexAcquisitionAttempt,
  type CodexSession,
  type CodexStructuredSessionAdapterDeps,
  type CodexStructuredSessionEvent
} from './codex-structured-session-state'
import {
  deliverCodexNotification,
  deliverCodexServerRequest,
  deliverCodexUnhandledFrame
} from './codex-structured-provider-events'
import { CodexStructuredTurnCancellation } from './codex-structured-turn-cancellation'
import { createCodexStructuredNotificationRetry } from './codex-structured-notification-retry'
import {
  answerCodexSessionPrompt,
  bindCodexPromptItemId,
  cancelCodexTurn,
  codexSessionHistoryFilePath,
  dispatchCodexSession,
  readCodexSessionOptions,
  setCodexSessionOption
} from './codex-structured-session-adapter-operations'
import { acquireCodexStructuredSession } from './codex-structured-session-acquire'

export type {
  CodexStructuredLaunch,
  CodexStructuredSessionAdapterDeps,
  CodexStructuredSessionEvent
} from './codex-structured-session-state'

export class CodexStructuredSessionAdapter implements StructuredAgentSessionAdapter {
  private readonly sessions = new Map<string, CodexSession>()
  private readonly acquisitions = new CodexAcquisitionRegistry()
  private readonly turnCancellation: CodexStructuredTurnCancellation
  private readonly notificationRetries: ReturnType<typeof createCodexStructuredNotificationRetry>

  constructor(private readonly deps: CodexStructuredSessionAdapterDeps) {
    this.notificationRetries = createCodexStructuredNotificationRetry({
      sessionFor: (sessionId) => this.sessions.get(sessionId),
      translate: (sessionId, session, method, params) =>
        this.translateNotification(sessionId, session, method, params)
    })
    this.turnCancellation = new CodexStructuredTurnCancellation({
      captureTurnProcesses: deps.captureTurnProcesses,
      terminateTurnProcesses: deps.terminateTurnProcesses,
      requestTimeoutMs: deps.requestTimeoutMs,
      emit: (session, event) => {
        const admission = this.emit(session, event)
        if (!admission.accepted && event.type === 'notification') {
          this.notificationRetries.handle(event.sessionId, event.method, event.params)
        }
        return admission
      }
    })
  }

  supportsLocation = supportsCodexStructuredLocation

  acquire = (input: StructuredAgentSessionAcquireInput): Promise<AgentSessionAcquisition> =>
    acquireCodexStructuredSession(
      {
        deps: this.deps,
        sessions: this.sessions,
        acquisitions: this.acquisitions,
        turnCancellation: this.turnCancellation,
        deliver: (acquisition, sessionId, event, retainedBytes) =>
          this.deliver(acquisition, sessionId, event, retainedBytes),
        handleNotification: (sessionId, method, params) =>
          this.handleNotification(sessionId, method, params),
        handleServerRequest: (sessionId, request) => this.handleServerRequest(sessionId, request),
        handleUnhandledFrame: (sessionId, kind, params) =>
          this.handleUnhandledFrame(sessionId, kind, params),
        clearNotificationRetries: (sessionId, connection) =>
          this.notificationRetries.clear(sessionId, connection),
        retryNotifications: (sessionId, connection) => {
          if (connection) {
            this.notificationRetries.retry(sessionId, connection)
          }
        },
        forceCloseUnexpected: (sessionId, fence, generation, reason) =>
          this.forceCloseUnexpected(sessionId, fence, generation, reason)
      },
      input
    )

  /** Buffers pre-publication events and drops events from superseded children. */
  private deliver(
    acquisition: CodexAcquisitionAttempt['window'],
    sessionId: string,
    event: () => void,
    retainedBytes?: number
  ): void {
    if (acquisition.buffer(event, retainedBytes)) {
      return
    }
    if (this.sessions.get(sessionId)?.connection === acquisition.connection) {
      event()
    }
  }

  private translateNotification(
    sessionId: string,
    session: CodexSession,
    method: string,
    params: unknown
  ): CodexJournalTranslationAdmission {
    if (this.turnCancellation.handleNotification(sessionId, session, method, params)) {
      return { accepted: true }
    }
    return deliverCodexNotification(sessionId, session, method, params, (current, event) =>
      this.emit(current, event)
    )
  }

  private handleNotification(sessionId: string, method: string, params: unknown): void {
    const session = this.sessions.get(sessionId)
    if (!session) {
      return
    }
    this.notificationRetries.handle(sessionId, method, params)
  }

  /** Journal first so observers never see an event ahead of its durable row. */
  private emit(
    session: CodexSession,
    event: CodexStructuredSessionEvent
  ): CodexJournalTranslationAdmission {
    const admission = session.translator?.handle(event) ?? { accepted: true }
    if (!admission.accepted) {
      return admission
    }
    this.deps.onEvent?.(event)
    return admission
  }

  private handleServerRequest(
    sessionId: string,
    request: Parameters<typeof deliverCodexServerRequest>[2]
  ): void {
    deliverCodexServerRequest(sessionId, this.sessions.get(sessionId), request, (session, event) =>
      this.emit(session, event)
    )
  }

  private handleUnhandledFrame(sessionId: string, kind: string, params: unknown): void {
    deliverCodexUnhandledFrame(
      sessionId,
      this.sessions.get(sessionId),
      kind,
      params,
      (session, event) => this.emit(session, event)
    )
  }

  private forceCloseUnexpected(
    sessionId: string,
    fence: number,
    acquisitionGeneration: string,
    reason: Error
  ): Promise<boolean> {
    const session = this.sessions.get(sessionId)
    if (
      !session ||
      session.ended ||
      session.fence !== fence ||
      session.acquisitionGeneration !== acquisitionGeneration
    ) {
      return Promise.resolve(false)
    }
    return closeCodexPublishedSession(this.sessions, sessionId, this.deps.onEvent, {
      allowFailedSettlement: true,
      requestedClose: false,
      expectedFence: fence,
      expectedAcquisitionGeneration: acquisitionGeneration,
      unexpectedReason: reason
    })
  }

  bindPromptItemId = (sessionId: string, journalItemId: string, promptKey: string): void =>
    bindCodexPromptItemId(this.sessions, sessionId, journalItemId, promptKey)

  dispatch = async (input: {
    sessionId: string
    clientMessageId: string
    body: AgentJournalMessageItem
    fence: number
  }): Promise<AgentSessionDispatchOutcome> =>
    dispatchCodexSession(this.sessions, this.turnCancellation, input, this.deps.requestTimeoutMs)

  cancelTurn = async (input: {
    sessionId: string
    turnId: string
    fence: number
  }): Promise<{ cancelled: boolean }> =>
    cancelCodexTurn(this.sessions, this.turnCancellation, input)

  answerPrompt = async (input: {
    sessionId: string
    itemId: string
    kind: 'approval' | 'question'
    optionId: string
    fence: number
  }): Promise<void> => {
    answerCodexSessionPrompt(this.sessions, input)
    this.sessions.get(input.sessionId)?.translator?.resolvePrompt(input.itemId)
  }

  setOption = async (
    input: StructuredAgentSessionSetOptionInput
  ): Promise<Readonly<Record<string, string>>> =>
    setCodexSessionOption(this.sessions, input, this.deps.requestTimeoutMs)

  readOptions = (input: { sessionId: string; fence: number }) =>
    readCodexSessionOptions(this.sessions, input.sessionId, this.deps.requestTimeoutMs)

  historyFilePath = async (input: {
    identity: AgentSessionJournalIdentity
  }): Promise<string | null> => codexSessionHistoryFilePath(this.sessions, input.identity)

  closeSession = async (sessionId: string): Promise<boolean> => {
    const closed = await closeCodexSession(
      sessionId,
      this.sessions,
      this.acquisitions,
      this.deps.onEvent
    )
    if (closed) {
      this.notificationRetries.clear(sessionId, null)
    }
    return closed
  }
  forceCloseSession = async (sessionId: string): Promise<boolean> => {
    const closed = await closeCodexPublishedSession(this.sessions, sessionId, this.deps.onEvent, {
      allowFailedSettlement: true,
      requestedClose: false
    })
    if (closed) {
      this.notificationRetries.clear(sessionId, null)
    }
    return closed
  }
  disposeSession = (sessionId: string): Promise<boolean> => this.closeSession(sessionId)
  closeAll = (): Promise<void> =>
    closeAllCodexSessions(this.sessions, this.acquisitions, (sessionId) =>
      this.disposeSession(sessionId)
    )
  releaseAcquisition = (input: { sessionId: string }): Promise<boolean> =>
    this.closeSession(input.sessionId)
}
