import { HttpErrorResponse } from '@angular/common/http';
import { Injectable, computed, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { ConversationTurn, KnowledgeBase, RagApiService } from '../rag-api.service';
import { ChatMessage, ChatSettings, MessageAction, SyncState } from './chat.models';

const STORAGE_KEY = 'rag.chat.v1';
const MAX_STORED_MESSAGES = 40;
const MAX_HISTORY_TURNS = 3;
export const MAX_QUESTION_LENGTH = 1000;

interface Persisted {
  selectedId: string | null;
  settings: ChatSettings;
  conversations: Record<string, ChatMessage[]>;
}

let idCounter = 0;
function newId(): string {
  idCounter += 1;
  return `${Date.now().toString(36)}-${idCounter}`;
}

function describeError(error: unknown, whatWasHappening: string): string {
  if (error instanceof HttpErrorResponse) {
    if (error.status === 0) {
      return `I couldn't reach the server while ${whatWasHappening}. Check that the backend is running.`;
    }
    if (error.status === 503) {
      return `A database is unavailable right now, so I couldn't finish ${whatWasHappening}. Please try again in a moment.`;
    }
    if (error.status === 422 || error.status === 400) {
      return `The server rejected that request while ${whatWasHappening}.`;
    }
  }
  return `Something went wrong while ${whatWasHappening}.`;
}

/** A confirm that was in flight when the page closed may or may not have run. */
function interruptActions(conversations: Record<string, ChatMessage[]>): Record<string, ChatMessage[]> {
  const result: Record<string, ChatMessage[]> = {};
  for (const [id, messages] of Object.entries(conversations)) {
    result[id] = messages.map((message) =>
      message.action?.state === 'working'
        ? { ...message, action: { ...message.action, state: 'failed', result: 'Interrupted. Check the store before trying again.' } }
        : message
    );
  }
  return result;
}

/**
 * All chat state. Opening the chat loads the available knowledge bases and
 * immediately runs ingestion for the selected one (idempotent: unchanged
 * source rows are skipped server-side), so answers reflect the latest source
 * data. The composer unlocks as soon as there is something to answer from.
 */
@Injectable({ providedIn: 'root' })
export class ChatStore {
  readonly knowledgeBases = signal<KnowledgeBase[]>([]);
  readonly engine = signal<{ provider: string; model: string } | null>(null);
  readonly selectedId = signal<string | null>(null);
  readonly loadState = signal<'loading' | 'ready' | 'error'>('loading');
  readonly loadError = signal<string | null>(null);
  readonly syncStates = signal<Record<string, SyncState>>({});
  readonly conversations = signal<Record<string, ChatMessage[]>>({});
  readonly sending = signal(false);
  readonly settings = signal<ChatSettings>({ rerank: true, agentic: false });

  readonly selected = computed(() => this.knowledgeBases().find((kb) => kb.id === this.selectedId()) ?? null);
  readonly messages = computed(() => this.conversations()[this.selectedId() ?? ''] ?? []);
  readonly syncState = computed<SyncState | null>(() => this.syncStates()[this.selectedId() ?? ''] ?? null);
  /** True once there is indexed data to answer from (fresh, or stale-but-present). */
  readonly canChat = computed(() => {
    const status = this.syncState()?.status;
    return (status === 'ready' || status === 'stale') && !this.sending();
  });

  private started = false;

  constructor(private readonly api: RagApiService) {
    this.restore();
  }

  /** Called when the chat opens. Safe to call more than once. */
  async start(): Promise<void> {
    if (this.started) {
      return;
    }
    this.started = true;
    await this.loadKnowledgeBases();
    const id = this.selectedId();
    if (id) {
      await this.ensureSynced(id);
    }
  }

  async reload(): Promise<void> {
    this.started = false;
    this.syncStates.set({});
    await this.start();
  }

  async selectKnowledgeBase(id: string): Promise<void> {
    if (id === this.selectedId() || this.sending()) {
      return;
    }
    this.selectedId.set(id);
    this.persist();
    await this.ensureSynced(id);
  }

  /** Re-run ingestion for a knowledge base. `force` re-checks even if already synced this session. */
  async ensureSynced(id: string, force = false): Promise<void> {
    const kb = this.knowledgeBases().find((candidate) => candidate.id === id);
    if (!kb) {
      return;
    }
    const current = this.syncStates()[id]?.status;
    if (!force && (current === 'syncing' || current === 'ready' || current === 'stale')) {
      return;
    }

    const startedAt = Date.now();
    this.setSync(id, { status: 'syncing', startedAt });

    try {
      const result = await firstValueFrom(this.api.ingest({ config_path: kb.config_path }));
      if (result.status !== 'completed') {
        throw new Error(result.error ?? 'ingestion did not complete');
      }
      this.setSync(id, {
        status: 'ready',
        startedAt,
        finishedAt: Date.now(),
        created: result.documents_created,
        updated: result.documents_updated,
        unchanged: result.documents_unchanged
      });
      await this.refreshCounts();
    } catch (error) {
      const message = describeError(error, 'refreshing the knowledge base');
      // If an earlier ingestion left data behind, the chat still works — just say it may be out of date.
      const hasData = kb.documents_indexed > 0;
      this.setSync(id, {
        status: hasData ? 'stale' : 'error',
        startedAt,
        finishedAt: Date.now(),
        message
      });
    }
  }

  async send(rawText: string): Promise<void> {
    const kb = this.selected();
    const text = rawText.trim().slice(0, MAX_QUESTION_LENGTH);
    if (!kb || !text || !this.canChat()) {
      return;
    }

    const history = this.buildHistory(this.messages());
    this.append(kb.id, { id: newId(), role: 'user', text, createdAt: Date.now() });
    this.sending.set(true);

    try {
      const { rerank, agentic } = this.settings();
      const request = { knowledge_base: kb.id, question: text, rerank, history };
      const response = await firstValueFrom(agentic ? this.api.chatAgentic(request) : this.api.chat(request));
      this.append(kb.id, {
        id: newId(),
        role: 'assistant',
        text: response.answer,
        createdAt: Date.now(),
        sources: response.sources,
        blocked: response.blocked,
        action: response.action ? { ...response.action, state: 'pending' } : undefined,
        fromAction: response.from_action || undefined
      });
    } catch (error) {
      this.append(kb.id, {
        id: newId(),
        role: 'assistant',
        text: describeError(error, 'answering that'),
        createdAt: Date.now(),
        error: true
      });
    } finally {
      this.sending.set(false);
    }
  }

  /** Removes a failed answer and its question, then asks again. */
  async retry(errorMessageId: string): Promise<void> {
    const kb = this.selected();
    if (!kb || this.sending()) {
      return;
    }
    const messages = this.messages();
    const position = messages.findIndex((message) => message.id === errorMessageId);
    const question = messages[position - 1];
    if (position < 1 || question?.role !== 'user') {
      return;
    }
    this.setMessages(kb.id, messages.slice(0, position - 1));
    await this.send(question.text);
  }

  /** Runs a proposed action (e.g. records the sale). The server makes each proposal single-use. */
  async confirmAction(messageId: string): Promise<void> {
    await this.settleAction(messageId, 'confirm');
  }

  async cancelAction(messageId: string): Promise<void> {
    await this.settleAction(messageId, 'cancel');
  }

  newChat(): void {
    const id = this.selectedId();
    if (!id || this.sending()) {
      return;
    }
    this.setMessages(id, []);
  }

  updateSettings(patch: Partial<ChatSettings>): void {
    this.settings.update((current) => ({ ...current, ...patch }));
    this.persist();
  }

  // ---- internals ---------------------------------------------------------

  private async settleAction(messageId: string, how: 'confirm' | 'cancel'): Promise<void> {
    const kbId = this.selectedId();
    const action = this.messages().find((message) => message.id === messageId)?.action;
    if (!kbId || action?.state !== 'pending') {
      return;
    }
    this.updateAction(kbId, messageId, { state: 'working' });
    try {
      const call = how === 'confirm' ? this.api.confirmAction(action.id) : this.api.cancelAction(action.id);
      const outcome = await firstValueFrom(call);
      this.updateAction(kbId, messageId, { state: outcome.status === 'done' ? 'done' : 'cancelled', result: outcome.message });
    } catch (error) {
      const detail = error instanceof HttpErrorResponse && typeof error.error?.detail === 'string' ? error.error.detail : null;
      this.updateAction(kbId, messageId, { state: 'failed', result: detail ?? describeError(error, 'doing that') });
    }
  }

  private updateAction(kbId: string, messageId: string, patch: Partial<MessageAction>): void {
    const messages = this.conversations()[kbId] ?? [];
    this.setMessages(
      kbId,
      messages.map((message) =>
        message.id === messageId && message.action ? { ...message, action: { ...message.action, ...patch } } : message
      )
    );
  }

  private async loadKnowledgeBases(): Promise<void> {
    this.loadState.set('loading');
    this.loadError.set(null);
    try {
      const response = await firstValueFrom(this.api.listKnowledgeBases());
      this.knowledgeBases.set(response.knowledge_bases);
      this.engine.set({ provider: response.answer_engine, model: response.answer_model });

      const known = response.knowledge_bases.some((kb) => kb.id === this.selectedId());
      if (!known) {
        this.selectedId.set(response.knowledge_bases[0]?.id ?? null);
      }
      this.loadState.set('ready');
    } catch (error) {
      this.loadState.set('error');
      this.loadError.set(describeError(error, 'loading the knowledge bases'));
    }
  }

  private async refreshCounts(): Promise<void> {
    try {
      const response = await firstValueFrom(this.api.listKnowledgeBases());
      this.knowledgeBases.set(response.knowledge_bases);
    } catch {
      // The counts are cosmetic; a failed refresh must not turn a good sync into an error.
    }
  }

  /**
   * Recent question/answer pairs for query rewriting. Failed and refused turns
   * are left out so an adversarial question can't leak into later retrieval,
   * and shop actions (stock, sales) because they aren't knowledge-base answers:
   * a small model rewrites later questions around their products.
   */
  private buildHistory(messages: ChatMessage[]): ConversationTurn[] {
    const turns: ConversationTurn[] = [];
    for (let i = 0; i < messages.length - 1; i++) {
      const question = messages[i];
      const answer = messages[i + 1];
      const isKnowledgeAnswer = !answer.error && !answer.blocked && !answer.fromAction && !answer.action;
      if (question.role === 'user' && answer.role === 'assistant' && isKnowledgeAnswer) {
        turns.push({ question: question.text, answer: answer.text });
      }
    }
    return turns.slice(-MAX_HISTORY_TURNS);
  }

  private setSync(id: string, state: SyncState): void {
    this.syncStates.update((current) => ({ ...current, [id]: state }));
  }

  private append(id: string, message: ChatMessage): void {
    this.setMessages(id, [...(this.conversations()[id] ?? []), message]);
  }

  private setMessages(id: string, messages: ChatMessage[]): void {
    this.conversations.update((current) => ({ ...current, [id]: messages }));
    this.persist();
  }

  private persist(): void {
    try {
      const conversations: Record<string, ChatMessage[]> = {};
      for (const [id, messages] of Object.entries(this.conversations())) {
        conversations[id] = messages.slice(-MAX_STORED_MESSAGES);
      }
      const data: Persisted = { selectedId: this.selectedId(), settings: this.settings(), conversations };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch {
      // Private mode / quota / blocked storage: the chat works, it just won't be remembered.
    }
  }

  private restore(): void {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return;
      }
      const data = JSON.parse(raw) as Partial<Persisted>;
      if (typeof data.selectedId === 'string') {
        this.selectedId.set(data.selectedId);
      }
      if (data.settings) {
        this.settings.set({ rerank: !!data.settings.rerank, agentic: !!data.settings.agentic });
      }
      if (data.conversations && typeof data.conversations === 'object') {
        this.conversations.set(interruptActions(data.conversations));
      }
    } catch {
      // Corrupt or unavailable storage: start fresh.
    }
  }
}
