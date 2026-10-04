import { NgTemplateOutlet } from '@angular/common';
import { Component, ElementRef, OnDestroy, OnInit, ViewChild, computed, effect, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ChatSource } from '../rag-api.service';
import { ChatStore, MAX_QUESTION_LENGTH } from './chat-store.service';
import { ChatMessage, SyncState } from './chat.models';
import { Block, formatAnswer } from './format-answer';

const ENGINE_NAMES: Record<string, string> = {
  anthropic: 'Claude',
  openai: 'OpenAI',
  gemini: 'Gemini',
  qwen: 'Qwen',
  ollama: 'Ollama (local)',
  openai_compatible: 'OpenAI-compatible API'
};

const HIDDEN_METADATA_KEYS = new Set(['entity_type', 'knowledge_base_id', 'record_id', 'source']);

@Component({
  selector: 'app-chat-page',
  imports: [NgTemplateOutlet, RouterLink],
  templateUrl: './chat-page.component.html',
  styleUrl: './chat-page.component.css'
})
export class ChatPageComponent implements OnInit, OnDestroy {
  readonly store = inject(ChatStore);
  readonly maxLength = MAX_QUESTION_LENGTH;

  readonly draft = signal('');
  readonly drawerOpen = signal(false);
  readonly now = signal(Date.now());
  readonly openSources = signal<ReadonlySet<string>>(new Set());
  readonly highlighted = signal<{ messageId: string; index: number } | null>(null);
  readonly copiedId = signal<string | null>(null);

  /** A pending proposal can't be confirmed after the server's expiry; checked against the ticking clock. */
  isExpired(message: ChatMessage): boolean {
    return !!message.action && message.action.state === 'pending' && this.now() / 1000 >= message.action.expires_at;
  }

  /** No indexed data to answer from yet (still syncing, or the sync failed with nothing saved). */
  readonly locked = computed(() => {
    const status = this.store.syncState()?.status;
    return status !== 'ready' && status !== 'stale';
  });

  readonly engineLabel = computed(() => {
    const engine = this.store.engine();
    if (!engine) {
      return null;
    }
    if (engine.provider === 'extractive') {
      return { title: 'Offline mode', detail: 'Answers are assembled from your data. Set LLM_PROVIDER for written answers.', offline: true };
    }
    return { title: `Answers by ${ENGINE_NAMES[engine.provider] ?? engine.provider}`, detail: engine.model, offline: false };
  });

  @ViewChild('scroller') private scroller?: ElementRef<HTMLElement>;
  @ViewChild('composer') private composer?: ElementRef<HTMLTextAreaElement>;

  private clock?: ReturnType<typeof setInterval>;
  private readonly formatted = new WeakMap<ChatMessage, Block[]>();

  constructor() {
    // Keep the newest message in view as the conversation and sync state change.
    effect(() => {
      this.store.messages();
      this.store.sending();
      this.store.syncState();
      setTimeout(() => this.scrollToBottom());
    });
    // Hand the keyboard back to the composer whenever the assistant is ready for input.
    effect(() => {
      if (this.store.canChat()) {
        setTimeout(() => this.composer?.nativeElement.focus());
      }
    });
  }

  ngOnInit(): void {
    this.clock = setInterval(() => this.now.set(Date.now()), 1000);
    void this.store.start();
  }

  ngOnDestroy(): void {
    clearInterval(this.clock);
  }

  // ---- composer ----------------------------------------------------------

  onInput(event: Event): void {
    const field = event.target as HTMLTextAreaElement;
    this.draft.set(field.value);
    field.style.height = 'auto';
    field.style.height = `${Math.min(field.scrollHeight, 180)}px`;
  }

  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      void this.submit();
    }
  }

  async submit(): Promise<void> {
    const text = this.draft().trim();
    if (!text || !this.store.canChat()) {
      return;
    }
    this.draft.set('');
    if (this.composer) {
      this.composer.nativeElement.value = '';
      this.composer.nativeElement.style.height = 'auto';
    }
    await this.store.send(text);
  }

  async ask(question: string): Promise<void> {
    this.draft.set(question);
    await this.submit();
  }

  // ---- navigation --------------------------------------------------------

  async pickKnowledgeBase(id: string): Promise<void> {
    this.drawerOpen.set(false);
    await this.store.selectKnowledgeBase(id);
  }

  newChat(): void {
    this.drawerOpen.set(false);
    this.store.newChat();
  }

  // ---- message rendering -------------------------------------------------

  blocksFor(message: ChatMessage): Block[] {
    let blocks = this.formatted.get(message);
    if (!blocks) {
      blocks = formatAnswer(message.text);
      this.formatted.set(message, blocks);
    }
    return blocks;
  }

  hasSource(message: ChatMessage, index: number): boolean {
    return !!message.sources?.some((source) => source.index === index);
  }

  isOpen(message: ChatMessage): boolean {
    return this.openSources().has(message.id);
  }

  toggleSources(message: ChatMessage): void {
    this.openSources.update((current) => {
      const next = new Set(current);
      if (!next.delete(message.id)) {
        next.add(message.id);
      }
      return next;
    });
  }

  jumpToSource(message: ChatMessage, index: number): void {
    this.openSources.update((current) => new Set(current).add(message.id));
    this.highlighted.set({ messageId: message.id, index });
    setTimeout(() => {
      document.getElementById(`source-${message.id}-${index}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
  }

  isHighlighted(message: ChatMessage, source: ChatSource): boolean {
    const target = this.highlighted();
    return target?.messageId === message.id && target.index === source.index;
  }

  tagsFor(source: ChatSource): string[] {
    return Object.entries(source.metadata)
      .filter(([key, value]) => !HIDDEN_METADATA_KEYS.has(key) && (typeof value === 'string' || typeof value === 'number'))
      .map(([, value]) => String(value))
      .slice(0, 2);
  }


  async copy(message: ChatMessage): Promise<void> {
    try {
      await navigator.clipboard.writeText(message.text);
      this.copiedId.set(message.id);
      setTimeout(() => this.copiedId.set(null), 1500);
    } catch {
      // Clipboard access can be denied; there's nothing useful to show.
    }
  }

  // ---- sync display ------------------------------------------------------

  elapsedSeconds(state: SyncState): number {
    return Math.max(0, Math.floor((this.now() - state.startedAt) / 1000));
  }

  syncSummary(state: SyncState): string {
    const created = state.created ?? 0;
    const updated = state.updated ?? 0;
    if (created === 0 && updated === 0) {
      return 'Up to date';
    }
    const parts = [];
    if (created) parts.push(`${created} new`);
    if (updated) parts.push(`${updated} updated`);
    return `Synced — ${parts.join(', ')}`;
  }

  private scrollToBottom(): void {
    const element = this.scroller?.nativeElement;
    if (element) {
      const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      element.scrollTo({ top: element.scrollHeight, behavior: reduce ? 'auto' : 'smooth' });
    }
  }
}
