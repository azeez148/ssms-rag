import { ChatAction, ChatSource } from '../rag-api.service';

export type ActionState = 'pending' | 'working' | 'done' | 'cancelled' | 'failed';

/** A write the assistant proposed (e.g. a sale), and what became of it. */
export interface MessageAction extends ChatAction {
  state: ActionState;
  /** What the server said after confirm/cancel, or why it failed. */
  result?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  createdAt: number;
  sources?: ChatSource[];
  /** The guardrails refused the question (prompt injection / sensitive request). */
  blocked?: boolean;
  /** The request failed; `text` is a friendly explanation and the message can be retried. */
  error?: boolean;
  /** A proposed write awaiting confirmation. */
  action?: MessageAction;
  /** Answered by the shop/action handler (stock, sales…), not from the knowledge base. */
  fromAction?: boolean;
}

export interface ChatSettings {
  /** Re-rank retrieved passages with the cross-encoder (better first result, a little slower). */
  rerank: boolean;
  /** Use the agentic pipeline, which retries with broader wording when nothing matches well. */
  agentic: boolean;
}

export type SyncStatus = 'syncing' | 'ready' | 'stale' | 'error';

export interface SyncState {
  status: SyncStatus;
  startedAt: number;
  finishedAt?: number;
  message?: string;
  created?: number;
  updated?: number;
  unchanged?: number;
}
