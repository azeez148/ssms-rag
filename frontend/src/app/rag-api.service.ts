import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../environments/environment';

export interface HealthResponse {
  status: string;
  app_name: string;
  app_env: string;
}

export interface IngestRequest {
  config_path: string;
  chunk_size?: number;
  chunk_overlap?: number;
}

export interface IngestResponse {
  run_id: number;
  status: string;
  documents_processed: number;
  documents_created: number;
  documents_updated: number;
  documents_unchanged: number;
  chunks_created: number;
  error: string | null;
}

export type SearchMode = 'vector' | 'keyword' | 'hybrid';

export interface SearchRequest {
  knowledge_base: string;
  query: string;
  top_k?: number;
  filters?: Record<string, string> | null;
  mode?: SearchMode;
  rerank?: boolean;
}

export interface RetrievedChunk {
  content: string;
  document_id: string;
  chunk_index: number;
  document_title: string | null;
  source: string;
  knowledge_base_id: string;
  metadata: Record<string, unknown>;
  similarity: number;
  distance: number;
}

export interface SearchResponse {
  results: RetrievedChunk[];
}

export interface ConversationTurn {
  question: string;
  answer: string;
}

export interface ChatRequest {
  knowledge_base: string;
  question: string;
  top_k?: number;
  filters?: Record<string, string> | null;
  rerank?: boolean;
  history?: ConversationTurn[];
}

export interface ChatSource {
  index: number;
  document_id: string;
  document_title: string | null;
  source: string;
  knowledge_base_id: string;
  similarity: number;
  metadata: Record<string, unknown>;
  cited: boolean;
}

/** A proposed write (e.g. recording a sale) that runs only when the user confirms it. */
export interface ChatAction {
  id: string;
  title: string;
  lines: string[];
  confirm_label: string;
  expires_at: number;
}

export interface ActionOutcome {
  status: 'done' | 'cancelled';
  message: string;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  cited_source_count: number;
  retrieval_query: string;
  blocked: boolean;
  blocked_category: string | null;
  action?: ChatAction | null;
  /** Answered by the shop/action handler, not the knowledge base. */
  from_action?: boolean;
}

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string | null;
  suggested_questions: string[];
  config_path: string;
  documents_indexed: number;
}

export interface KnowledgeBasesResponse {
  answer_engine: string;
  answer_model: string;
  knowledge_bases: KnowledgeBase[];
}

@Injectable({ providedIn: 'root' })
export class RagApiService {
  private readonly baseUrl = environment.apiBaseUrl;

  constructor(private readonly http: HttpClient) {}

  listKnowledgeBases(): Observable<KnowledgeBasesResponse> {
    return this.http.get<KnowledgeBasesResponse>(`${this.baseUrl}/api/knowledge-bases`);
  }

  getHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`);
  }

  ingest(request: IngestRequest): Observable<IngestResponse> {
    return this.http.post<IngestResponse>(`${this.baseUrl}/admin/ingest`, request);
  }

  search(request: SearchRequest): Observable<SearchResponse> {
    return this.http.post<SearchResponse>(`${this.baseUrl}/api/search`, request);
  }

  chat(request: ChatRequest): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.baseUrl}/api/chat`, request);
  }

  chatAgentic(request: ChatRequest): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.baseUrl}/api/chat/agentic`, request);
  }

  confirmAction(id: string): Observable<ActionOutcome> {
    return this.http.post<ActionOutcome>(`${this.baseUrl}/api/actions/${encodeURIComponent(id)}/confirm`, {});
  }

  cancelAction(id: string): Observable<ActionOutcome> {
    return this.http.post<ActionOutcome>(`${this.baseUrl}/api/actions/${encodeURIComponent(id)}/cancel`, {});
  }
}
