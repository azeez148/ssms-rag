import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatSource, ConversationTurn, RagApiService } from './rag-api.service';

interface DisplayTurn extends ConversationTurn {
  retrievalQuery: string;
  sources: ChatSource[];
  citedSourceCount: number;
  blocked: boolean;
  blockedCategory: string | null;
}

@Component({
  selector: 'app-chat-panel',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './chat-panel.component.html'
})
export class ChatPanelComponent {
  knowledgeBase = 'sports-store';
  question = 'Do you have Argentina jerseys?';
  topK = 5;
  filterKey = 'category';
  filterValue = '';
  rerank = false;
  agentic = false;

  history: DisplayTurn[] = [];
  error: string | null = null;
  loading = false;

  constructor(private readonly api: RagApiService) {}

  ask(): void {
    if (!this.question.trim()) {
      return;
    }

    this.loading = true;
    this.error = null;

    const filters =
      this.filterKey.trim() && this.filterValue.trim() ? { [this.filterKey.trim()]: this.filterValue.trim() } : null;

    // Send only real, non-blocked turns as history — a blocked/refused turn's
    // adversarial question isn't legitimate conversation context, and feeding
    // it into the next turn's query rewrite would contaminate retrieval.
    // The backend also caps how many turns it actually uses (MAX_HISTORY_TURNS).
    const historyForRequest: ConversationTurn[] = this.history
      .filter((turn) => !turn.blocked)
      .map((turn) => ({ question: turn.question, answer: turn.answer }));
    const question = this.question;

    const chatRequest = {
      knowledge_base: this.knowledgeBase,
      question,
      top_k: this.topK,
      filters,
      rerank: this.rerank,
      history: historyForRequest
    };

    (this.agentic ? this.api.chatAgentic(chatRequest) : this.api.chat(chatRequest))
      .subscribe({
        next: (response) => {
          this.history.push({
            question,
            answer: response.answer,
            retrievalQuery: response.retrieval_query,
            sources: response.sources,
            citedSourceCount: response.cited_source_count,
            blocked: response.blocked,
            blockedCategory: response.blocked_category
          });
          this.question = '';
          this.loading = false;
        },
        error: (err) => {
          this.error = err.error?.detail ?? err.message ?? 'Request failed';
          this.loading = false;
        }
      });
  }

  clearConversation(): void {
    this.history = [];
    this.error = null;
  }

  formatSimilarity(value: number): string {
    // Rerank replaces the vector similarity with an unbounded cross-encoder
    // score — showing that as a percentage would be misleading.
    if (this.rerank) {
      return value.toFixed(4);
    }
    return (value * 100).toFixed(1) + '%';
  }
}
