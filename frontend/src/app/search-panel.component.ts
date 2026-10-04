import { KeyValuePipe } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RagApiService, RetrievedChunk, SearchMode } from './rag-api.service';

@Component({
  selector: 'app-search-panel',
  standalone: true,
  imports: [FormsModule, KeyValuePipe],
  templateUrl: './search-panel.component.html'
})
export class SearchPanelComponent {
  knowledgeBase = 'sports-store';
  query = 'Do you have Argentina jerseys?';
  topK = 5;
  filterKey = 'category';
  filterValue = '';
  mode: SearchMode = 'vector';
  modes: SearchMode[] = ['vector', 'keyword', 'hybrid'];
  rerank = false;

  results: RetrievedChunk[] = [];
  error: string | null = null;
  loading = false;
  searched = false;

  constructor(private readonly api: RagApiService) {}

  runSearch(): void {
    if (!this.query.trim()) {
      return;
    }

    this.loading = true;
    this.error = null;
    this.searched = true;

    const filters =
      this.filterKey.trim() && this.filterValue.trim() ? { [this.filterKey.trim()]: this.filterValue.trim() } : null;

    this.api
      .search({
        knowledge_base: this.knowledgeBase,
        query: this.query,
        top_k: this.topK,
        filters,
        mode: this.mode,
        rerank: this.rerank
      })
      .subscribe({
        next: (response) => {
          this.results = response.results;
          this.loading = false;
        },
        error: (err) => {
          this.error = err.error?.detail ?? err.message ?? 'Request failed';
          this.results = [];
          this.loading = false;
        }
      });
  }

  formatSimilarity(value: number): string {
    // Not always a [0,1] cosine similarity: keyword mode uses ts_rank
    // (unbounded) and rerank uses a cross-encoder logit (unbounded, can be
    // negative) — showing it as a percentage would be misleading for those.
    if (this.mode === 'vector' && !this.rerank) {
      return (value * 100).toFixed(1) + '%';
    }
    return value.toFixed(4);
  }
}
