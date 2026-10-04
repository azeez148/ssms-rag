import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { IngestResponse, RagApiService } from './rag-api.service';

@Component({
  selector: 'app-ingest-panel',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './ingest-panel.component.html'
})
export class IngestPanelComponent {
  configPath = 'config/sports_store_source.yaml';
  chunkSize = 500;
  chunkOverlap = 50;

  result: IngestResponse | null = null;
  error: string | null = null;
  loading = false;

  constructor(private readonly api: RagApiService) {}

  runIngest(): void {
    this.loading = true;
    this.error = null;
    this.result = null;

    this.api
      .ingest({
        config_path: this.configPath,
        chunk_size: this.chunkSize,
        chunk_overlap: this.chunkOverlap
      })
      .subscribe({
        next: (response) => {
          this.result = response;
          this.loading = false;
        },
        error: (err) => {
          this.error = err.error?.detail ?? err.message ?? 'Request failed';
          this.loading = false;
        }
      });
  }
}
