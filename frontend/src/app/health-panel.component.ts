import { Component } from '@angular/core';
import { HealthResponse, RagApiService } from './rag-api.service';

@Component({
  selector: 'app-health-panel',
  standalone: true,
  templateUrl: './health-panel.component.html'
})
export class HealthPanelComponent {
  result: HealthResponse | null = null;
  error: string | null = null;
  loading = false;

  constructor(private readonly api: RagApiService) {}

  check(): void {
    this.loading = true;
    this.error = null;
    this.result = null;

    this.api.getHealth().subscribe({
      next: (response) => {
        this.result = response;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.message ?? 'Request failed';
        this.loading = false;
      }
    });
  }
}
