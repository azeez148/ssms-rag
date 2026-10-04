import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ChatPanelComponent } from '../chat-panel.component';
import { HealthPanelComponent } from '../health-panel.component';
import { IngestPanelComponent } from '../ingest-panel.component';
import { SearchPanelComponent } from '../search-panel.component';

@Component({
  selector: 'app-console-page',
  imports: [RouterLink, HealthPanelComponent, IngestPanelComponent, SearchPanelComponent, ChatPanelComponent],
  templateUrl: './console-page.component.html',
  styleUrl: './console-page.component.css'
})
export class ConsolePageComponent {
  title = 'RAG Platform — Test Console';
}
