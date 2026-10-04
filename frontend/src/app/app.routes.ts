import { Routes } from '@angular/router';
import { ChatPageComponent } from './chat/chat-page.component';
import { ConsolePageComponent } from './console/console-page.component';

export const routes: Routes = [
  { path: '', component: ChatPageComponent, title: 'Knowledge Assistant' },
  { path: 'console', component: ConsolePageComponent, title: 'RAG Test Console' },
  { path: '**', redirectTo: '' }
];
