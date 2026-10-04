import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideRouter } from '@angular/router';
import { AppComponent } from './app.component';
import { ConsolePageComponent } from './console/console-page.component';

describe('AppComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [provideHttpClient(), provideRouter([])]
    }).compileComponents();
  });

  it('should create the app shell', () => {
    const fixture = TestBed.createComponent(AppComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });
});

describe('ConsolePageComponent', () => {
  it('should render the test console title and a link back to the chat', async () => {
    await TestBed.configureTestingModule({
      imports: [ConsolePageComponent],
      providers: [provideHttpClient(), provideRouter([])]
    }).compileComponents();

    const fixture = TestBed.createComponent(ConsolePageComponent);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;

    expect(compiled.querySelector('h1')?.textContent).toContain('RAG Platform');
    expect(compiled.querySelector('a.back')?.textContent).toContain('chat assistant');
  });
});
