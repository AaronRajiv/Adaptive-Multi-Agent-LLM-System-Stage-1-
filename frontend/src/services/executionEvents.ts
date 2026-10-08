import type { ExecutionEvent, ExecutionModel } from "@/types/execution";
export interface ExecutionEventSource {
  subscribe(listener: (event: ExecutionEvent) => void): () => void;
}
export class ResponseExecutionEventSource implements ExecutionEventSource {
  constructor(private run: ExecutionModel) {}
  subscribe(listener: (event: ExecutionEvent) => void) {
    this.run.events.forEach(listener);
    return () => {};
  }
}
/** Presentation-only playback. Never supplies results, scores, provenance or timings. */
export class DemoExecutionEventSource implements ExecutionEventSource {
  constructor(private events: ExecutionEvent[]) {}
  subscribe(listener: (event: ExecutionEvent) => void) {
    const timers = this.events.map((event, i) =>
      setTimeout(() => listener({ ...event, source: "DEMO" }), i * 600),
    );
    return () => timers.forEach(clearTimeout);
  }
}
