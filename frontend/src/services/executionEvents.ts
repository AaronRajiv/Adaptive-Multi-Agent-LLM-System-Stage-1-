import type { ExecutionEvent, ExecutionModel } from "@/types/execution";
import { API_BASE_URL } from "./api";

export interface BackendExecutionEvent {
  event_id: string;
  run_id: string;
  type: string;
  timestamp: string;
  stage?: string;
  task_id?: string;
  agent_type?: string;
  payload?: Record<string, unknown>;
  message?: string;
}

export interface ExecutionEventSource {
  subscribe(listener: (event: ExecutionEvent) => void, onError?: (err: Error) => void): () => void;
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

/** Server-Sent Events source connecting to real backend execution events. */
export class ServerExecutionEventSource implements ExecutionEventSource {
  constructor(private runId: string) {}

  subscribe(
    listener: (event: ExecutionEvent) => void,
    onError?: (err: Error) => void
  ): () => void {
    const url = `${API_BASE_URL}/api/runs/${this.runId}/events`;
    const eventSource = new EventSource(url);

    eventSource.onmessage = (e) => {
      try {
        const raw: BackendExecutionEvent = JSON.parse(e.data);
        const event: ExecutionEvent = {
          id: raw.event_id || `${Date.now()}-${Math.random()}`,
          label: raw.type,
          message: raw.message || raw.type,
          source: "SSE",
          time: raw.timestamp
            ? new Date(raw.timestamp).toLocaleTimeString("en-GB")
            : new Date().toLocaleTimeString("en-GB"),
          rawType: raw.type,
          taskId: raw.task_id,
          stage: raw.stage,
          payload: raw.payload,
        };
        listener(event);

        if (raw.type === "RUN_COMPLETED" || raw.type === "RUN_FAILED") {
          eventSource.close();
        }
      } catch {
        // Ignore JSON parse error
      }
    };

    eventSource.onerror = (err) => {
      if (onError) {
        onError(new Error("SSE connection error"));
      }
    };

    return () => {
      eventSource.close();
    };
  }
}

