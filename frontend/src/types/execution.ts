export type TaskStatus = "PENDING" | "READY" | "RUNNING" | "COMPLETED" | "FAILED" | "BLOCKED";
export interface ProvenanceChunk {
  documentId?: string | undefined;
  chunkIndex?: number | undefined;
  source?: string | undefined;
  content?: string | undefined;
  similarity?: number | undefined;
}
export interface ExecutionTask {
  id: string;
  title: string;
  description?: string | undefined;
  type: string;
  status: TaskStatus;
  dependencies: string[];
  capability?: string | undefined;
  durationMs?: number | undefined;
  agent?: string | undefined;
  input?: string | undefined;
  output?: string | undefined;
  metadata?: Record<string, unknown> | undefined;
  score?: number | undefined;
  evaluationStatus?: string | undefined;
  chunks?: ProvenanceChunk[] | undefined;
  query?: string | undefined;
}
export interface ExecutionEvent {
  id: string;
  label: string;
  message: string;
  source: "CLIENT" | "POST-RUN" | "DEMO" | "SSE";
  time?: string | undefined;
  durationMs?: number | undefined;
  status?: TaskStatus | undefined;
  rawType?: string | undefined;
  taskId?: string | undefined;
  stage?: string | undefined;
  payload?: Record<string, unknown> | undefined;
}
export interface ExecutionModel {
  runId: string;
  task: string;
  tasks: ExecutionTask[];
  events: ExecutionEvent[];
  finalAnswer: string;
  createdAt?: string | undefined;
  dependenciesAvailable: boolean;
  chunks: ProvenanceChunk[];
  executionStrategy?: string | undefined;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  files?: File[] | undefined;
  events?: ExecutionEvent[] | undefined;
  error?: string | undefined;
  running?: boolean | undefined;
  executionStrategy?: string | undefined;
}


