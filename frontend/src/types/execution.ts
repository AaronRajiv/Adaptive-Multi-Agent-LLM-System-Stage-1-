export type TaskStatus = "PENDING" | "READY" | "RUNNING" | "COMPLETED" | "FAILED" | "BLOCKED";

export interface ProvenanceChunk {
  id?: string | undefined;
  documentId?: string | undefined;
  chunkIndex?: number | undefined;
  source?: string | undefined;
  content?: string | undefined;
  similarity?: number | undefined;
  pageStart?: number | undefined;
  pageEnd?: number | undefined;
  articlePath?: string | undefined;
  heading?: string | undefined;
  retrievalMethod?: string | undefined;
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

export interface CallMetricItem {
  stage: string;
  model: string;
  tokensIn: number;
  tokensOut: number;
  totalTokens: number;
  latencyMs: number;
  estimatedCost: number;
  success: boolean;
}

export interface RunMetricsData {
  runId: string;
  strategy: string;
  totalTokens: number;
  totalLlmCalls: number;
  totalEstimatedCost: number;
  wallTimeMs: number;
  calls?: CallMetricItem[];
}

export interface VerificationData {
  citationCount: number;
  validCitationCount: number;
  citationValidityPercent: number;
  validCitations: string[];
  invalidCitations: string[];
  passed: boolean;
}

export interface EvaluationData {
  score: number;
  status: "PASS" | "FAIL" | "SKIPPED" | "NOT_APPLICABLE";
  feedback: string;
  groundednessScore?: number;
  coverageScore?: number;
  distinctionScore?: number;
  honestyScore?: number;
  uncoveredDomains?: string[];
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
  replanningCount?: number | undefined;
  metrics?: RunMetricsData | undefined;
  verification?: VerificationData | undefined;
  evaluation?: EvaluationData | undefined;
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
  metrics?: RunMetricsData | undefined;
}
