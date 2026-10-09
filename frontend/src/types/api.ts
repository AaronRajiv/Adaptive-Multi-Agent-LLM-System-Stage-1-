export type SubTaskType = "research" | "analysis";

export interface DocumentPayload {
  filename: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface DocumentMetadata {
  document_id: string;
  filename: string;
  label?: string;
  sha256: string;
  file_size_bytes: number;
  status: "queued" | "extracting" | "embedding" | "ready" | "failed";
  chunk_count: number;
  embedded_chunk_count: number;
  error_message?: string;
  created_at: string;
  updated_at: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  label?: string;
  sha256: string;
  status: "queued" | "extracting" | "embedding" | "ready" | "failed";
  message: string;
}

export interface ChatMessagePayload {
  role: "user" | "assistant";
  content: string;
}


export interface SubTask {
  id: string;
  description: string;
  type: SubTaskType;
}

export interface ResearchResult {
  subtask_id: string;
  subtask_description: string;
  findings: string;
}

export interface EvaluationResult {
  score: number;
  status: "PASS" | "FAIL";
  feedback: string;
}

export interface StageTrace {
  stage_name: string;
  agent_name: string;
  duration_ms: number;
  status: "SUCCESS" | "FAILED";
  summary?: string;
}

export interface RunResponse {
  run_id: string;
  user_task: string;
  subtasks: SubTask[];
  research_results: ResearchResult[];
  analysis: string;
  evaluation: EvaluationResult;
  final_answer: string;
  execution_trace: StageTrace[];
  created_at: string;
}

export interface SystemStatusResponse {
  configured: boolean;
  provider: string;
  model: string;
  evaluator_threshold: number;
  message: string;
}

export interface ApiErrorDetail {
  error_type?: string;
  message?: string;
  hint?: string;
}
