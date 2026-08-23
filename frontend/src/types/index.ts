export type SubTaskType = 'research' | 'analysis';

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
  status: 'PASS' | 'FAIL';
  feedback: string;
}

export interface StageTrace {
  stage_name: string;
  agent_name: string;
  duration_ms: number;
  status: 'SUCCESS' | 'FAILED';
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
