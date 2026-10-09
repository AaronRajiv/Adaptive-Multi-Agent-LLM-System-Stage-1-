export type ArchitectureNodeId =
  | "user_task"
  | "api_gateway"
  | "planner"
  | "task_graph"
  | "orchestrator"
  | "agents"
  | "tools"
  | "rag"
  | "evaluation"
  | "synthesis"
  | "final_answer";

export type ArchitectureNodeStatus = "IDLE" | "ACTIVE" | "COMPLETED" | "FAILED" | "BLOCKED";

export interface ArchitectureNode {
  id: ArchitectureNodeId;
  label: string;
  subtitle: string;
  status: ArchitectureNodeStatus;
  isConditional?: boolean | undefined;
  activeCount?: number | undefined;
  completedCount?: number | undefined;
  totalCount?: number | undefined;
  detail?: string | undefined;
  x: number; // percentage X position in visualization layout (0 to 100)
  y: number; // percentage Y position in visualization layout (0 to 100)
}

export interface ArchitectureEdge {
  id: string;
  source: ArchitectureNodeId;
  target: ArchitectureNodeId;
  isConditional?: boolean | undefined;
}

export interface ArchitectureParticle {
  id: string;
  source: ArchitectureNodeId;
  target: ArchitectureNodeId;
  startTime: number;
  durationMs: number;
  label?: string | undefined;
  color?: string | undefined;
}

export interface DynamicTaskDetail {
  id: string;
  title: string;
  capability?: string | undefined;
  status: ArchitectureNodeStatus;
  dependencies: string[];
  durationMs?: number | undefined;
}
