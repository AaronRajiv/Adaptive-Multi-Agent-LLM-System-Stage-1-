import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import {
  ArrowDownToLine,
  BrainCircuit,
  Check,
  Circle,
  Database,
  FileText,
  GitBranch,
  Layers3,
  Network,
  ShieldCheck,
  Sparkles,
  Wrench,
} from "lucide-react";
import type { ExecutionTask } from "@/types/execution";
export type TaskFlowNode = Node<{ task: ExecutionTask; conceptual: boolean }, "task">;
const icons = {
  input: FileText,
  gateway: ArrowDownToLine,
  planning: BrainCircuit,
  graph: GitBranch,
  orchestration: Network,
  research: Layers3,
  analysis: BrainCircuit,
  tools: Wrench,
  rag: Database,
  evaluation: ShieldCheck,
  synthesis: Sparkles,
};
export function TaskNode({ data, selected }: NodeProps<TaskFlowNode>) {
  const { task, conceptual } = data;
  const Icon = icons[task.type as keyof typeof icons] ?? Layers3;
  const tone =
    task.type === "evaluation"
      ? "evaluation"
      : task.type === "synthesis"
        ? "synthesis"
        : task.type === "rag"
          ? "retrieval"
          : "default";
  return (
    <div
      className={`task-node tone-${tone} ${conceptual ? "is-conceptual" : `status-${task.status.toLowerCase()}`} ${selected ? "is-selected" : ""}`}
    >
      <Handle type="target" position={Position.Left} />
      <div className="node-heading">
        <span className="node-icon">
          <Icon size={15} />
        </span>
        <span>{task.type === "rag" ? "RAG RETRIEVAL" : task.type.toUpperCase()}</span>
        <span className="node-order">{conceptual ? "SYS" : task.id.slice(0, 8)}</span>
      </div>
      <div className="node-body">
        <div className="node-title">{task.title}</div>
        <p>{task.description || task.capability || "Task details not provided"}</p>
        {task.type === "rag" && !conceptual && (
          <div className="node-detail">
            {task.query ? `Query: ${task.query}` : "RAG telemetry unavailable for this run."}
            {task.chunks?.length ? <span>{task.chunks.length} retrieved chunks</span> : null}
          </div>
        )}
        {task.type === "evaluation" && !conceptual && (
          <div className="evaluation-detail">
            <span className="score-ring">{task.score ?? "—"}</span>
            <div>
              <strong>{task.evaluationStatus || "Unavailable"}</strong>
              <span>
                {task.score !== undefined ? "Evaluation score" : "Awaiting evaluation data"}
              </span>
            </div>
          </div>
        )}
      </div>
      {!conceptual && (
        <div className="node-footer">
          <span className="node-state">
            {task.status === "COMPLETED" ? (
              <Check size={11} />
            ) : (
              <Circle size={5} fill="currentColor" />
            )}
            {task.status}
          </span>
          <span>
            {task.durationMs !== undefined ? `${(task.durationMs / 1000).toFixed(2)}s` : "—"}
          </span>
        </div>
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
