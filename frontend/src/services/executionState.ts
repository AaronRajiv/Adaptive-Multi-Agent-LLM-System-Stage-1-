import type { ExecutionEvent, ExecutionTask, ProvenanceChunk, TaskStatus } from "@/types/execution";
import type { EvaluationResult } from "@/types/api";

export interface LiveExecutionState {
  running: boolean;
  tasks: ExecutionTask[];
  events: ExecutionEvent[];
  evaluation?: EvaluationResult | undefined;
  finalAnswer?: string | undefined;
  chunks: ProvenanceChunk[];
  error?: string | undefined;
}

export function createInitialLiveState(): LiveExecutionState {
  return {
    running: false,
    tasks: [],
    events: [],
    chunks: [],
  };
}

export function processExecutionEvent(
  state: LiveExecutionState,
  event: ExecutionEvent
): LiveExecutionState {
  const nextEvents = [...state.events, event];
  let nextRunning = state.running;
  let nextTasks = [...state.tasks];
  let nextChunks = [...state.chunks];
  let nextEvaluation = state.evaluation;
  let nextFinalAnswer = state.finalAnswer;
  let nextError = state.error;

  const rawType = event.rawType || event.label;
  const taskId = event.taskId;
  const payload: Record<string, unknown> = event.payload || {};

  switch (rawType) {
    case "RUN_STARTED":
      nextRunning = true;
      nextError = undefined;
      break;

    case "PLANNING_STARTED": {
      let pNode = nextTasks.find((t) => t.id === "planner" || t.type === "planning");
      if (!pNode) {
        pNode = {
          id: "planner",
          title: "Planner",
          description: "Decomposing user task",
          type: "planning",
          status: "RUNNING",
          dependencies: [],
        };
        nextTasks.push(pNode);
      } else {
        nextTasks = nextTasks.map((t) =>
          t.id === pNode!.id ? { ...t, status: "RUNNING" as TaskStatus } : t
        );
      }
      break;
    }

    case "PLANNING_COMPLETED": {
      nextTasks = nextTasks.map((t) =>
        t.id === "planner" || t.type === "planning" ? { ...t, status: "COMPLETED" as TaskStatus } : t
      );
      const rawTasks = Array.isArray(payload["tasks"])
        ? (payload["tasks"] as Record<string, unknown>[])
        : [];
      if (rawTasks.length > 0) {
        const dynamicNodes: ExecutionTask[] = rawTasks.map((raw) => ({
          id: String(raw["id"] || raw["task_id"]),
          title: String(raw["title"] || raw["description"] || raw["id"]).slice(0, 70),
          description: String(raw["description"] || ""),
          type: String(raw["type"] || raw["capability"] || "research"),
          capability: String(raw["capability"] || ""),
          status: "PENDING" as TaskStatus,
          dependencies: Array.isArray(raw["dependencies"])
            ? (raw["dependencies"] as unknown[]).map(String)
            : Array.isArray(raw["depends_on"])
              ? (raw["depends_on"] as unknown[]).map(String)
              : [],
        }));
        const systemNodes = nextTasks.filter((t) => t.type === "planning" || t.id === "planner");
        nextTasks = [...systemNodes, ...dynamicNodes];
      }
      break;
    }

    case "TASK_STARTED":
      if (taskId) {
        if (!nextTasks.some((t) => t.id === taskId)) {
          nextTasks.push({
            id: taskId,
            title: String(payload["title"] || taskId),
            description: String(payload["description"] || ""),
            type: String(payload["capability"] || "task"),
            status: "RUNNING" as TaskStatus,
            dependencies: [],
          });
        } else {
          nextTasks = nextTasks.map((t) =>
            t.id === taskId ? { ...t, status: "RUNNING" as TaskStatus } : t
          );
        }
      }
      break;

    case "TASK_COMPLETED":
      if (taskId) {
        const outVal = payload["output"];
        const durVal = payload["duration_ms"];
        nextTasks = nextTasks.map((t) => {
          if (t.id === taskId) {
            return {
              ...t,
              status: "COMPLETED" as TaskStatus,
              output: typeof outVal === "string" ? outVal : t.output,
              durationMs: typeof durVal === "number" ? durVal : t.durationMs,
            };
          }
          return t;
        });
      }
      break;

    case "TASK_FAILED":
      if (taskId) {
        nextTasks = nextTasks.map((t) =>
          t.id === taskId ? { ...t, status: "FAILED" as TaskStatus } : t
        );
      }
      break;

    case "TASK_BLOCKED":
      if (taskId) {
        nextTasks = nextTasks.map((t) =>
          t.id === taskId ? { ...t, status: "BLOCKED" as TaskStatus } : t
        );
      }
      break;

    case "RAG_RETRIEVAL_STARTED": {
      let ragNode = nextTasks.find((t) => t.id === "rag" || t.type === "rag");
      if (!ragNode) {
        ragNode = {
          id: "rag",
          title: "Knowledge Retrieval",
          description: "Conditional RAG context grounding",
          type: "rag",
          status: "RUNNING",
          dependencies: [],
        };
        nextTasks.push(ragNode);
      } else {
        nextTasks = nextTasks.map((t) =>
          t.id === ragNode!.id ? { ...t, status: "RUNNING" as TaskStatus } : t
        );
      }
      break;
    }

    case "RAG_RETRIEVAL_COMPLETED": {
      const chunks = Array.isArray(payload["chunks"])
        ? (payload["chunks"] as Record<string, unknown>[])
        : [];
      if (chunks.length > 0) {
        const formattedChunks: ProvenanceChunk[] = chunks.map((c) => ({
          documentId: typeof c["document_id"] === "string" ? c["document_id"] : undefined,
          chunkIndex: typeof c["chunk_index"] === "number" ? c["chunk_index"] : undefined,
          source: typeof c["source"] === "string" ? c["source"] : undefined,
          content: typeof c["content"] === "string" ? c["content"] : undefined,
          similarity:
            typeof c["similarity_score"] === "number"
              ? c["similarity_score"]
              : typeof c["score"] === "number"
                ? c["score"]
                : undefined,
        }));
        nextChunks = [...nextChunks, ...formattedChunks];
      }
      const qVal = payload["query"];
      nextTasks = nextTasks.map((t) =>
        t.id === "rag" || t.type === "rag"
          ? {
              ...t,
              status: "COMPLETED" as TaskStatus,
              chunks: nextChunks,
              query: typeof qVal === "string" ? qVal : t.query,
            }
          : t
      );
      break;
    }

    case "EVALUATION_STARTED": {
      let evNode = nextTasks.find((t) => t.id === "evaluator" || t.type === "evaluation");
      if (!evNode) {
        evNode = {
          id: "evaluator",
          title: "Evaluator",
          description: "Quality assessment",
          type: "evaluation",
          status: "RUNNING",
          dependencies: [],
        };
        nextTasks.push(evNode);
      } else {
        nextTasks = nextTasks.map((t) =>
          t.id === evNode!.id ? { ...t, status: "RUNNING" as TaskStatus } : t
        );
      }
      break;
    }

    case "EVALUATION_COMPLETED": {
      const scoreVal = payload["score"];
      const evStatusVal = payload["status"];
      const fbVal = payload["feedback"];

      const score = typeof scoreVal === "number" ? scoreVal : undefined;
      const evStatus = typeof evStatusVal === "string" ? evStatusVal : undefined;
      const feedback = typeof fbVal === "string" ? fbVal : undefined;

      if (score !== undefined && evStatus) {
        nextEvaluation = {
          score,
          status: evStatus as "PASS" | "FAIL",
          feedback: feedback || "",
        };
      }

      nextTasks = nextTasks.map((t) =>
        t.id === "evaluator" || t.type === "evaluation"
          ? {
              ...t,
              status: (evStatus === "FAIL" ? "FAILED" : "COMPLETED") as TaskStatus,
              score,
              evaluationStatus: evStatus,
              output: feedback,
            }
          : t
      );
      break;
    }

    case "SYNTHESIS_STARTED": {
      let synNode = nextTasks.find((t) => t.id === "synthesis" || t.type === "synthesis");
      if (!synNode) {
        synNode = {
          id: "synthesis",
          title: "Final Synthesis",
          description: "Consolidation & final answer",
          type: "synthesis",
          status: "RUNNING",
          dependencies: [],
        };
        nextTasks.push(synNode);
      } else {
        nextTasks = nextTasks.map((t) =>
          t.id === synNode!.id ? { ...t, status: "RUNNING" as TaskStatus } : t
        );
      }
      break;
    }

    case "SYNTHESIS_COMPLETED": {
      const faVal = payload["final_answer"];
      if (typeof faVal === "string") {
        nextFinalAnswer = faVal;
      }
      nextTasks = nextTasks.map((t) =>
        t.id === "synthesis" || t.type === "synthesis"
          ? {
              ...t,
              status: "COMPLETED" as TaskStatus,
              output: nextFinalAnswer || t.output,
            }
          : t
      );
      break;
    }

    case "RUN_COMPLETED":
      nextRunning = false;
      break;

    case "RUN_FAILED": {
      const errVal = payload["error"];
      nextRunning = false;
      nextError = typeof errVal === "string" ? errVal : event.message;
      break;
    }
  }

  return {
    running: nextRunning,
    tasks: nextTasks,
    events: nextEvents,
    evaluation: nextEvaluation,
    finalAnswer: nextFinalAnswer,
    chunks: nextChunks,
    error: nextError,
  };
}
