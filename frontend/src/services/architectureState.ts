import type {
  ArchitectureEdge,
  ArchitectureNode,
  ArchitectureNodeId,
  ArchitectureParticle,
} from "@/types/architecture";
import type { ExecutionEvent } from "@/types/execution";

export const ARCHITECTURE_NODES_INITIAL: ArchitectureNode[] = [
  // Stage 1: Pipeline Input & Planning (Row 1)
  {
    id: "user_task",
    label: "User Task",
    subtitle: "Input Prompt",
    status: "IDLE",
    x: 14,
    y: 15,
  },
  {
    id: "api_gateway",
    label: "API Gateway",
    subtitle: "Validation & Bus",
    status: "IDLE",
    x: 38,
    y: 15,
  },
  {
    id: "planner",
    label: "Task Planner",
    subtitle: "PlannerAgent",
    status: "IDLE",
    x: 62,
    y: 15,
  },
  {
    id: "task_graph",
    label: "Dynamic Task Graph",
    subtitle: "DAG Decomposition",
    status: "IDLE",
    x: 86,
    y: 15,
  },

  // Stage 2: Central Orchestration Engine (Row 2)
  {
    id: "orchestrator",
    label: "Orchestrator",
    subtitle: "Bounded Parallel Engine",
    status: "IDLE",
    x: 50,
    y: 40,
  },

  // Stage 3: Specialist Execution & Grounding Services (Row 3)
  {
    id: "rag",
    label: "RAG Retrieval",
    subtitle: "Knowledge Context",
    status: "IDLE",
    isConditional: true,
    x: 20,
    y: 65,
  },
  {
    id: "agents",
    label: "Agent Execution",
    subtitle: "Research & Analyst Agents",
    status: "IDLE",
    x: 50,
    y: 65,
  },
  {
    id: "tools",
    label: "Tool Registry",
    subtitle: "Controlled Tools",
    status: "IDLE",
    isConditional: true,
    x: 80,
    y: 65,
  },

  // Stage 4: Evaluation Audit & Final Synthesis (Row 4)
  {
    id: "evaluation",
    label: "Evaluation",
    subtitle: "Quality Audit",
    status: "IDLE",
    x: 26,
    y: 88,
  },
  {
    id: "synthesis",
    label: "Synthesis",
    subtitle: "Consensus Synthesis",
    status: "IDLE",
    x: 52,
    y: 88,
  },
  {
    id: "final_answer",
    label: "Final Answer",
    subtitle: "Structured Result",
    status: "IDLE",
    x: 78,
    y: 88,
  },
];

export const ARCHITECTURE_EDGES: ArchitectureEdge[] = [
  { id: "e-user-api", source: "user_task", target: "api_gateway" },
  { id: "e-api-planner", source: "api_gateway", target: "planner" },
  { id: "e-planner-graph", source: "planner", target: "task_graph" },
  { id: "e-graph-orch", source: "task_graph", target: "orchestrator" },
  { id: "e-orch-agents", source: "orchestrator", target: "agents" },
  { id: "e-orch-rag", source: "orchestrator", target: "rag", isConditional: true },
  { id: "e-rag-agents", source: "rag", target: "agents", isConditional: true },
  { id: "e-orch-tools", source: "orchestrator", target: "tools", isConditional: true },
  { id: "e-tools-agents", source: "tools", target: "agents", isConditional: true },
  { id: "e-agents-eval", source: "agents", target: "evaluation" },
  { id: "e-eval-synth", source: "evaluation", target: "synthesis" },
  { id: "e-synth-final", source: "synthesis", target: "final_answer" },
];

export interface ArchitectureState {
  nodes: ArchitectureNode[];
  particles: ArchitectureParticle[];
  ragUsed: boolean;
  toolsUsed: boolean;
  activeTasks: Map<string, string>;
  selectedStrategy?: string | undefined;
}

export function createInitialArchitectureState(): ArchitectureState {
  return {
    nodes: ARCHITECTURE_NODES_INITIAL.map((n) => ({ ...n })),
    particles: [],
    ragUsed: false,
    toolsUsed: false,
    activeTasks: new Map(),
    selectedStrategy: undefined,
  };
}

let particleCounter = 0;

export function updateArchitectureFromEvent(
  state: ArchitectureState,
  event: ExecutionEvent
): ArchitectureState {
  const rawType = event.rawType || event.label;
  const taskId = event.taskId;
  const now = Date.now();

  const nextNodes = state.nodes.map((n) => ({ ...n }));
  let nextParticles = [...state.particles];
  let nextRagUsed = state.ragUsed;
  let nextToolsUsed = state.toolsUsed;
  const nextActiveTasks = new Map(state.activeTasks);
  let nextSelectedStrategy = state.selectedStrategy;

  const getNode = (id: ArchitectureNodeId) => nextNodes.find((n) => n.id === id);
  const setStatus = (id: ArchitectureNodeId, status: ArchitectureNode["status"], detail?: string) => {
    const node = getNode(id);
    if (node) {
      node.status = status;
      if (detail !== undefined) node.detail = detail;
    }
  };

  const spawnParticle = (
    from: ArchitectureNodeId,
    to: ArchitectureNodeId,
    durationMs = 750,
    color?: string
  ) => {
    particleCounter++;
    nextParticles.push({
      id: `particle-${particleCounter}-${now}`,
      source: from,
      target: to,
      startTime: now,
      durationMs,
      label: event.message,
      ...(color ? { color } : {}),
    });
  };

  switch (rawType) {
    case "RUN_STARTED":
    case "TASK_RECEIVED":
      setStatus("user_task", "COMPLETED");
      setStatus("api_gateway", "ACTIVE");
      spawnParticle("user_task", "api_gateway", 600);
      setTimeout(() => setStatus("api_gateway", "COMPLETED"), 400);
      spawnParticle("api_gateway", "planner", 700);
      break;

    case "PLANNING_STARTED":
      setStatus("api_gateway", "COMPLETED");
      setStatus("planner", "ACTIVE");
      break;

    case "PLANNING_COMPLETED": {
      setStatus("planner", "COMPLETED");
      setStatus("task_graph", "ACTIVE");
      spawnParticle("planner", "task_graph", 650);

      const tasks = event.payload && Array.isArray(event.payload["tasks"]) ? event.payload["tasks"] : [];
      const count = tasks.length || (event.payload && Array.isArray(event.payload["task_ids"]) ? (event.payload["task_ids"] as unknown[]).length : 0);
      const graphNode = getNode("task_graph");
      if (graphNode) {
        graphNode.totalCount = count;
        graphNode.detail = `${count} task(s) planned`;
      }
      setTimeout(() => setStatus("task_graph", "COMPLETED"), 500);
      spawnParticle("task_graph", "orchestrator", 700);
      break;
    }

    case "TASK_STARTED": {
      setStatus("orchestrator", "ACTIVE");
      setStatus("agents", "ACTIVE");

      if (taskId) {
        nextActiveTasks.set(taskId, "RUNNING");
      }

      const agentNode = getNode("agents");
      if (agentNode) {
        const runningCount = Array.from(nextActiveTasks.values()).filter((s) => s === "RUNNING").length;
        agentNode.activeCount = runningCount;
        agentNode.detail = `${runningCount} parallel task(s) active`;
      }

      spawnParticle("orchestrator", "agents", 750);
      break;
    }

    case "TASK_COMPLETED": {
      if (taskId) {
        nextActiveTasks.set(taskId, "COMPLETED");
      }
      const agentNode = getNode("agents");
      if (agentNode) {
        const runningCount = Array.from(nextActiveTasks.values()).filter((s) => s === "RUNNING").length;
        const completedCount = Array.from(nextActiveTasks.values()).filter((s) => s === "COMPLETED").length;
        agentNode.activeCount = runningCount;
        agentNode.completedCount = completedCount;
        agentNode.detail = runningCount > 0 ? `${runningCount} running · ${completedCount} done` : `${completedCount} task(s) done`;

        if (runningCount === 0 && completedCount > 0) {
          agentNode.status = "COMPLETED";
        }
      }
      break;
    }

    case "TASK_FAILED": {
      if (taskId) nextActiveTasks.set(taskId, "FAILED");
      setStatus("agents", "FAILED", `Task ${taskId || ""} failed`);
      setStatus("orchestrator", "FAILED");
      break;
    }

    case "TASK_BLOCKED": {
      if (taskId) nextActiveTasks.set(taskId, "BLOCKED");
      setStatus("agents", "BLOCKED", `Task ${taskId || ""} blocked`);
      break;
    }

    case "RAG_STARTED":
    case "RAG_RETRIEVAL_STARTED":
      nextRagUsed = true;
      setStatus("rag", "ACTIVE", "Retrieving context...");
      spawnParticle("orchestrator", "rag", 700);
      break;

    case "RAG_COMPLETED":
    case "RAG_RETRIEVAL_COMPLETED":
      nextRagUsed = true;
      setStatus("rag", "COMPLETED", "Context grounding ready");
      spawnParticle("rag", "agents", 700);
      break;

    case "TOOL_STARTED":
    case "TOOL_INVOCATION_STARTED":
    case "TOOL_INVOKED":
      nextToolsUsed = true;
      setStatus("tools", "ACTIVE", "Invoking tool...");
      spawnParticle("orchestrator", "tools", 700);
      break;

    case "TOOL_COMPLETED":
      nextToolsUsed = true;
      setStatus("tools", "COMPLETED", "Tool completed");
      spawnParticle("tools", "agents", 700);
      break;

    case "EVALUATION_STARTED":
      setStatus("evaluation", "ACTIVE", "Auditing quality");
      spawnParticle("agents", "evaluation", 750);
      break;

    case "EVALUATION_COMPLETED": {
      const score = event.payload?.["score"];
      const evStatus = event.payload?.["status"];
      const detail = score !== undefined ? `Score ${score}/100 (${evStatus || "PASS"})` : "Evaluation pass";
      setStatus("evaluation", evStatus === "FAIL" ? "FAILED" : "COMPLETED", detail);
      spawnParticle("evaluation", "synthesis", 750);
      break;
    }

    case "SYNTHESIS_STARTED":
      setStatus("synthesis", "ACTIVE", "Consolidating response");
      break;

    case "STRATEGY_SELECTED": {
      const strat = (event.payload?.["strategy"] as string | undefined) || "MULTI_AGENT";
      nextSelectedStrategy = strat;
      if (strat === "DIRECT") {
        setStatus("api_gateway", "COMPLETED", "Direct Execution");
        spawnParticle("api_gateway", "synthesis", 700);
      } else if (strat === "SINGLE_AGENT") {
        setStatus("api_gateway", "COMPLETED", "Single-Agent Execution");
      }
      break;
    }

    case "SYNTHESIS_COMPLETED":
      setStatus("synthesis", "COMPLETED", "Final answer compiled");
      setStatus("final_answer", "COMPLETED", "Response ready");
      spawnParticle("synthesis", "final_answer", 750);
      break;

    case "RUN_COMPLETED":
      if (nextSelectedStrategy === "DIRECT") {
        setStatus("user_task", "COMPLETED");
        setStatus("api_gateway", "COMPLETED");
        setStatus("synthesis", "COMPLETED");
        setStatus("final_answer", "COMPLETED");
      } else if (nextSelectedStrategy === "SINGLE_AGENT") {
        setStatus("user_task", "COMPLETED");
        setStatus("api_gateway", "COMPLETED");
        setStatus("orchestrator", "COMPLETED");
        setStatus("agents", "COMPLETED");
        setStatus("synthesis", "COMPLETED");
        setStatus("final_answer", "COMPLETED");
        if (nextRagUsed) setStatus("rag", "COMPLETED");
      } else {
        nextNodes.forEach((node) => {
          if (node.id === "rag" && !nextRagUsed) return;
          if (node.id === "tools" && !nextToolsUsed) return;
          if (node.status !== "FAILED" && node.status !== "BLOCKED") {
            node.status = "COMPLETED";
          }
        });
      }
      break;

    case "RUN_FAILED":
      nextNodes.forEach((node) => {
        if (node.status === "ACTIVE") {
          node.status = "FAILED";
        }
      });
      break;
  }

  nextParticles = nextParticles.filter((p) => now - p.startTime < p.durationMs + 200);

  return {
    nodes: nextNodes,
    particles: nextParticles,
    ragUsed: nextRagUsed,
    toolsUsed: nextToolsUsed,
    activeTasks: nextActiveTasks,
    selectedStrategy: nextSelectedStrategy,
  };
}

export function updateArchitectureFromModel(model: {
  tasks?: Array<{ status?: string | undefined; type?: string | undefined; score?: number | undefined; evaluationStatus?: string | undefined }>;
  chunks?: unknown[];
  executionStrategy?: string | undefined;
}): ArchitectureState {
  const state = createInitialArchitectureState();
  const ragUsed = Boolean(model.chunks && model.chunks.length > 0);
  state.ragUsed = ragUsed;
  state.selectedStrategy = model.executionStrategy;

  const nodeMap = new Map<ArchitectureNodeId, ArchitectureNode>();
  state.nodes.forEach((n) => nodeMap.set(n.id, n));

  const setNode = (id: ArchitectureNodeId, status: ArchitectureNode["status"], detail?: string) => {
    const node = nodeMap.get(id);
    if (node) {
      node.status = status;
      if (detail !== undefined) node.detail = detail;
    }
  };

  setNode("user_task", "COMPLETED");
  setNode("api_gateway", "COMPLETED");

  if (model.executionStrategy === "DIRECT") {
    setNode("planner", "IDLE", "Bypassed (Direct)");
    setNode("task_graph", "IDLE", "Bypassed (Direct)");
    setNode("orchestrator", "IDLE", "Bypassed (Direct)");
    setNode("agents", "IDLE", "Bypassed (Direct)");
    setNode("evaluation", "IDLE", "Not Applicable");
    setNode("synthesis", "COMPLETED", "Direct response");
    setNode("final_answer", "COMPLETED", "Response ready");
    return state;
  }

  if (model.executionStrategy === "SINGLE_AGENT") {
    setNode("planner", "IDLE", "Single-agent");
    setNode("task_graph", "IDLE", "Single-agent");
    setNode("orchestrator", "COMPLETED");
    setNode("agents", "COMPLETED", "1 task executed");
    if (ragUsed) {
      setNode("rag", "COMPLETED", `${model.chunks?.length} chunk(s) retrieved`);
    }
    setNode("evaluation", "IDLE", "Evaluation skipped");
    setNode("synthesis", "COMPLETED", "Single-agent response");
    setNode("final_answer", "COMPLETED", "Response ready");
    return state;
  }

  setNode("planner", "COMPLETED");

  const taskCount = model.tasks?.length || 0;
  setNode("task_graph", "COMPLETED", `${taskCount} task(s) planned`);
  setNode("orchestrator", "COMPLETED");

  if (ragUsed) {
    setNode("rag", "COMPLETED", `${model.chunks?.length} chunk(s) retrieved`);
  }

  const completedCount = model.tasks?.filter((t) => t.status === "COMPLETED").length || 0;
  setNode("agents", "COMPLETED", `${completedCount} task(s) done`);

  const evalTask = model.tasks?.find((t) => t.type === "evaluation");
  if (evalTask) {
    const score = evalTask.score;
    const evStatus = evalTask.evaluationStatus || (evalTask.status === "COMPLETED" ? "PASS" : "FAIL");
    setNode(
      "evaluation",
      evStatus === "FAIL" ? "FAILED" : "COMPLETED",
      score !== undefined ? `Score ${score}/100 (${evStatus})` : "Quality audit pass"
    );
  } else {
    setNode("evaluation", "COMPLETED", "Quality audit pass");
  }

  setNode("synthesis", "COMPLETED", "Consolidated answer");
  setNode("final_answer", "COMPLETED", "Response ready");

  return state;
}
