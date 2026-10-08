import type { ExecutionModel, ExecutionTask, ProvenanceChunk, TaskStatus } from "@/types/execution";
const record = (v: unknown): Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
const text = (v: unknown) => (typeof v === "string" ? v : "");
const array = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const numeric = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
export function normalizeStatus(v: unknown): TaskStatus {
  const s = text(v).toUpperCase();
  if (s === "SUCCESS" || s === "SUCCEEDED" || s === "DONE" || s === "COMPLETED") return "COMPLETED";
  if (s === "FAILED" || s === "FAIL" || s === "ERROR") return "FAILED";
  if (s === "RUNNING" || s === "READY" || s === "BLOCKED") return s;
  return "PENDING";
}
function chunksFrom(v: unknown): ProvenanceChunk[] {
  return array(v).map((item) => {
    const c = record(item);
    return {
      documentId: text(c["document_id"]) || undefined,
      chunkIndex: numeric(c["chunk_index"]),
      source: text(c["source"]) || undefined,
      content: text(c["content"]) || undefined,
      similarity: numeric(c["similarity_score"] ?? c["score"]),
    };
  });
}
export function adaptRun(value: unknown): ExecutionModel {
  const r = record(value);
  if (!text(r["run_id"]))
    throw new Error("The response is missing a run ID. Check the backend API contract.");
  const graph = record(r["task_graph"] ?? r["graph"]);
  const explicit = array(graph["tasks"] ?? graph["nodes"] ?? r["tasks"]);
  const rawTasks = explicit.length ? explicit : array(r["subtasks"]);
  const trace = array(r["execution_trace"]).map(record);
  const results = array(r["research_results"]).map(record);
  const tasks: ExecutionTask[] = rawTasks.map((value, index) => {
    const t = record(value);
    const data = record(t["data"]);
    const id = text(t["id"] ?? t["task_id"]) || `task-${index}`;
    const result = results.find((v) => v["subtask_id"] === id);
    const ownTrace = trace.find((v) => v["task_id"] === id || v["subtask_id"] === id);
    const metadata = record(t["metadata"]);
    const chunks = chunksFrom(t["retrieved_chunks"] ?? metadata["retrieved_chunks"]);
    return {
      id,
      title: text(t["title"] ?? data["title"]) || text(t["description"]).slice(0, 70) || id,
      description: text(t["description"] ?? data["description"]),
      type: text(t["type"] ?? t["capability"] ?? data["type"]) || "task",
      capability: text(t["capability"]),
      status: t["status"]
        ? normalizeStatus(t["status"])
        : result
          ? "COMPLETED"
          : ownTrace
            ? normalizeStatus(ownTrace["status"])
            : "PENDING",
      dependencies: array(t["dependencies"] ?? t["depends_on"])
        .map((v) => (typeof v === "string" ? v : text(record(v)["id"])))
        .filter(Boolean),
      durationMs: numeric(t["duration_ms"] ?? ownTrace?.["duration_ms"]),
      agent: text(t["agent"] ?? ownTrace?.["agent_name"]),
      input: text(t["input"]),
      output: text(t["output"] ?? result?.["findings"]),
      metadata,
      chunks,
      query: text(t["query"] ?? metadata["query"]),
    };
  });
  for (const value of array(graph["edges"] ?? r["edges"])) {
    const e = record(value);
    const target = tasks.find((t) => t["id"] === text(e["target"] ?? e["to"]));
    const source = text(e["source"] ?? e["from"]);
    if (
      target &&
      source &&
      tasks.some((t) => t["id"] === source) &&
      !target.dependencies.includes(source)
    )
      target.dependencies.push(source);
  }
  if (!explicit.length) {
    if (text(r["analysis"])) {
      const tr = trace.find((t) => /analy/i.test(text(t["stage_name"])));
      tasks.push({
        id: "__analysis",
        title: "Analysis",
        description: "Returned analytical findings",
        type: "analysis",
        status: "COMPLETED",
        dependencies: [],
        output: text(r["analysis"]),
        durationMs: numeric(tr?.["duration_ms"]),
        agent: text(tr?.["agent_name"]),
      });
    }
    const ev = record(r["evaluation"]);
    if (Object.keys(ev).length) {
      const tr = trace.find((t) => /evaluat/i.test(text(t["stage_name"])));
      tasks.push({
        id: "__evaluation",
        title: "Quality evaluation",
        type: "evaluation",
        status:
          ev["status"] === "PASS" ? "COMPLETED" : ev["status"] === "FAIL" ? "FAILED" : "PENDING",
        dependencies: [],
        score: numeric(ev["score"]),
        evaluationStatus: text(ev["status"]),
        output: text(ev["feedback"]),
        durationMs: numeric(tr?.["duration_ms"]),
        agent: text(tr?.["agent_name"]),
      });
    }
    if (text(r["final_answer"]))
      tasks.push({
        id: "__synthesis",
        title: "Final synthesis",
        description: "Final answer returned by the system",
        type: "synthesis",
        status: "COMPLETED",
        dependencies: [],
        output: text(r["final_answer"]),
      });
  }
  const chunks = [
    ...chunksFrom(r["retrieved_chunks"] ?? r["rag_provenance"]),
    ...tasks.flatMap((t) => t["chunks"] ?? []),
  ];
  return {
    runId: text(r["run_id"]),
    task: text(r["user_task"]),
    tasks,
    chunks,
    finalAnswer: text(r["final_answer"]),
    createdAt: text(r["created_at"]),
    dependenciesAvailable:
      rawTasks.some((v) => {
        const t = record(v);
        return Array.isArray(t["dependencies"]) || Array.isArray(t["depends_on"]);
      }) ||
      Array.isArray(graph["edges"]) ||
      Array.isArray(r["edges"]),
    events: trace.map((t, i) => ({
      id: `trace-${i}`,
      label: text(t["stage_name"]).toUpperCase() || "STAGE",
      message: [text(t["agent_name"]), text(t["summary"])].filter(Boolean).join(" · "),
      source: "POST-RUN",
      durationMs: numeric(t["duration_ms"]),
      status: normalizeStatus(t["status"]),
    })),
  };
}
export const architectureTasks: ExecutionTask[] = [
  {
    id: "user",
    title: "User task",
    description: "Complex analytical request",
    type: "input",
    dependencies: [],
  },
  {
    id: "api",
    title: "API Gateway",
    description: "Request validation & dispatch",
    type: "gateway",
    dependencies: ["user"],
  },
  {
    id: "planner",
    title: "Task Planner",
    description: "Decompose the analytical objective",
    type: "planning",
    dependencies: ["api"],
  },
  {
    id: "graph",
    title: "Dynamic Task Graph",
    description: "Tasks, capabilities & dependencies",
    type: "graph",
    dependencies: ["planner"],
  },
  {
    id: "orchestrator",
    title: "Orchestrator",
    description: "Dependency-aware bounded execution",
    type: "orchestration",
    dependencies: ["graph"],
  },
  {
    id: "agents",
    title: "Specialist Agents",
    description: "Research & analytical reasoning",
    type: "research",
    dependencies: ["orchestrator"],
  },
  {
    id: "tools",
    title: "Tool Registry",
    description: "Controlled capability access",
    type: "tools",
    dependencies: ["orchestrator"],
  },
  {
    id: "rag",
    title: "Knowledge Retrieval",
    description: "Conditional context enrichment",
    type: "rag",
    dependencies: ["orchestrator"],
  },
  {
    id: "evaluator",
    title: "Evaluator",
    description: "When exposed by the backend",
    type: "evaluation",
    dependencies: ["agents"],
  },
  {
    id: "synthesis",
    title: "Final Synthesis",
    description: "Consolidated analytical response",
    type: "synthesis",
    dependencies: ["evaluator"],
  },
].map((t) => ({
  ...t,
  status: "PENDING",
  metadata: { view: "Conceptual architecture; not execution telemetry" },
}));
