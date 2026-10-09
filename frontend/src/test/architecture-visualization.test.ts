import { describe, expect, it } from "vitest";
import {
  createInitialArchitectureState,
  updateArchitectureFromEvent,
  type ArchitectureState,
} from "@/services/architectureState";
import type { ExecutionEvent } from "@/types/execution";

function makeEvent(
  type: string,
  taskId?: string,
  payload?: Record<string, unknown>
): ExecutionEvent {
  return {
    id: `evt-${Math.random()}`,
    label: type,
    message: `${type} message`,
    source: "SSE",
    time: "12:00:00",
    rawType: type,
    taskId,
    payload,
  };
}

describe("Live Architecture Visualization & Data-Flow Mapping", () => {
  it("1. Initializes with all nodes IDLE and conditional nodes inactive", () => {
    const state = createInitialArchitectureState();
    expect(state.nodes.every((n) => n.status === "IDLE")).toBe(true);
    expect(state.particles.length).toBe(0);
    expect(state.ragUsed).toBe(false);
    expect(state.toolsUsed).toBe(false);
  });

  it("2. RUN_STARTED activates User Task & API Gateway and spawns particle", () => {
    let state = createInitialArchitectureState();
    state = updateArchitectureFromEvent(state, makeEvent("RUN_STARTED"));

    const userNode = state.nodes.find((n) => n.id === "user_task");
    const apiNode = state.nodes.find((n) => n.id === "api_gateway");

    expect(userNode?.status).toBe("COMPLETED");
    expect(apiNode?.status).toBe("ACTIVE");
    expect(state.particles.length).toBeGreaterThan(0);
    expect(state.particles[0]?.source).toBe("user_task");
    expect(state.particles[0]?.target).toBe("api_gateway");
  });

  it("3. PLANNING_STARTED and PLANNING_COMPLETED transition Planner to COMPLETED and spawn particle", () => {
    let state = createInitialArchitectureState();
    state = updateArchitectureFromEvent(state, makeEvent("PLANNING_STARTED"));

    expect(state.nodes.find((n) => n.id === "planner")?.status).toBe("ACTIVE");

    state = updateArchitectureFromEvent(
      state,
      makeEvent("PLANNING_COMPLETED", undefined, { tasks: [{ id: "T1" }, { id: "T2" }] })
    );

    expect(state.nodes.find((n) => n.id === "planner")?.status).toBe("COMPLETED");
    expect(state.nodes.find((n) => n.id === "task_graph")?.status).toBe("ACTIVE");

    const plannerParticle = state.particles.find(
      (p) => p.source === "planner" && p.target === "task_graph"
    );
    expect(plannerParticle).toBeDefined();
  });

  it("4. TASK_STARTED activates Orchestrator and Agent Execution", () => {
    let state = createInitialArchitectureState();
    state = updateArchitectureFromEvent(state, makeEvent("TASK_STARTED", "T1"));

    expect(state.nodes.find((n) => n.id === "orchestrator")?.status).toBe("ACTIVE");
    expect(state.nodes.find((n) => n.id === "agents")?.status).toBe("ACTIVE");

    const orchParticle = state.particles.find(
      (p) => p.source === "orchestrator" && p.target === "agents"
    );
    expect(orchParticle).toBeDefined();
  });

  it("5. Parallel execution produces multiple simultaneous particles", () => {
    let state = createInitialArchitectureState();
    state = updateArchitectureFromEvent(state, makeEvent("TASK_STARTED", "T1"));
    state = updateArchitectureFromEvent(state, makeEvent("TASK_STARTED", "T2"));

    const particlesToAgents = state.particles.filter(
      (p) => p.source === "orchestrator" && p.target === "agents"
    );
    expect(particlesToAgents.length).toBe(2);

    const agentNode = state.nodes.find((n) => n.id === "agents");
    expect(agentNode?.activeCount).toBe(2);
  });

  it("6. RAG Retrieval remains IDLE when not invoked, and activates on RAG_STARTED", () => {
    let state = createInitialArchitectureState();

    // Normal task execution without RAG
    state = updateArchitectureFromEvent(state, makeEvent("TASK_STARTED", "T1"));
    expect(state.nodes.find((n) => n.id === "rag")?.status).toBe("IDLE");
    expect(state.ragUsed).toBe(false);
    expect(state.particles.some((p) => p.target === "rag" || p.source === "rag")).toBe(false);

    // Explicit RAG event
    state = updateArchitectureFromEvent(state, makeEvent("RAG_STARTED"));
    expect(state.nodes.find((n) => n.id === "rag")?.status).toBe("ACTIVE");
    expect(state.ragUsed).toBe(true);

    const ragParticle = state.particles.find(
      (p) => p.source === "orchestrator" && p.target === "rag"
    );
    expect(ragParticle).toBeDefined();
  });

  it("7. Tool Registry remains IDLE when not invoked, and activates on TOOL_STARTED", () => {
    let state = createInitialArchitectureState();

    state = updateArchitectureFromEvent(state, makeEvent("TASK_STARTED", "T1"));
    expect(state.nodes.find((n) => n.id === "tools")?.status).toBe("IDLE");
    expect(state.toolsUsed).toBe(false);

    state = updateArchitectureFromEvent(state, makeEvent("TOOL_STARTED"));
    expect(state.nodes.find((n) => n.id === "tools")?.status).toBe("ACTIVE");
    expect(state.toolsUsed).toBe(true);
  });

  it("8. Evaluation, Synthesis, and Final Answer complete in order", () => {
    let state = createInitialArchitectureState();

    state = updateArchitectureFromEvent(state, makeEvent("EVALUATION_STARTED"));
    expect(state.nodes.find((n) => n.id === "evaluation")?.status).toBe("ACTIVE");

    state = updateArchitectureFromEvent(
      state,
      makeEvent("EVALUATION_COMPLETED", undefined, { score: 95, status: "PASS" })
    );
    expect(state.nodes.find((n) => n.id === "evaluation")?.status).toBe("COMPLETED");

    state = updateArchitectureFromEvent(state, makeEvent("SYNTHESIS_STARTED"));
    expect(state.nodes.find((n) => n.id === "synthesis")?.status).toBe("ACTIVE");

    state = updateArchitectureFromEvent(state, makeEvent("SYNTHESIS_COMPLETED"));
    expect(state.nodes.find((n) => n.id === "synthesis")?.status).toBe("COMPLETED");
    expect(state.nodes.find((n) => n.id === "final_answer")?.status).toBe("COMPLETED");
  });

  it("9. RUN_COMPLETED settles active nodes into COMPLETED state", () => {
    let state = createInitialArchitectureState();
    state = updateArchitectureFromEvent(state, makeEvent("RUN_STARTED"));
    state = updateArchitectureFromEvent(state, makeEvent("TASK_STARTED", "T1"));
    state = updateArchitectureFromEvent(state, makeEvent("RUN_COMPLETED"));

    const activeNodes = state.nodes.filter((n) => n.status === "ACTIVE");
    expect(activeNodes.length).toBe(0);
  });

  it("10. TASK_FAILED marks nodes as FAILED", () => {
    let state = createInitialArchitectureState();
    state = updateArchitectureFromEvent(state, makeEvent("TASK_STARTED", "T1"));
    state = updateArchitectureFromEvent(state, makeEvent("TASK_FAILED", "T1"));

    expect(state.nodes.find((n) => n.id === "agents")?.status).toBe("FAILED");
    expect(state.nodes.find((n) => n.id === "orchestrator")?.status).toBe("FAILED");
  });
});
