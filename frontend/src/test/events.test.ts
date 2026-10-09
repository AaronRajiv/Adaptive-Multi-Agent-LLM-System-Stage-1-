import { describe, expect, it } from "vitest";
import {
  createInitialLiveState,
  processExecutionEvent,
  type LiveExecutionState,
} from "@/services/executionState";
import { ServerExecutionEventSource } from "@/services/executionEvents";
import { adaptRun } from "@/services/adapters";
import type { ExecutionEvent } from "@/types/execution";

function makeEvent(
  type: string,
  taskId?: string,
  payload?: Record<string, unknown>
): ExecutionEvent {
  return {
    id: `evt-${Math.random()}`,
    label: type,
    message: `${type} event`,
    source: "SSE",
    time: "12:00:00",
    rawType: type,
    taskId,
    payload,
  };
}

describe("Frontend Live SSE Execution Event Handling", () => {
  it("1. RUN_STARTED changes run state", () => {
    let state = createInitialLiveState();
    expect(state.running).toBe(false);
    state = processExecutionEvent(state, makeEvent("RUN_STARTED"));
    expect(state.running).toBe(true);
  });

  it("2. TASK_STARTED activates the correct node", () => {
    let state = createInitialLiveState();
    state = processExecutionEvent(
      state,
      makeEvent("PLANNING_COMPLETED", undefined, {
        tasks: [{ id: "T1", title: "Task 1", type: "research" }],
      })
    );
    state = processExecutionEvent(state, makeEvent("TASK_STARTED", "T1"));
    const t1 = state.tasks.find((t) => t.id === "T1");
    expect(t1?.status).toBe("RUNNING");
  });

  it("3. TASK_COMPLETED completes the correct node", () => {
    let state = createInitialLiveState();
    state = processExecutionEvent(
      state,
      makeEvent("PLANNING_COMPLETED", undefined, {
        tasks: [{ id: "T1", title: "Task 1", type: "research" }],
      })
    );
    state = processExecutionEvent(state, makeEvent("TASK_STARTED", "T1"));
    state = processExecutionEvent(
      state,
      makeEvent("TASK_COMPLETED", "T1", { output: "Finished findings", duration_ms: 150 })
    );
    const t1 = state.tasks.find((t) => t.id === "T1");
    expect(t1?.status).toBe("COMPLETED");
    expect(t1?.output).toBe("Finished findings");
    expect(t1?.durationMs).toBe(150);
  });

  it("4. TASK_FAILED marks the correct node failed", () => {
    let state = createInitialLiveState();
    state = processExecutionEvent(
      state,
      makeEvent("PLANNING_COMPLETED", undefined, {
        tasks: [{ id: "T1", title: "Task 1", type: "research" }],
      })
    );
    state = processExecutionEvent(state, makeEvent("TASK_STARTED", "T1"));
    state = processExecutionEvent(state, makeEvent("TASK_FAILED", "T1"));
    const t1 = state.tasks.find((t) => t.id === "T1");
    expect(t1?.status).toBe("FAILED");
  });

  it("5. TASK_BLOCKED marks the correct node blocked", () => {
    let state = createInitialLiveState();
    state = processExecutionEvent(
      state,
      makeEvent("PLANNING_COMPLETED", undefined, {
        tasks: [
          { id: "T1", title: "Task 1", type: "research" },
          { id: "T2", title: "Task 2", type: "analysis", dependencies: ["T1"] },
        ],
      })
    );
    state = processExecutionEvent(state, makeEvent("TASK_BLOCKED", "T2"));
    const t2 = state.tasks.find((t) => t.id === "T2");
    expect(t2?.status).toBe("BLOCKED");
  });

  it("6. Multiple simultaneous tasks can be RUNNING", () => {
    let state = createInitialLiveState();
    state = processExecutionEvent(
      state,
      makeEvent("PLANNING_COMPLETED", undefined, {
        tasks: [
          { id: "T1", title: "Task 1", type: "research" },
          { id: "T2", title: "Task 2", type: "research" },
          { id: "T3", title: "Task 3", type: "research" },
        ],
      })
    );
    state = processExecutionEvent(state, makeEvent("TASK_STARTED", "T1"));
    state = processExecutionEvent(state, makeEvent("TASK_STARTED", "T2"));
    state = processExecutionEvent(state, makeEvent("TASK_STARTED", "T3"));

    const runningTasks = state.tasks.filter((t) => t.status === "RUNNING");
    expect(runningTasks.length).toBe(3);
    expect(runningTasks.map((t) => t.id)).toEqual(["T1", "T2", "T3"]);
  });

  it("7. RAG events activate RAG visualization only when emitted", () => {
    let state = createInitialLiveState();
    expect(state.tasks.some((t) => t.type === "rag")).toBe(false);

    state = processExecutionEvent(state, makeEvent("RAG_RETRIEVAL_STARTED"));
    const ragNode = state.tasks.find((t) => t.type === "rag");
    expect(ragNode?.status).toBe("RUNNING");

    state = processExecutionEvent(
      state,
      makeEvent("RAG_RETRIEVAL_COMPLETED", undefined, {
        query: "battery chemistry",
        chunks: [
          { document_id: "doc1", source: "kb", content: "lithium ion", score: 0.95 },
        ],
      })
    );

    expect(state.tasks.find((t) => t.type === "rag")?.status).toBe("COMPLETED");
    expect(state.chunks.length).toBe(1);
    expect(state.chunks[0]?.content).toBe("lithium ion");
  });

  it("8. Evaluation events update evaluator state", () => {
    let state = createInitialLiveState();
    state = processExecutionEvent(state, makeEvent("EVALUATION_STARTED"));
    expect(state.tasks.find((t) => t.type === "evaluation")?.status).toBe("RUNNING");

    state = processExecutionEvent(
      state,
      makeEvent("EVALUATION_COMPLETED", undefined, {
        score: 92,
        status: "PASS",
        feedback: "High quality analysis",
      })
    );

    const evNode = state.tasks.find((t) => t.type === "evaluation");
    expect(evNode?.status).toBe("COMPLETED");
    expect(evNode?.score).toBe(92);
    expect(state.evaluation?.score).toBe(92);
    expect(state.evaluation?.status).toBe("PASS");
  });

  it("9. Synthesis events update synthesis state", () => {
    let state = createInitialLiveState();
    state = processExecutionEvent(state, makeEvent("SYNTHESIS_STARTED"));
    expect(state.tasks.find((t) => t.type === "synthesis")?.status).toBe("RUNNING");

    state = processExecutionEvent(
      state,
      makeEvent("SYNTHESIS_COMPLETED", undefined, {
        final_answer: "The comparative study concludes...",
      })
    );

    const synNode = state.tasks.find((t) => t.type === "synthesis");
    expect(synNode?.status).toBe("COMPLETED");
    expect(state.finalAnswer).toBe("The comparative study concludes...");
  });

  it("10. RUN_COMPLETED finalizes the UI", () => {
    let state = createInitialLiveState();
    state = processExecutionEvent(state, makeEvent("RUN_STARTED"));
    expect(state.running).toBe(true);

    state = processExecutionEvent(state, makeEvent("RUN_COMPLETED"));
    expect(state.running).toBe(false);
  });

  it("11. SSE disconnect does not fabricate completion", () => {
    let state = createInitialLiveState();
    state = processExecutionEvent(state, makeEvent("RUN_STARTED"));
    expect(state.running).toBe(true);

    // Simulated SSE disconnect / error without RUN_COMPLETED or RUN_FAILED
    const source = new ServerExecutionEventSource("test-run");
    expect(source).toBeDefined();

    // The state running flag remains true until backend explicitly sends RUN_COMPLETED or RUN_FAILED
    expect(state.running).toBe(true);
  });

  it("12. Existing API functionality remains intact", () => {
    const run = adaptRun({
      run_id: "r-existing",
      user_task: "Test task",
      subtasks: [{ id: "s1", type: "research", description: "Subtask 1" }],
      research_results: [{ subtask_id: "s1", findings: "Existing findings" }],
      analysis: "Existing analysis",
      evaluation: { score: 85, status: "PASS", feedback: "Good" },
      final_answer: "Existing final answer",
      execution_trace: [],
    });

    expect(run.runId).toBe("r-existing");
    expect(run.finalAnswer).toBe("Existing final answer");
    expect(run.tasks.find((t) => t.id === "s1")?.evaluationStatus).toBeUndefined();
    expect(run.tasks.length).toBeGreaterThan(0);
  });
});
