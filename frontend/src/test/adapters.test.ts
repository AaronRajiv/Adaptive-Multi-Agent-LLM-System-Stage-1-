import { describe, expect, it } from "vitest";
import { adaptRun } from "@/services/adapters";

describe("Execution response adapter", () => {
  it("maps baseline results without inventing dependencies or RAG data", () => {
    const run = adaptRun({
      run_id: "r1",
      user_task: "Compare options",
      subtasks: [{ id: "a", type: "research", description: "Research options" }],
      research_results: [{ subtask_id: "a", findings: "Real findings" }],
      analysis: "Analysis",
      evaluation: { score: 87, status: "PASS", feedback: "Good" },
      final_answer: "**Real answer**",
      execution_trace: [
        { stage_name: "Research", agent_name: "Researcher", duration_ms: 1234, status: "SUCCESS" },
      ],
    });
    expect(run.tasks[0]?.output).toBe("Real findings");
    expect(run.tasks.every((t) => t.dependencies.length === 0)).toBe(true);
    expect(run.dependenciesAvailable).toBe(false);
    expect(run.tasks.find((t) => t.type === "evaluation")?.score).toBe(87);
    expect(run.chunks).toEqual([]);
    expect(run.events[0]?.source).toBe("POST-RUN");
    expect(run.events[0]?.durationMs).toBe(1234);
  });
  it("preserves declared dependencies and flexible task types", () => {
    const run = adaptRun({
      run_id: "r2",
      task_graph: {
        tasks: [
          { id: "one", type: "custom-tool", status: "SUCCESS", dependencies: [] },
          { id: "two", type: "analysis", status: "BLOCKED", dependencies: ["one"] },
        ],
      },
    });
    expect(run.tasks[0]?.type).toBe("custom-tool");
    expect(run.tasks[1]?.dependencies).toEqual(["one"]);
    expect(run.tasks[1]?.status).toBe("BLOCKED");
    expect(run.dependenciesAvailable).toBe(true);
  });
  it("rejects malformed responses and preserves missing evaluation as unavailable", () => {
    expect(() => adaptRun(null)).toThrow("missing a run ID");
    expect(adaptRun({ run_id: "minimal" }).tasks).toEqual([]);
  });
});
