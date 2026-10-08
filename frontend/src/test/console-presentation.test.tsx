import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ExecutionConsole, TaskDetails } from "@/components/execution/ExecutionConsole";
import { ExecutionEdge } from "@/components/graph/ExecutionEdge";
import { Position } from "@xyflow/react";
import { adaptRun } from "@/services/adapters";

afterEach(cleanup);

describe("Execution presentation", () => {
  it("renders the returned final answer as markdown without adding retrieval data", () => {
    const run = adaptRun({
      run_id: "test-response",
      final_answer: "## Returned answer\n\n**Reported findings**",
      execution_trace: [{ stage_name: "Synthesis", status: "SUCCESS", duration_ms: 210 }],
    });
    const { rerender } = render(
      <ExecutionConsole
        events={run.events}
        run={run}
        tab="synthesis"
        setTab={vi.fn()}
        running={false}
      />,
    );
    expect(screen.getByRole("heading", { name: "Returned answer" })).toBeInTheDocument();
    expect(screen.getByText("Reported findings").tagName).toBe("STRONG");
    expect(screen.getByText("POST-RUN TRACE")).toBeInTheDocument();
    rerender(
      <ExecutionConsole events={run.events} run={run} tab="rag" setTab={vi.fn()} running={false} />,
    );
    expect(screen.getByText("RAG telemetry unavailable for this run.")).toBeInTheDocument();
  });

  it("shows only returned query, source, chunk identity and similarity", () => {
    const run = adaptRun({
      run_id: "test-rag",
      tasks: [
        {
          id: "retrieval",
          query: "Reported query",
          retrieved_chunks: [
            {
              document_id: "doc-4",
              source: "Reported source",
              chunk_index: 2,
              similarity_score: 0.831,
              content: "Reported chunk",
            },
          ],
        },
      ],
    });
    render(<ExecutionConsole events={[]} run={run} tab="rag" setTab={vi.fn()} running={false} />);
    expect(screen.getByText("Query · Reported query")).toBeInTheDocument();
    expect(screen.getByText("Reported source")).toBeInTheDocument();
    expect(screen.getByText("Document · doc-4")).toBeInTheDocument();
    expect(screen.getByText(/Chunk #2 · Similarity 0.831/)).toBeInTheDocument();
  });

  it("preserves tab interactions and labels conceptual nodes without execution status", () => {
    const setTab = vi.fn();
    render(<ExecutionConsole events={[]} tab="synthesis" setTab={setTab} running={false} />);
    fireEvent.click(screen.getByRole("tab", { name: "Node Inspector" }));
    expect(setTab).toHaveBeenCalledWith("node");
    render(
      <TaskDetails
        task={{
          id: "concept",
          title: "Conceptual node",
          type: "analysis",
          status: "PENDING",
          dependencies: [],
          metadata: { view: "Conceptual architecture; not execution telemetry" },
        }}
      />,
    );
    expect(screen.getByText("ARCHITECTURE")).toBeInTheDocument();
    expect(screen.queryByText("PENDING")).not.toBeInTheDocument();
  });

  it("renders a tiny moving light only when the edge is active", () => {
    const props = {
      id: "a-b",
      source: "a",
      target: "b",
      sourceX: 0,
      sourceY: 0,
      targetX: 100,
      targetY: 0,
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      data: { active: false, completed: true },
    };
    const { container, rerender } = render(
      <svg>
        <ExecutionEdge {...props} />
      </svg>,
    );
    expect(container.querySelector("animateMotion")).toBeNull();
    rerender(
      <svg>
        <ExecutionEdge {...props} data={{ active: true, completed: false }} />
      </svg>,
    );
    expect(container.querySelector("animateMotion")?.getAttribute("dur")).toBe("3.6s");
    expect(container.querySelector("circle")?.getAttribute("r")).toBe("1.8");
  });
});
