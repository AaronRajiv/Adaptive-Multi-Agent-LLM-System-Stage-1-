import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import {
  activityReducer,
  createInitialActivityState,
  sanitizeHttpUrl,
} from "../services/activityReducer";
import { ResearchActivityFeed } from "../components/chat/ResearchActivityFeed";
import { TypewriterMarkdown } from "../components/chat/TypewriterMarkdown";
import type { ExecutionEvent } from "../types/execution";

describe("Research Activity Reducer", () => {
  it("deduplicates events with identical seq numbers", () => {
    let state = createInitialActivityState("run-1");

    state = activityReducer(state, {
      type: "PROCESS_EVENT",
      event: {
        seq: 1,
        type: "WEB_SEARCH_STARTED",
        metadata: { query: "FIA 2026 regulations" },
      },
    });
    expect(state.steps).toHaveLength(1);
    expect(state.summary.queriesCount).toBe(1);

    // Duplicate event with same seq 1
    state = activityReducer(state, {
      type: "PROCESS_EVENT",
      event: {
        seq: 1,
        type: "WEB_SEARCH_STARTED",
        metadata: { query: "FIA 2026 regulations" },
      },
    });
    expect(state.steps).toHaveLength(1);
    expect(state.summary.queriesCount).toBe(1);
  });

  it("tolerates out-of-order events and maintains monotonic order in steps", () => {
    let state = createInitialActivityState("run-2");

    // Receive seq 3 before seq 1 and 2
    state = activityReducer(state, {
      type: "PROCESS_EVENT",
      event: {
        seq: 3,
        type: "WEB_FETCH_COMPLETED",
        metadata: { url: "https://example.com/page", title: "Page 3", domain: "example.com" },
      },
    });

    state = activityReducer(state, {
      type: "PROCESS_EVENT",
      event: {
        seq: 1,
        type: "WEB_SEARCH_STARTED",
        metadata: { query: "Test Query" },
      },
    });

    state = activityReducer(state, {
      type: "PROCESS_EVENT",
      event: {
        seq: 2,
        type: "WEB_SEARCH_COMPLETED",
        metadata: { query: "Test Query", urls: ["https://example.com/page"] },
      },
    });

    expect(state.steps).toHaveLength(3);
    expect(state.steps[0].seq).toBe(1);
    expect(state.steps[1].seq).toBe(2);
    expect(state.steps[2].seq).toBe(3);
    expect(state.lastSeq).toBe(3);
  });

  it("handles reconnecting / lost connection state honestly", () => {
    let state = createInitialActivityState("run-3");

    state = activityReducer(state, {
      type: "SET_RECONNECTING",
      message: "lost connection, checking run status...",
    });

    expect(state.status).toBe("reconnecting");
    expect(state.statusMessage).toBe("lost connection, checking run status...");

    // When connection is restored and run finishes
    state = activityReducer(state, {
      type: "FINALIZE_RUN",
      status: "done",
      finalAnswer: "According to [web:a1b2c3d4].",
    });

    expect(state.status).toBe("done");
    expect(state.statusMessage).toBeUndefined();
  });

  it("groups parallel tasks by task_id with distinct subtask progress", () => {
    let state = createInitialActivityState("run-parallel");

    state = activityReducer(state, {
      type: "PROCESS_EVENT",
      event: {
        seq: 1,
        task_id: "T1",
        type: "TASK_STARTED",
      },
    });

    state = activityReducer(state, {
      type: "PROCESS_EVENT",
      event: {
        seq: 2,
        task_id: "T2",
        type: "TASK_STARTED",
      },
    });

    state = activityReducer(state, {
      type: "PROCESS_EVENT",
      event: {
        seq: 3,
        task_id: "T1",
        type: "TASK_COMPLETED",
      },
    });

    expect(state.taskGroups["T1"]).toBeDefined();
    expect(state.taskGroups["T2"]).toBeDefined();
    expect(state.taskGroups["T1"].status).toBe("done");
    expect(state.taskGroups["T2"].status).toBe("running");
  });
});

describe("ResearchActivityFeed Component & Security Defense", () => {
  it("renders malicious page title as inert plain text without executing XSS", () => {
    const maliciousTitle = '<img src=x onerror=alert(1)>';
    const events: ExecutionEvent[] = [
      {
        id: "ev-1",
        label: "WEB_FETCH_COMPLETED",
        message: `Read ${maliciousTitle}`,
        source: "SSE",
        rawType: "WEB_FETCH_COMPLETED",
        payload: {
          seq: 1,
          url: "https://example.com/exploit",
          domain: "example.com",
          title: maliciousTitle,
          status: "completed",
        },
      },
    ];

    const { container } = render(
      <ResearchActivityFeed events={events} running={true} runId="test-sec" />
    );

    // Verify no <img> tag was rendered in the DOM
    const imgElement = container.querySelector("img");
    expect(imgElement).toBeNull();

    // Verify the raw text appears safely escaped as text content
    const matchingElements = screen.getAllByText((_content, element) =>
      element?.textContent?.includes(maliciousTitle) || false
    );
    expect(matchingElements.length).toBeGreaterThanOrEqual(1);
  });

  it("neutralizes dangerous javascript: URLs into inert text", () => {
    expect(sanitizeHttpUrl("javascript:alert(document.cookie)")).toBeUndefined();
    expect(sanitizeHttpUrl("data:text/html,<script>alert(1)</script>")).toBeUndefined();
    expect(sanitizeHttpUrl("https://safe.example.com/article")).toBe("https://safe.example.com/article");
    expect(sanitizeHttpUrl("http://insecure.example.com/news")).toBe("http://insecure.example.com/news");

    // Test in TypewriterMarkdown
    const { container } = render(
      <TypewriterMarkdown content="[Attack Link](javascript:alert(1))" speedMs={0} />
    );

    // Should not create an <a> tag pointing to javascript:
    const link = container.querySelector('a[href^="javascript:"]');
    expect(link).toBeNull();
  });

  it("renders blocked and failed fetches visibly with the real reason", () => {
    const events: ExecutionEvent[] = [
      {
        id: "ev-blocked",
        label: "WEB_FETCH_BLOCKED",
        message: "Blocked: 127.0.0.1",
        source: "SSE",
        rawType: "WEB_FETCH_BLOCKED",
        payload: {
          seq: 1,
          domain: "127.0.0.1",
          reason: "Direct IP access to prohibited address is blocked.",
          status: "blocked",
        },
      },
      {
        id: "ev-failed",
        label: "WEB_FETCH_FAILED",
        message: "Failed to read: timeout.example",
        source: "SSE",
        rawType: "WEB_FETCH_FAILED",
        payload: {
          seq: 2,
          domain: "timeout.example",
          reason: "Connection timed out after 10000ms",
          status: "failed",
        },
      },
    ];

    render(<ResearchActivityFeed events={events} running={true} runId="test-fail-blocked" />);

    expect(screen.getByText(/Blocked: 127.0.0.1/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Direct IP access to prohibited address is blocked/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Failed to read: timeout.example/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Connection timed out after 10000ms/i).length).toBeGreaterThanOrEqual(1);
  });

  it("auto-collapses on completion to one-line summary and expands when clicked", () => {
    const events: ExecutionEvent[] = [
      {
        id: "ev-1",
        label: "WEB_SEARCH_STARTED",
        message: "Searching",
        source: "SSE",
        rawType: "WEB_SEARCH_STARTED",
        payload: { seq: 1, query: "aerodynamics 2026" },
      },
      {
        id: "ev-2",
        label: "WEB_FETCH_COMPLETED",
        message: "Read source",
        source: "SSE",
        rawType: "WEB_FETCH_COMPLETED",
        payload: { seq: 2, url: "https://fia.com/aero", domain: "fia.com", title: "FIA Aero", status: "completed" },
      },
    ];

    render(
      <ResearchActivityFeed
        events={events}
        running={false}
        runId="test-complete"
        finalAnswer="According to [web:fia.com]."
      />
    );

    // Verify summary is rendered
    expect(screen.getByText(/Searched 1 query · read 1 source/i)).toBeInTheDocument();

    // Click to expand full timeline
    const toggleBtn = screen.getByRole("button", { name: /Expand research summary/i });
    fireEvent.click(toggleBtn);

    // Verify timeline and sources are now visible
    expect(screen.getByText(/Activity Timeline/i)).toBeInTheDocument();
    expect(screen.getByText(/Sources \(1\)/i)).toBeInTheDocument();
    expect(screen.getByText(/cited in answer/i)).toBeInTheDocument();
  });

  it("visibly flags unverified citations with unverified badge in TypewriterMarkdown", () => {
    const validCitations = ["[FIA 2025, art 3.4.1, p.24]"];
    const markdownContent = "Verified source [FIA 2025, art 3.4.1, p.24] vs hallucinated [FakeDoc, art 99.9, p.1].";

    render(
      <TypewriterMarkdown
        content={markdownContent}
        speedMs={0}
        validCitations={validCitations}
      />
    );

    // Verify unverified badge is rendered for the hallucinated citation
    expect(screen.getByText(/unverified/i)).toBeInTheDocument();
  });
});
