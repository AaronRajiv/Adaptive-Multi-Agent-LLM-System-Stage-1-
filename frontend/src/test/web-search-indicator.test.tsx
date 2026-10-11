import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { WebSearchIndicator } from "../components/chat/WebSearchIndicator";
import type { ExecutionEvent } from "../types/execution";

describe("WebSearchIndicator Component", () => {
  it("renders nothing if there are no web events", () => {
    const nonWebEvents: ExecutionEvent[] = [
      {
        id: "1",
        label: "PLANNING_STARTED",
        message: "Planning started",
        source: "SSE",
        rawType: "PLANNING_STARTED",
      },
    ];

    const { container } = render(<WebSearchIndicator events={nonWebEvents} running={true} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders active search state when WEB_SEARCH_STARTED is present", () => {
    const events: ExecutionEvent[] = [
      {
        id: "web-1",
        label: "WEB_SEARCH_STARTED",
        message: 'Searching the web for: "2026 F1 engine rules"',
        source: "SSE",
        rawType: "WEB_SEARCH_STARTED",
        payload: { query: "2026 F1 engine rules" },
      },
    ];

    render(<WebSearchIndicator events={events} running={true} />);

    expect(screen.getByText(/Searching the web for/i)).toBeInTheDocument();
    expect(screen.getByText(/"2026 F1 engine rules"/i)).toBeInTheDocument();
    expect(screen.getByText("SEARCHING")).toBeInTheDocument();
  });

  it("renders viewing website state when WEB_FETCH_STARTED is present", () => {
    const events: ExecutionEvent[] = [
      {
        id: "web-1",
        label: "WEB_SEARCH_STARTED",
        message: 'Searching the web for: "FIA 2026 regulations"',
        source: "SSE",
        rawType: "WEB_SEARCH_STARTED",
        payload: { query: "FIA 2026 regulations" },
      },
      {
        id: "web-fetch-1",
        label: "WEB_FETCH_STARTED",
        message: "Viewing website: FIA Official Regulations",
        source: "SSE",
        rawType: "WEB_FETCH_STARTED",
        payload: { url: "https://www.fia.com/regulations/2026", title: "FIA Official Regulations" },
      },
    ];

    render(<WebSearchIndicator events={events} running={true} />);

    expect(screen.getByText(/Viewing/i)).toBeInTheDocument();
    expect(screen.getByText(/FIA Official Regulations/i)).toBeInTheDocument();
  });

  it("renders completed source preview cards when finished", () => {
    const events: ExecutionEvent[] = [
      {
        id: "web-1",
        label: "WEB_SEARCH_STARTED",
        message: 'Searching the web for: "2026 F1 engine rules"',
        source: "SSE",
        rawType: "WEB_SEARCH_STARTED",
        payload: { query: "2026 F1 engine rules" },
      },
      {
        id: "web-comp",
        label: "WEB_SEARCH_COMPLETED",
        message: "Found 2 web results",
        source: "SSE",
        rawType: "WEB_SEARCH_COMPLETED",
        payload: {
          query: "2026 F1 engine rules",
          results: [
            {
              title: "Formula 1 2026 Power Unit Regulations Explained",
              url: "https://www.formula1.com/en/latest/article.2026-rules.html",
              snippet: "The 2026 regulations drop the MGU-H and increase electrical output to 350kW.",
            },
            {
              title: "FIA Technical Regulations 2026",
              url: "https://www.fia.com/olympic-motorsport/2026-tech-regulations",
              snippet: "Overview of the technical framework approved by the World Motor Sport Council.",
            },
          ],
        },
      },
      {
        id: "web-fetch-comp",
        label: "WEB_FETCH_COMPLETED",
        message: "Extracted content from Formula 1 2026 Power Unit",
        source: "SSE",
        rawType: "WEB_FETCH_COMPLETED",
        payload: {
          url: "https://www.formula1.com/en/latest/article.2026-rules.html",
          title: "Formula 1 2026 Power Unit Regulations Explained",
          snippet: "The 2026 regulations drop the MGU-H and increase electrical output to 350kW.",
          content_hash: "a1b2c3d4",
        },
      },
    ];

    render(<WebSearchIndicator events={events} running={false} />);

    // Header says Searched 2 websites
    expect(screen.getByText(/Searched 2 websites/i)).toBeInTheDocument();
    expect(screen.getAllByText(/formula1.com/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/fia.com/i).length).toBeGreaterThanOrEqual(1);

    // Cards are visible
    expect(screen.getByText(/Formula 1 2026 Power Unit Regulations Explained/i)).toBeInTheDocument();
    expect(screen.getByText(/FIA Technical Regulations 2026/i)).toBeInTheDocument();
    expect(screen.getByText(/drop the MGU-H and increase electrical output to 350kW/i)).toBeInTheDocument();

    // Clicking collapse toggles the cards
    const toggleButton = screen.getByRole("button");
    fireEvent.click(toggleButton);

    // Cards are hidden
    expect(screen.queryByText(/drop the MGU-H and increase electrical output to 350kW/i)).toBeNull();

    // Clicking again re-opens
    fireEvent.click(toggleButton);
    expect(screen.getByText(/drop the MGU-H and increase electrical output to 350kW/i)).toBeInTheDocument();
  });
});
