/**
 * Pure reducer and state models for the live Research Activity Feed.
 *
 * Requirements:
 * - Deduplicate by event seq (and event_id fallback)
 * - Tolerate out-of-order events
 * - Resume via after_seq
 * - Group parallel subtasks by task_id
 * - Track honest states: running, done, failed, blocked, skipped, cancelled, reconnecting
 * - Maintain unified Sources list (domain, title, status, cited in answer)
 */

export type HonestStatus =
  | "running"
  | "done"
  | "failed"
  | "blocked"
  | "skipped"
  | "cancelled"
  | "reconnecting";

export interface ActivitySource {
  id: string; // hash8 or chunkId
  citation: string; // [web:xyz] or [doc, art 3.4, p.24]
  sourceType: "web" | "document";
  url?: string;
  domain?: string;
  title?: string;
  articlePath?: string;
  pageStart?: number;
  pageEnd?: number;
  status: "completed" | "failed" | "blocked" | "read";
  reason?: string;
  bytes?: number;
  latencyMs?: number;
  timestamp?: string;
  citedInAnswer?: boolean;
}

export interface ActivityStep {
  id: string;
  seq: number;
  taskId?: string;
  action:
    | "strategy_selected"
    | "document_retrieval"
    | "web_search"
    | "web_fetch"
    | "task_execution"
    | "evaluation"
    | "replanning"
    | "synthesis"
    | "general";
  label: string;
  detail?: string;
  status: HonestStatus;
  timestamp: string;
  latencyMs?: number;
  reason?: string;
  sources?: ActivitySource[];
  metadata?: Record<string, unknown>;
}

export interface TaskGroup {
  taskId: string;
  name: string;
  status: HonestStatus;
  steps: ActivityStep[];
  startedAt?: string;
  completedAt?: string;
}

export interface ResearchActivityState {
  runId: string;
  status: HonestStatus;
  statusMessage?: string;
  lastSeq: number;
  seenSeqs: number[]; // stored as array for serialization
  steps: ActivityStep[];
  taskGroups: Record<string, TaskGroup>;
  sources: ActivitySource[];
  summary: {
    queriesCount: number;
    sourcesReadCount: number;
    sourcesBlockedCount: number;
    sourcesFailedCount: number;
    durationSeconds: number;
    startTimeMs?: number;
    endTimeMs?: number;
  };
}

export interface RawEventPayload {
  seq?: number;
  event_id?: string;
  id?: string;
  run_id?: string;
  type?: string;
  event_type?: string;
  rawType?: string;
  label?: string;
  timestamp?: string;
  time?: string;
  task_id?: string;
  taskId?: string;
  capability?: string;
  message?: string;
  metadata?: Record<string, any>;
  payload?: Record<string, any>;
  status?: string;
}

export function createInitialActivityState(runId = ""): ResearchActivityState {
  return {
    runId,
    status: "running",
    lastSeq: 0,
    seenSeqs: [],
    steps: [],
    taskGroups: {},
    sources: [],
    summary: {
      queriesCount: 0,
      sourcesReadCount: 0,
      sourcesBlockedCount: 0,
      sourcesFailedCount: 0,
      durationSeconds: 0,
    },
  };
}

/** Sanitize untrusted web URLs: enforce http/https only */
export function sanitizeHttpUrl(url?: string): string | undefined {
  if (!url) return undefined;
  const trimmed = url.trim();
  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed;
  }
  return undefined;
}

/** Extract clean hostname without www */
export function extractCleanDomain(url?: string): string {
  if (!url) return "unknown";
  try {
    const parsed = new URL(url);
    return parsed.hostname.replace(/^www\./i, "");
  } catch {
    return url.replace(/^https?:\/\//i, "").split("/")[0].replace(/^www\./i, "");
  }
}

/** Pure reducer function */
export function activityReducer(
  state: ResearchActivityState,
  action: { type: "PROCESS_EVENT"; event: RawEventPayload } | { type: "SET_RECONNECTING"; message: string } | { type: "FINALIZE_RUN"; finalAnswer?: string; status?: HonestStatus }
): ResearchActivityState {
  if (action.type === "SET_RECONNECTING") {
    return {
      ...state,
      status: "reconnecting",
      statusMessage: action.message,
    };
  }

  if (action.type === "FINALIZE_RUN") {
    const finalStatus = action.status || (state.status === "running" ? "done" : state.status);
    const now = Date.now();
    const endTime = state.summary.endTimeMs || now;
    const startTime = state.summary.startTimeMs || now;
    const durSec = Math.max(0, Math.round((endTime - startTime) / 1000));

    // Detect citations cited in the final answer
    const answer = action.finalAnswer || "";
    const updatedSources = state.sources.map((s) => {
      let isCited = false;
      if (s.citation && answer.includes(s.citation)) {
        isCited = true;
      } else if (s.id && answer.includes(s.id)) {
        isCited = true;
      }
      return { ...s, citedInAnswer: isCited };
    });

    return {
      ...state,
      status: finalStatus,
      statusMessage: undefined,
      sources: updatedSources,
      summary: {
        ...state.summary,
        endTimeMs: endTime,
        durationSeconds: durSec,
      },
    };
  }

  const raw = action.event;
  const seq = typeof raw.seq === "number" ? raw.seq : undefined;

  // Deduplication by seq
  if (seq !== undefined && state.seenSeqs.includes(seq)) {
    return state;
  }

  const eventType = (raw.type || raw.event_type || raw.rawType || raw.label || "").toUpperCase();
  const meta = { ...(raw.payload || {}), ...(raw.metadata || {}) };
  const taskId = raw.task_id || raw.taskId;
  const timestamp = raw.timestamp || raw.time || new Date().toISOString();
  const eventTimeMs = new Date(timestamp).getTime() || Date.now();

  const newSeenSeqs = seq !== undefined ? [...state.seenSeqs, seq] : state.seenSeqs;
  const nextLastSeq = seq !== undefined ? Math.max(state.lastSeq, seq) : state.lastSeq;

  // Compute timings
  const startTimeMs = state.summary.startTimeMs ?? eventTimeMs;
  const endTimeMs = eventTimeMs;
  const durationSeconds = Math.max(0, Math.round((endTimeMs - startTimeMs) / 1000));

  let newStatus: HonestStatus = state.status === "reconnecting" ? "running" : state.status;
  let queriesInc = 0;
  let sourcesReadInc = 0;
  let sourcesBlockedInc = 0;
  let sourcesFailedInc = 0;

  const newSources: ActivitySource[] = [...state.sources];
  let createdStep: ActivityStep | null = null;

  switch (eventType) {
    case "STRATEGY_SELECTED":
    case "STRATEGY_DECIDED": {
      const strat = meta.strategy || "MULTI_AGENT";
      const needsWeb = Boolean(meta.needs_web);
      const rationale = meta.rationale || meta.reasoning || meta.web_rationale || "";
      createdStep = {
        id: `strat-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "strategy_selected",
        label: `Strategy: ${strat}`,
        detail: needsWeb ? `Web search enabled: ${rationale}` : `Document retrieval: ${rationale}`,
        status: "done",
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "RAG_STARTED": {
      const queries = Array.isArray(meta.queries) ? meta.queries : [];
      const qSummary = queries.length > 0 ? queries[0] : "knowledge context";
      createdStep = {
        id: `rag-start-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "document_retrieval",
        label: `Retrieving documents: ${qSummary}`,
        status: "running",
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "RAG_COMPLETED": {
      const nChunks = meta.n_chunks ?? 0;
      const articles = Array.isArray(meta.articles) && meta.articles.length > 0 ? `Art ${meta.articles.join(", ")}` : "";
      const pages = Array.isArray(meta.pages) && meta.pages.length > 0 ? `p.${meta.pages.join(", ")}` : "";
      const details = [articles, pages].filter(Boolean).join(", ");
      createdStep = {
        id: `rag-done-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "document_retrieval",
        label: `Retrieved ${nChunks} chunks${details ? ` (${details})` : ""}`,
        status: "done",
        timestamp,
        metadata: meta,
      };

      // Add document source items
      if (Array.isArray(meta.articles)) {
        for (const art of meta.articles) {
          const cit = `[doc, art ${art}]`;
          if (!newSources.some((s) => s.citation === cit)) {
            newSources.push({
              id: `doc-${art}`,
              citation: cit,
              sourceType: "document",
              title: `Article ${art}`,
              articlePath: String(art),
              status: "completed",
              timestamp,
            });
          }
        }
      }
      break;
    }

    case "WEB_SEARCH_STARTED": {
      queriesInc = 1;
      const q = meta.query || raw.message?.replace(/^Searching the web for:\s*"?/, "").replace(/"?$/, "") || "web search";
      createdStep = {
        id: `search-start-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "web_search",
        label: `Searching the web: "${q}"`,
        status: "running",
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "WEB_SEARCH_COMPLETED": {
      const q = meta.query || "web search";
      const urls = Array.isArray(meta.urls) ? meta.urls : [];
      createdStep = {
        id: `search-done-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "web_search",
        label: `Found ${urls.length} search results for "${q}"`,
        status: "done",
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "WEB_FETCH_STARTED": {
      const rawUrl = meta.url || "";
      const safeUrl = sanitizeHttpUrl(rawUrl);
      const domain = meta.domain || extractCleanDomain(safeUrl || rawUrl);
      const title = meta.title || "";
      createdStep = {
        id: `fetch-start-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "web_fetch",
        label: `Reading ${domain}${title ? ` - ${title}` : ""}`,
        status: "running",
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "WEB_FETCH_COMPLETED": {
      sourcesReadInc = 1;
      const rawUrl = meta.url || "";
      const safeUrl = sanitizeHttpUrl(rawUrl);
      const domain = meta.domain || extractCleanDomain(safeUrl || rawUrl);
      const title = meta.title || domain;
      const latencyMs = meta.latency_ms;
      const hash8 = meta.content_hash || "";
      const citation = hash8 ? `[web:${hash8}]` : `[web:${domain}]`;

      createdStep = {
        id: `fetch-done-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "web_fetch",
        label: `Read ${domain} - ${title}`,
        status: "done",
        timestamp,
        latencyMs,
        metadata: meta,
      };

      const existingIdx = newSources.findIndex((s) => (safeUrl && s.url === safeUrl) || s.citation === citation);
      const sourceItem: ActivitySource = {
        id: hash8 || domain,
        citation,
        sourceType: "web",
        url: safeUrl,
        domain,
        title,
        status: "completed",
        bytes: meta.bytes,
        latencyMs,
        timestamp,
      };
      if (existingIdx >= 0) {
        newSources[existingIdx] = { ...newSources[existingIdx], ...sourceItem };
      } else {
        newSources.push(sourceItem);
      }
      break;
    }

    case "WEB_FETCH_BLOCKED": {
      sourcesBlockedInc = 1;
      const rawUrl = meta.url || "";
      const safeUrl = sanitizeHttpUrl(rawUrl);
      const domain = meta.domain || extractCleanDomain(safeUrl || rawUrl);
      const reason = meta.reason || "Security restriction / Robots disallowed";
      createdStep = {
        id: `fetch-blocked-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "web_fetch",
        label: `Blocked: ${domain}`,
        detail: reason,
        status: "blocked",
        reason,
        timestamp,
        metadata: meta,
      };

      newSources.push({
        id: `blocked-${domain}-${Date.now()}`,
        citation: `[blocked:${domain}]`,
        sourceType: "web",
        url: safeUrl,
        domain,
        title: domain,
        status: "blocked",
        reason,
        timestamp,
      });
      break;
    }

    case "WEB_FETCH_FAILED": {
      sourcesFailedInc = 1;
      const rawUrl = meta.url || "";
      const safeUrl = sanitizeHttpUrl(rawUrl);
      const domain = meta.domain || extractCleanDomain(safeUrl || rawUrl);
      const reason = meta.reason || meta.error || "Connection failure";
      createdStep = {
        id: `fetch-failed-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "web_fetch",
        label: `Failed to read: ${domain}`,
        detail: reason,
        status: "failed",
        reason,
        timestamp,
        metadata: meta,
      };

      newSources.push({
        id: `failed-${domain}-${Date.now()}`,
        citation: `[failed:${domain}]`,
        sourceType: "web",
        url: safeUrl,
        domain,
        title: domain,
        status: "failed",
        reason,
        timestamp,
      });
      break;
    }

    case "TASK_STARTED": {
      createdStep = {
        id: `task-start-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "task_execution",
        label: `Task ${taskId || ""}: started`,
        status: "running",
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "TASK_COMPLETED": {
      createdStep = {
        id: `task-done-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "task_execution",
        label: `Task ${taskId || ""}: completed`,
        status: "done",
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "EVALUATION_STARTED": {
      createdStep = {
        id: `eval-start-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "evaluation",
        label: "Evaluating grounding & coverage...",
        status: "running",
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "EVALUATION_COMPLETED": {
      const score = meta.score ?? meta.total_score;
      const evalStatus = meta.status || (score !== undefined ? (score >= 70 ? "PASS" : "FAIL") : "COMPLETED");
      createdStep = {
        id: `eval-done-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "evaluation",
        label: `Evaluation ${evalStatus}${score !== undefined ? ` (Score: ${score}/100)` : ""}`,
        status: evalStatus === "FAIL" ? "failed" : "done",
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "REPLANNING_TRIGGERED":
    case "REPLANNING_STARTED": {
      createdStep = {
        id: `replan-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "replanning",
        label: `Replanning: addressing uncovered domains`,
        status: "running",
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "RUN_COMPLETED": {
      newStatus = "done";
      createdStep = {
        id: `run-done-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "general",
        label: "Research synthesis complete",
        status: "done",
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "RUN_FAILED": {
      newStatus = "failed";
      const err = meta.error || raw.message || "Pipeline execution failed";
      createdStep = {
        id: `run-failed-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "general",
        label: `Execution failed: ${err}`,
        status: "failed",
        reason: err,
        timestamp,
        metadata: meta,
      };
      break;
    }

    case "RUN_CANCELLED": {
      newStatus = "cancelled";
      createdStep = {
        id: `run-cancelled-${seq || Date.now()}`,
        seq: seq ?? nextLastSeq,
        taskId,
        action: "general",
        label: "Execution cancelled by user",
        status: "cancelled",
        timestamp,
        metadata: meta,
      };
      break;
    }

    default:
      if (raw.message) {
        createdStep = {
          id: `gen-${seq || Date.now()}`,
          seq: seq ?? nextLastSeq,
          taskId,
          action: "general",
          label: raw.message,
          status: "done",
          timestamp,
          metadata: meta,
        };
      }
      break;
  }

  // Update steps with out-of-order tolerance (sorted by seq)
  let nextSteps = [...state.steps];
  if (createdStep) {
    nextSteps.push(createdStep);
    nextSteps.sort((a, b) => (a.seq - b.seq) || a.timestamp.localeCompare(b.timestamp));
  }

  // Update parallel task groups if taskId is present
  const nextTaskGroups = { ...state.taskGroups };
  if (taskId && createdStep) {
    const existingGroup = nextTaskGroups[taskId] || {
      taskId,
      name: `Task ${taskId}`,
      status: "running" as HonestStatus,
      steps: [],
      startedAt: timestamp,
    };
    const groupStatus =
      createdStep.status === "failed"
        ? "failed"
        : createdStep.status === "blocked"
        ? "blocked"
        : createdStep.action === "task_execution" && createdStep.status === "done"
        ? "done"
        : existingGroup.status;

    nextTaskGroups[taskId] = {
      ...existingGroup,
      status: groupStatus,
      steps: [...existingGroup.steps, createdStep],
      completedAt: createdStep.status === "done" ? timestamp : existingGroup.completedAt,
    };
  }

  return {
    ...state,
    status: newStatus,
    statusMessage: undefined,
    lastSeq: nextLastSeq,
    seenSeqs: newSeenSeqs,
    steps: nextSteps,
    taskGroups: nextTaskGroups,
    sources: newSources,
    summary: {
      queriesCount: state.summary.queriesCount + queriesInc,
      sourcesReadCount: state.summary.sourcesReadCount + sourcesReadInc,
      sourcesBlockedCount: state.summary.sourcesBlockedCount + sourcesBlockedInc,
      sourcesFailedCount: state.summary.sourcesFailedCount + sourcesFailedInc,
      durationSeconds,
      startTimeMs,
      endTimeMs,
    },
  };
}
