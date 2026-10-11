import { useState, useMemo } from "react";
import {
  Compass,
  ChevronDown,
  ChevronRight,
  Globe,
  FileText,
  Search,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ExternalLink,
  Loader2,
  ShieldAlert,
  Layers,
  Sparkles,
  WifiOff,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ExecutionEvent } from "@/types/execution";
import {
  activityReducer,
  createInitialActivityState,
  sanitizeHttpUrl,
  type ActivitySource,
  type ActivityStep,
  type HonestStatus,
} from "@/services/activityReducer";
import { EvidencePassageModal } from "./EvidencePassageModal";

interface Props {
  events: ExecutionEvent[];
  running: boolean;
  runId?: string;
  finalAnswer?: string;
  statusMessage?: string;
  onCitationClick?: (citation: string) => void;
}

export function ResearchActivityFeed({
  events,
  running,
  runId,
  finalAnswer,
  statusMessage,
  onCitationClick,
}: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState<string | null>(null);

  // Compute activity state from raw events via pure reducer
  const activityState = useMemo(() => {
    let state = createInitialActivityState(runId);
    for (const ev of events) {
      state = activityReducer(state, {
        type: "PROCESS_EVENT",
        event: {
          seq: (ev.payload as any)?.seq ?? (ev as any).seq,
          event_type: ev.rawType || ev.label,
          type: ev.rawType || ev.label,
          task_id: ev.taskId,
          timestamp: ev.time,
          message: ev.message,
          metadata: ev.payload,
        },
      });
    }

    if (statusMessage) {
      state = activityReducer(state, {
        type: "SET_RECONNECTING",
        message: statusMessage,
      });
    }

    if (!running) {
      state = activityReducer(state, {
        type: "FINALIZE_RUN",
        finalAnswer,
      });
    }

    return state;
  }, [events, running, runId, finalAnswer, statusMessage]);

  const { steps, sources, summary, status } = activityState;

  // Don't render empty container if no events and idle
  if (steps.length === 0 && !running && !statusMessage) {
    return null;
  }

  const handleOpenCitation = (citation: string) => {
    setSelectedCitation(citation);
    if (onCitationClick) {
      onCitationClick(citation);
    }
  };

  // Build one-line completion summary string
  const summaryText = useMemo(() => {
    const parts: string[] = [];
    if (summary.queriesCount > 0) {
      parts.push(`Searched ${summary.queriesCount} ${summary.queriesCount === 1 ? "query" : "queries"}`);
    }
    if (summary.sourcesReadCount > 0) {
      parts.push(`read ${summary.sourcesReadCount} ${summary.sourcesReadCount === 1 ? "source" : "sources"}`);
    } else if (sources.length > 0) {
      parts.push(`${sources.length} ${sources.length === 1 ? "source" : "sources"} referenced`);
    }
    if (summary.durationSeconds > 0) {
      parts.push(`${summary.durationSeconds} s`);
    }
    return parts.join(" · ") || "Research completed";
  }, [summary, sources]);

  const isLive = running || status === "reconnecting";

  return (
    <div
      aria-live="polite"
      className="rounded-xl border border-white/10 bg-neutral-950/70 p-3 space-y-3 text-xs text-neutral-200 transition-all w-full max-w-full overflow-hidden"
    >
      {/* Header bar: Live Status or Auto-Collapsed One-Line Summary */}
      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => setIsExpanded((prev) => !prev)}
          className="flex items-center gap-2 text-left hover:text-neutral-100 focus:outline-none focus:ring-1 focus:ring-emerald-400 rounded-md p-1 -m-1 transition-colors flex-1 min-w-0"
          aria-expanded={isLive || isExpanded}
          aria-label={isLive ? "Research activity in progress" : `Expand research summary: ${summaryText}`}
        >
          {isLive ? (
            status === "reconnecting" ? (
              <WifiOff size={15} className="text-amber-400 flex-shrink-0 animate-pulse" />
            ) : (
              <Loader2 size={15} className="text-emerald-400 flex-shrink-0 animate-spin" />
            )
          ) : (
            <Compass size={15} className="text-emerald-400 flex-shrink-0" />
          )}

          <div className="flex items-center gap-2 min-w-0 truncate">
            {isLive ? (
              <span className="font-semibold text-neutral-200 truncate">
                {status === "reconnecting"
                  ? statusMessage || "lost connection, checking run status..."
                  : "Research Activity"}
              </span>
            ) : (
              <span className="font-medium text-neutral-300 truncate">{summaryText}</span>
            )}
          </div>

          <div className="ml-auto flex items-center gap-1.5 flex-shrink-0 text-neutral-400">
            {isLive ? (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] bg-emerald-950/80 text-emerald-400 border border-emerald-500/30">
                live
              </span>
            ) : (
              <span className="text-[11px] text-neutral-400">
                {isExpanded ? (
                  <ChevronDown size={14} className="text-neutral-400" />
                ) : (
                  <ChevronRight size={14} className="text-neutral-400" />
                )}
              </span>
            )}
          </div>
        </button>
      </div>

      {/* Expanded Timeline or Live In-Progress View */}
      {(isLive || isExpanded) && (
        <div className="space-y-3 pt-2 border-t border-white/5">
          {/* Timeline of Real Steps */}
          {steps.length > 0 && (
            <div className="space-y-2">
              <span className="text-[10px] font-semibold tracking-wider text-neutral-400 uppercase">
                Activity Timeline
              </span>
              <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                {steps.map((step, idx) => {
                  const isLatest = idx === steps.length - 1 && isLive;
                  return (
                    <div
                      key={step.id}
                      className={`flex items-start gap-2 p-2 rounded-lg text-[11px] leading-relaxed transition-colors ${
                        isLatest
                          ? "bg-emerald-950/30 border border-emerald-500/40 text-neutral-100"
                          : step.status === "blocked"
                          ? "bg-amber-950/20 border border-amber-500/30 text-amber-200"
                          : step.status === "failed"
                          ? "bg-red-950/20 border border-red-500/30 text-red-200"
                          : "bg-neutral-900/40 border border-white/5 text-neutral-300"
                      }`}
                    >
                      <div className="mt-0.5 flex-shrink-0">
                        {step.status === "blocked" ? (
                          <ShieldAlert size={13} className="text-amber-400" />
                        ) : step.status === "failed" ? (
                          <XCircle size={13} className="text-red-400" />
                        ) : step.status === "running" ? (
                          <Loader2 size={13} className="text-emerald-400 animate-spin" />
                        ) : step.action === "document_retrieval" ? (
                          <FileText size={13} className="text-emerald-400" />
                        ) : step.action === "web_search" ? (
                          <Search size={13} className="text-cyan-400" />
                        ) : step.action === "web_fetch" ? (
                          <Globe size={13} className="text-cyan-400" />
                        ) : (
                          <CheckCircle2 size={13} className="text-neutral-400" />
                        )}
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-1">
                          <span className="font-medium truncate">{step.label}</span>
                          {step.latencyMs !== undefined && (
                            <span className="text-[9px] text-neutral-400 font-mono flex-shrink-0">
                              {Math.round(step.latencyMs)}ms
                            </span>
                          )}
                        </div>

                        {step.detail && (
                          <p className="text-[10px] text-neutral-400 truncate mt-0.5">{step.detail}</p>
                        )}
                        {step.reason && (
                          <p className="text-[10px] text-amber-300/90 mt-0.5 leading-snug">{step.reason}</p>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Sources List */}
          {sources.length > 0 && (
            <div className="space-y-2 pt-1 border-t border-white/5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold tracking-wider text-neutral-400 uppercase">
                  Sources ({sources.length})
                </span>
                <span className="text-[10px] text-neutral-400">Click citation to view verbatim text</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-48 overflow-y-auto pr-1">
                {sources.map((s, idx) => {
                  const safeUrl = sanitizeHttpUrl(s.url);
                  return (
                    <div
                      key={s.id || idx}
                      className="p-2.5 rounded-lg bg-neutral-900/60 border border-white/5 space-y-1.5 flex flex-col justify-between text-[11px]"
                    >
                      <div className="space-y-1 min-w-0">
                        <div className="flex items-center justify-between gap-1">
                          <div className="flex items-center gap-1.5 min-w-0 truncate">
                            {s.sourceType === "web" ? (
                              <Globe size={12} className="text-cyan-400 flex-shrink-0" />
                            ) : (
                              <FileText size={12} className="text-emerald-400 flex-shrink-0" />
                            )}
                            <span className="font-semibold text-neutral-200 truncate" title={s.domain || s.title}>
                              {s.domain || s.title || "Document Source"}
                            </span>
                          </div>

                          {/* Citation Button */}
                          <button
                            type="button"
                            onClick={() => handleOpenCitation(s.citation || s.id)}
                            className="inline-flex items-center px-1.5 py-0.5 rounded bg-white/5 hover:bg-emerald-500/20 text-emerald-300 border border-white/10 text-[9px] font-mono cursor-pointer transition-colors focus:ring-1 focus:ring-emerald-400 flex-shrink-0"
                            title="Click to view stored passage"
                          >
                            {s.citation}
                          </button>
                        </div>

                        {/* Title plain text */}
                        <p className="text-[10px] text-neutral-300 truncate" title={s.title}>
                          {s.title || s.domain}
                        </p>
                      </div>

                      {/* Footer: status & cited in answer */}
                      <div className="flex items-center justify-between pt-1 border-t border-white/5 text-[9px]">
                        <span
                          className={`font-mono ${
                            s.citedInAnswer
                              ? "text-emerald-400 font-semibold"
                              : s.status === "blocked"
                              ? "text-amber-400"
                              : s.status === "failed"
                              ? "text-red-400"
                              : "text-neutral-400"
                          }`}
                        >
                          {s.citedInAnswer
                            ? "cited in answer"
                            : s.status === "blocked"
                            ? "blocked"
                            : s.status === "failed"
                            ? "failed"
                            : "read, not used"}
                        </span>

                        {safeUrl && (
                          <a
                            href={safeUrl}
                            target="_blank"
                            rel="noopener noreferrer nofollow"
                            className="inline-flex items-center gap-0.5 text-cyan-400 hover:text-cyan-300 hover:underline"
                          >
                            link <ExternalLink size={9} />
                          </a>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Stored Evidence Passage Modal */}
      <EvidencePassageModal
        runId={runId}
        evidenceId={selectedCitation}
        isOpen={Boolean(selectedCitation)}
        onClose={() => setSelectedCitation(null)}
      />
    </div>
  );
}
