import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  ArrowUpRight,
  Award,
  CheckCircle2,
  Clock,
  Compass,
  Database,
  DollarSign,
  FileSearch,
  FileText,
  GitBranch,
  Layers,
  RefreshCw,
  ShieldAlert,
  Terminal,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ExecutionEvent, ExecutionModel, ExecutionTask, ProvenanceChunk } from "@/types/execution";

export type TelemetryTab = "events" | "strategy" | "evidence" | "replanning" | "metrics" | "graph" | "node";

interface Props {
  events: ExecutionEvent[];
  run?: ExecutionModel | undefined;
  selectedTask?: ExecutionTask | undefined;
  activeTab: TelemetryTab;
  setActiveTab: (tab: TelemetryTab) => void;
  running: boolean;
  tasks: ExecutionTask[];
  onSelectCitation?: (chunk: ProvenanceChunk) => void;
}

export function TelemetryConsole({
  events,
  run,
  selectedTask,
  activeTab,
  setActiveTab,
  running,
  tasks,
  onSelectCitation,
}: Props) {
  const [selectedChunk, setSelectedChunk] = useState<ProvenanceChunk | null>(null);
  const eventStreamRef = useRef<HTMLDivElement>(null);

  // Auto scroll events stream
  useEffect(() => {
    if (eventStreamRef.current) {
      eventStreamRef.current.scrollTop = eventStreamRef.current.scrollHeight;
    }
  }, [events.length]);

  const strategy = run?.executionStrategy || "MULTI_AGENT";
  const metrics = run?.metrics;
  const verification = run?.verification;
  const evaluation = run?.evaluation;
  const chunks = run?.chunks || [];

  return (
    <div className="flex flex-col h-full bg-neutral-950 border border-white/10 rounded-xl overflow-hidden text-neutral-200">
      {/* Tab Navigation Header */}
      <div className="flex items-center justify-between px-2.5 py-2 border-b border-white/10 bg-neutral-900/70 overflow-x-auto">
        <div className="flex items-center gap-1 min-w-max">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setActiveTab("events")}
            className={`text-xs h-7 px-2.5 rounded-lg ${
              activeTab === "events"
                ? "bg-white/10 text-white font-medium shadow-sm"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <Terminal size={12} className="mr-1.5 text-neutral-400" />
            Events ({events.length})
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => setActiveTab("strategy")}
            className={`text-xs h-7 px-2.5 rounded-lg ${
              activeTab === "strategy"
                ? "bg-white/10 text-white font-medium shadow-sm"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <Compass size={12} className="mr-1.5 text-cyan-400" />
            Strategy
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => setActiveTab("evidence")}
            className={`text-xs h-7 px-2.5 rounded-lg ${
              activeTab === "evidence"
                ? "bg-white/10 text-white font-medium shadow-sm"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <FileSearch size={12} className="mr-1.5 text-emerald-400" />
            Evidence ({chunks.length})
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => setActiveTab("replanning")}
            className={`text-xs h-7 px-2.5 rounded-lg ${
              activeTab === "replanning"
                ? "bg-white/10 text-white font-medium shadow-sm"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <RefreshCw size={12} className="mr-1.5 text-amber-400" />
            Replanning ({run?.replanningCount || 0})
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => setActiveTab("metrics")}
            className={`text-xs h-7 px-2.5 rounded-lg ${
              activeTab === "metrics"
                ? "bg-white/10 text-white font-medium shadow-sm"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <Zap size={12} className="mr-1.5 text-violet-400" />
            Metrics
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => setActiveTab("graph")}
            className={`text-xs h-7 px-2.5 rounded-lg ${
              activeTab === "graph"
                ? "bg-white/10 text-white font-medium shadow-sm"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <GitBranch size={12} className="mr-1.5 text-neutral-400" />
            Graph ({tasks.length})
          </Button>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0 pl-2">
          <span
            className={`text-[9.5px] font-mono px-2 py-0.5 rounded-full ${
              running
                ? "bg-amber-950/80 text-amber-400 border border-amber-500/30 animate-pulse"
                : run
                ? "bg-emerald-950/80 text-emerald-400 border border-emerald-500/30"
                : "bg-neutral-900 text-neutral-400 border border-white/5"
            }`}
          >
            {running ? "LIVE SSE" : run ? "COMPLETED" : "IDLE"}
          </span>
        </div>
      </div>

      {/* Tab Contents */}
      <div className="flex-1 overflow-y-auto p-3 text-xs">
        {/* 1. Live Events Stream */}
        {activeTab === "events" && (
          <div ref={eventStreamRef} className="space-y-1.5 font-mono text-[11px]">
            {events.length > 0 ? (
              events.map((ev) => (
                <div
                  key={ev.id}
                  className="flex items-start gap-2 py-1.5 px-2 rounded-lg bg-white/[0.02] border border-white/5 hover:border-white/10 transition-colors"
                >
                  <span className="text-neutral-500 text-[10px] flex-shrink-0">
                    {ev.time || "00:00:00"}
                  </span>
                  <span
                    className={`font-semibold text-[9.5px] uppercase tracking-wider flex-shrink-0 px-1.5 py-0.5 rounded ${
                      ev.status === "FAILED"
                        ? "bg-red-950 text-red-400 border border-red-500/30"
                        : ev.status === "COMPLETED"
                        ? "bg-emerald-950 text-emerald-400 border border-emerald-500/30"
                        : ev.status === "RUNNING"
                        ? "bg-amber-950 text-amber-400 border border-amber-500/30"
                        : "bg-neutral-900 text-neutral-400"
                    }`}
                  >
                    {ev.label.replaceAll("_", " ")}
                  </span>
                  <span className="text-neutral-300 flex-1 truncate">{ev.message}</span>
                  <span className="text-neutral-500 text-[9px] flex-shrink-0">
                    {ev.source}
                  </span>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-neutral-500 italic">
                Awaiting real backend event emissions...
              </div>
            )}
          </div>
        )}

        {/* 2. Strategy Decision Display */}
        {activeTab === "strategy" && (
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-neutral-900/60 border border-white/10 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-mono text-neutral-400 uppercase">
                  Adaptive Strategy Selector
                </span>
                <span
                  className={`px-2.5 py-1 rounded-md text-xs font-bold font-mono ${
                    strategy === "DIRECT"
                      ? "bg-blue-950 text-blue-300 border border-blue-500/40"
                      : strategy === "SINGLE_AGENT"
                      ? "bg-cyan-950 text-cyan-300 border border-cyan-500/40"
                      : strategy === "SINGLE_AGENT_RAG"
                      ? "bg-emerald-950 text-emerald-300 border border-emerald-500/40"
                      : "bg-purple-950 text-purple-300 border border-purple-500/40"
                  }`}
                >
                  {strategy}
                </span>
              </div>

              <p className="text-xs text-neutral-300 leading-relaxed">
                {strategy === "DIRECT"
                  ? "Single-turn factual question or conversational greeting routed directly to synthesizer with 0-agent overhead."
                  : strategy === "SINGLE_AGENT"
                  ? "Single domain reasoning task executed by an isolated specialist agent with minimal coordination budget."
                  : strategy === "SINGLE_AGENT_RAG"
                  ? "Document lookup requirement grounded via targeted hybrid BM25 + dense vector retrieval."
                  : "Complex multi-domain analytical challenge decomposed into a bounded DAG of parallel specialist agents."}
              </p>

              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-white/5 font-mono text-[10px]">
                <div className="p-2 rounded bg-black/40 border border-white/5 space-y-1">
                  <span className="text-neutral-500">MAX LLM CALLS</span>
                  <div className="text-white text-xs font-semibold">
                    {strategy === "DIRECT" ? "1" : strategy === "SINGLE_AGENT" ? "3" : strategy === "SINGLE_AGENT_RAG" ? "4" : "20"}
                  </div>
                </div>
                <div className="p-2 rounded bg-black/40 border border-white/5 space-y-1">
                  <span className="text-neutral-500">TOKEN BUDGET</span>
                  <div className="text-white text-xs font-semibold">
                    {strategy === "DIRECT" ? "3,000" : strategy === "SINGLE_AGENT" ? "12,000" : strategy === "SINGLE_AGENT_RAG" ? "18,000" : "80,000"}
                  </div>
                </div>
                <div className="p-2 rounded bg-black/40 border border-white/5 space-y-1">
                  <span className="text-neutral-500">MAX WALL TIME</span>
                  <div className="text-white text-xs font-semibold">
                    {strategy === "DIRECT" ? "15s" : strategy === "SINGLE_AGENT" ? "45s" : strategy === "SINGLE_AGENT_RAG" ? "60s" : "240s"}
                  </div>
                </div>
              </div>
            </div>

            {/* Strategy Comparison Breakdown */}
            <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 space-y-2">
              <span className="text-[10px] font-mono text-neutral-400 uppercase">
                Observed Resource Consumption
              </span>
              <div className="flex items-center justify-between text-xs text-neutral-300 py-1 border-b border-white/5">
                <span>Actual LLM Calls</span>
                <span className="font-mono text-white font-semibold">{metrics?.totalLlmCalls ?? (run ? 1 : 0)}</span>
              </div>
              <div className="flex items-center justify-between text-xs text-neutral-300 py-1 border-b border-white/5">
                <span>Total Tokens Used</span>
                <span className="font-mono text-white font-semibold">{metrics?.totalTokens?.toLocaleString() ?? "N/A"}</span>
              </div>
              <div className="flex items-center justify-between text-xs text-neutral-300 py-1">
                <span>Execution Wall Time</span>
                <span className="font-mono text-white font-semibold">
                  {metrics?.wallTimeMs ? `${(metrics.wallTimeMs / 1000).toFixed(2)}s` : "N/A"}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* 3. Evidence Panel & Citation Inspector */}
        {activeTab === "evidence" && (
          <div className="space-y-3">
            {verification && (
              <div className="p-3 rounded-xl bg-emerald-950/20 border border-emerald-500/20 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle2 size={16} className="text-emerald-400" />
                  <div>
                    <span className="font-semibold text-neutral-200 text-xs">Grounding Verifier</span>
                    <p className="text-[10px] text-neutral-400">
                      Programmatic citation audit across retrieved chunks
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-emerald-400 font-mono font-bold text-sm">
                    {verification.citationValidityPercent.toFixed(1)}%
                  </div>
                  <span className="text-[9.5px] font-mono text-neutral-500">
                    {verification.validCitationCount}/{verification.citationCount} valid
                  </span>
                </div>
              </div>
            )}

            {chunks.length > 0 ? (
              chunks.map((chunk, idx) => {
                const isSelected = selectedChunk?.id === chunk.id || selectedChunk?.chunkIndex === chunk.chunkIndex;
                return (
                  <div
                    key={chunk.id || idx}
                    onClick={() => {
                      setSelectedChunk(chunk);
                      onSelectCitation?.(chunk);
                    }}
                    className={`p-3 rounded-xl border transition-all cursor-pointer space-y-2 ${
                      isSelected
                        ? "bg-neutral-900 border-emerald-500/50 shadow-md"
                        : "bg-neutral-900/50 border-white/5 hover:border-white/10"
                    }`}
                  >
                    <div className="flex items-center justify-between font-mono text-[10px]">
                      <span className="text-emerald-400 font-semibold truncate max-w-[200px]">
                        {chunk.id || `chunk-${idx}`}
                      </span>
                      {chunk.similarity !== undefined && (
                        <span className="text-neutral-400">
                          score: {chunk.similarity.toFixed(3)}
                        </span>
                      )}
                    </div>

                    <div className="flex flex-wrap items-center gap-2 text-[9.5px] font-mono text-neutral-400">
                      {chunk.articlePath && (
                        <span className="px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-cyan-300">
                          Article {chunk.articlePath}
                        </span>
                      )}
                      {chunk.heading && (
                        <span className="px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-neutral-300">
                          {chunk.heading}
                        </span>
                      )}
                      {chunk.pageStart && (
                        <span className="text-neutral-500">
                          p.{chunk.pageStart}{chunk.pageEnd && chunk.pageEnd !== chunk.pageStart ? `-${chunk.pageEnd}` : ""}
                        </span>
                      )}
                      {chunk.retrievalMethod && (
                        <span className="text-neutral-500">
                          method: {chunk.retrievalMethod}
                        </span>
                      )}
                    </div>

                    <p className="text-neutral-300 text-[11px] leading-relaxed line-clamp-3">
                      {chunk.content}
                    </p>
                  </div>
                );
              })
            ) : (
              <div className="text-center py-8 text-neutral-500 italic">
                No RAG evidence chunks retrieved for this run.
              </div>
            )}
          </div>
        )}

        {/* 4. Replanning Timeline */}
        {activeTab === "replanning" && (
          <div className="space-y-4">
            {evaluation ? (
              <div className="space-y-3">
                <div className="p-4 rounded-xl bg-neutral-900/60 border border-white/10 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Award size={16} className="text-amber-400" />
                      <span className="font-semibold text-neutral-200 text-xs">Evaluator Agent Score</span>
                    </div>
                    <span
                      className={`px-2 py-0.5 rounded font-mono font-bold text-xs ${
                        evaluation.status === "PASS"
                          ? "bg-emerald-950 text-emerald-400 border border-emerald-500/30"
                          : "bg-red-950 text-red-400 border border-red-500/30"
                      }`}
                    >
                      {evaluation.status} ({evaluation.score}/100)
                    </span>
                  </div>

                  {/* 4 Rubric Dimension Cards */}
                  <div className="grid grid-cols-2 gap-2 font-mono text-[10px]">
                    <div className="p-2 rounded bg-black/40 border border-white/5 flex justify-between">
                      <span className="text-neutral-400">Groundedness:</span>
                      <span className="text-white font-semibold">{evaluation.groundednessScore ?? "N/A"}/25</span>
                    </div>
                    <div className="p-2 rounded bg-black/40 border border-white/5 flex justify-between">
                      <span className="text-neutral-400">Coverage:</span>
                      <span className="text-white font-semibold">{evaluation.coverageScore ?? "N/A"}/25</span>
                    </div>
                    <div className="p-2 rounded bg-black/40 border border-white/5 flex justify-between">
                      <span className="text-neutral-400">Distinction:</span>
                      <span className="text-white font-semibold">{evaluation.distinctionScore ?? "N/A"}/25</span>
                    </div>
                    <div className="p-2 rounded bg-black/40 border border-white/5 flex justify-between">
                      <span className="text-neutral-400">Honesty:</span>
                      <span className="text-white font-semibold">{evaluation.honestyScore ?? "N/A"}/25</span>
                    </div>
                  </div>

                  {evaluation.feedback && (
                    <div className="p-2.5 rounded bg-black/30 border border-white/5 space-y-1">
                      <span className="text-[10px] font-mono text-neutral-500 uppercase">Constructive Critique</span>
                      <p className="text-neutral-300 text-xs leading-normal">{evaluation.feedback}</p>
                    </div>
                  )}
                </div>

                <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 space-y-2">
                  <span className="text-[10px] font-mono text-neutral-400 uppercase">
                    Targeted Replanning Rounds ({run?.replanningCount || 0})
                  </span>
                  {run?.replanningCount && run.replanningCount > 0 ? (
                    <div className="space-y-2">
                      {Array.from({ length: run.replanningCount }).map((_, i) => (
                        <div key={i} className="flex items-center gap-2 p-2 rounded bg-neutral-900/60 border border-white/5 text-xs">
                          <span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center font-mono text-[10px]">
                            {i + 1}
                          </span>
                          <span className="text-neutral-300">Replanning iteration {i + 1} targeted missing dimensions</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-neutral-500 text-xs italic">
                      Initial evaluation passed the quality threshold without requiring additional replanning rounds.
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-neutral-500 italic">
                No evaluation trace available for this task.
              </div>
            )}
          </div>
        )}

        {/* 5. Metrics Panel */}
        {activeTab === "metrics" && (
          <div className="space-y-3">
            {metrics ? (
              <div className="space-y-3">
                {/* Metric Summary Cards */}
                <div className="grid grid-cols-2 gap-2 font-mono">
                  <div className="p-3 rounded-xl bg-neutral-900/60 border border-white/10 space-y-1">
                    <span className="text-[10px] text-neutral-400">TOTAL TOKENS</span>
                    <div className="text-lg font-bold text-white">
                      {metrics.totalTokens.toLocaleString()}
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-neutral-900/60 border border-white/10 space-y-1">
                    <span className="text-[10px] text-neutral-400">ESTIMATED COST</span>
                    <div className="text-lg font-bold text-emerald-400">
                      ${metrics.totalEstimatedCost.toFixed(5)}
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-neutral-900/60 border border-white/10 space-y-1">
                    <span className="text-[10px] text-neutral-400">WALL LATENCY</span>
                    <div className="text-lg font-bold text-cyan-400">
                      {(metrics.wallTimeMs / 1000).toFixed(2)}s
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-neutral-900/60 border border-white/10 space-y-1">
                    <span className="text-[10px] text-neutral-400">TOTAL LLM CALLS</span>
                    <div className="text-lg font-bold text-violet-400">
                      {metrics.totalLlmCalls}
                    </div>
                  </div>
                </div>

                {/* Latency Waterfall / Call Breakdown */}
                {metrics.calls && metrics.calls.length > 0 && (
                  <div className="p-3 rounded-xl bg-black/40 border border-white/10 space-y-2">
                    <span className="text-[10px] font-mono text-neutral-400 uppercase">
                      Call Breakdown Waterfall ({metrics.calls.length})
                    </span>
                    <div className="space-y-1.5 max-h-48 overflow-y-auto">
                      {metrics.calls.map((call, idx) => (
                        <div
                          key={idx}
                          className="flex items-center justify-between p-2 rounded bg-neutral-900/70 border border-white/5 text-[11px] font-mono"
                        >
                          <div className="space-y-0.5">
                            <span className="font-semibold text-neutral-200">{call.stage}</span>
                            <div className="text-[9.5px] text-neutral-500">{call.model}</div>
                          </div>
                          <div className="text-right space-y-0.5">
                            <div className="text-neutral-300 font-semibold">{call.latencyMs.toFixed(0)}ms</div>
                            <div className="text-[9.5px] text-neutral-500">{call.totalTokens} tok</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-neutral-500 italic">
                Run execution telemetry metrics not yet collected or reported.
              </div>
            )}
          </div>
        )}

        {/* 6. Dynamic Task Graph */}
        {activeTab === "graph" && (
          <div className="space-y-2">
            {tasks.length > 0 ? (
              tasks.map((t) => (
                <div
                  key={t.id}
                  className="p-2.5 rounded-lg bg-neutral-900/80 border border-white/10 space-y-1"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-neutral-200 text-xs">{t.title}</span>
                    <span
                      className={`text-[9.5px] font-mono px-1.5 py-0.5 rounded ${
                        t.status === "COMPLETED"
                          ? "bg-emerald-950 text-emerald-400 border border-emerald-500/20"
                          : t.status === "RUNNING"
                          ? "bg-amber-950 text-amber-400 border border-amber-500/20"
                          : t.status === "FAILED"
                          ? "bg-red-950 text-red-400 border border-red-500/20"
                          : "bg-neutral-800 text-neutral-400"
                      }`}
                    >
                      {t.status}
                    </span>
                  </div>
                  <p className="text-neutral-400 text-[11px]">{t.description}</p>
                  <div className="flex items-center gap-3 text-[10px] font-mono text-neutral-500 pt-1">
                    <span>ID: {t.id}</span>
                    <span>Cap: {t.capability || t.type}</span>
                    {t.dependencies.length > 0 && <span>Deps: {t.dependencies.join(", ")}</span>}
                    {t.durationMs !== undefined && <span>Latency: {(t.durationMs / 1000).toFixed(2)}s</span>}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-neutral-500 italic">
                No dynamic tasks created yet
              </div>
            )}
          </div>
        )}

        {/* 7. Node Inspector */}
        {activeTab === "node" && (
          <div>
            {selectedTask ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-white/10">
                  <h4 className="font-semibold text-neutral-100 text-sm">{selectedTask.title}</h4>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/10 text-emerald-400">
                    {selectedTask.status}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-neutral-400 bg-neutral-900/60 p-2.5 rounded-lg border border-white/5">
                  <div>ID: {selectedTask.id}</div>
                  <div>Capability: {selectedTask.capability || selectedTask.type}</div>
                  <div>Agent: {selectedTask.agent || "N/A"}</div>
                  <div>Latency: {selectedTask.durationMs ? `${selectedTask.durationMs}ms` : "N/A"}</div>
                </div>

                {selectedTask.output && (
                  <div className="space-y-1">
                    <span className="text-[10px] font-mono uppercase text-neutral-400">Task Output</span>
                    <div className="p-3 rounded-lg bg-neutral-900 border border-white/10 text-neutral-200 text-xs">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {selectedTask.output}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-neutral-500 italic">
                Click any architecture node or task to inspect details
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
