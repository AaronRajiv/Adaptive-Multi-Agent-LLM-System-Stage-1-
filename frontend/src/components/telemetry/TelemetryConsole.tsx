import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  ArrowUpRight,
  Check,
  ChevronRight,
  Copy,
  Database,
  FileText,
  GitBranch,
  Terminal,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ExecutionEvent, ExecutionModel, ExecutionTask } from "@/types/execution";

export type TelemetryTab = "events" | "node" | "graph" | "rag";

interface Props {
  events: ExecutionEvent[];
  run?: ExecutionModel | undefined;
  selectedTask?: ExecutionTask | undefined;
  activeTab: TelemetryTab;
  setActiveTab: (tab: TelemetryTab) => void;
  running: boolean;
  tasks: ExecutionTask[];
}

export function TelemetryConsole({
  events,
  run,
  selectedTask,
  activeTab,
  setActiveTab,
  running,
  tasks,
}: Props) {
  const [copied, setCopied] = useState(false);
  const eventStreamRef = useRef<HTMLDivElement>(null);

  // Auto scroll events stream
  useEffect(() => {
    if (eventStreamRef.current) {
      eventStreamRef.current.scrollTop = eventStreamRef.current.scrollHeight;
    }
  }, [events.length]);

  return (
    <div className="flex flex-col h-full bg-neutral-950 border border-white/10 rounded-lg overflow-hidden text-neutral-200">
      {/* Tab Navigation Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/10 bg-neutral-900/60">
        <div className="flex items-center gap-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setActiveTab("events")}
            className={`text-xs h-7 px-2.5 rounded ${
              activeTab === "events"
                ? "bg-white/10 text-white font-medium"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <Terminal size={12} className="mr-1.5" /> Live Events ({events.length})
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => setActiveTab("graph")}
            className={`text-xs h-7 px-2.5 rounded ${
              activeTab === "graph"
                ? "bg-white/10 text-white font-medium"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <GitBranch size={12} className="mr-1.5" /> Task Graph ({tasks.length})
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => setActiveTab("node")}
            className={`text-xs h-7 px-2.5 rounded ${
              activeTab === "node"
                ? "bg-white/10 text-white font-medium"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <ArrowUpRight size={12} className="mr-1.5" /> Node Inspector
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => setActiveTab("rag")}
            className={`text-xs h-7 px-2.5 rounded ${
              activeTab === "rag"
                ? "bg-white/10 text-white font-medium"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <Database size={12} className="mr-1.5" /> RAG Provenance
          </Button>
        </div>

        <span className="text-[10px] font-mono text-neutral-400">
          {running ? "STREAMING" : run ? "COMPLETE" : "IDLE"}
        </span>
      </div>

      {/* Console Tab Content */}
      <div className="flex-1 overflow-y-auto p-3 text-xs">
        {/* Tab 1: Live Event Stream */}
        {activeTab === "events" && (
          <div ref={eventStreamRef} className="space-y-1.5 font-mono text-[11px]">
            {events.length > 0 ? (
              events.map((ev) => (
                <div
                  key={ev.id}
                  className="flex items-start gap-2 py-1 px-2 rounded bg-white/[0.02] border border-white/5"
                >
                  <span className="text-neutral-500 text-[10px] flex-shrink-0">
                    {ev.time || "00:00:00"}
                  </span>
                  <span
                    className={`font-semibold text-[10px] uppercase tracking-wider flex-shrink-0 px-1.5 py-0.2 rounded ${
                      ev.status === "FAILED"
                        ? "bg-red-950 text-red-400"
                        : ev.status === "COMPLETED"
                        ? "bg-emerald-950 text-emerald-400"
                        : ev.status === "RUNNING"
                        ? "bg-amber-950 text-amber-400"
                        : "bg-neutral-900 text-neutral-400"
                    }`}
                  >
                    {ev.label.replaceAll("_", " ")}
                  </span>
                  <span className="text-neutral-300 flex-1 truncate">{ev.message}</span>
                  <span className="text-neutral-500 text-[9.5px] flex-shrink-0">
                    {ev.source}
                  </span>
                </div>
              ))
            ) : (
              <div className="text-center py-6 text-neutral-500 italic">
                Awaiting backend event emissions...
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Dynamic Task Graph Inspector */}
        {activeTab === "graph" && (
          <div className="space-y-2">
            {tasks.length > 0 ? (
              tasks.map((t) => (
                <div
                  key={t.id}
                  className="p-2.5 rounded bg-neutral-900/80 border border-white/10 space-y-1"
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
                    {t.dependencies.length > 0 && (
                      <span>Deps: {t.dependencies.join(", ")}</span>
                    )}
                    {t.durationMs !== undefined && (
                      <span>Latency: {(t.durationMs / 1000).toFixed(2)}s</span>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-6 text-neutral-500 italic">
                No dynamic tasks created yet
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Node Inspector */}
        {activeTab === "node" && (
          <div>
            {selectedTask ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-white/10">
                  <h4 className="font-semibold text-neutral-100 text-sm">
                    {selectedTask.title}
                  </h4>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/10 text-emerald-400">
                    {selectedTask.status}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-neutral-400 bg-neutral-900/60 p-2.5 rounded border border-white/5">
                  <div>ID: {selectedTask.id}</div>
                  <div>Capability: {selectedTask.capability || selectedTask.type}</div>
                  <div>Agent: {selectedTask.agent || "N/A"}</div>
                  <div>
                    Latency: {selectedTask.durationMs ? `${selectedTask.durationMs}ms` : "N/A"}
                  </div>
                </div>

                {selectedTask.output && (
                  <div className="space-y-1">
                    <span className="text-[10px] font-mono uppercase text-neutral-400">
                      Task Output
                    </span>
                    <div className="p-3 rounded bg-neutral-900 border border-white/10 text-neutral-200 text-xs">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {selectedTask.output}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-6 text-neutral-500 italic">
                Click any architecture node or task to inspect details
              </div>
            )}
          </div>
        )}

        {/* Tab 4: RAG Provenance */}
        {activeTab === "rag" && (
          <div className="space-y-3">
            {run?.chunks && run.chunks.length > 0 ? (
              run.chunks.map((chunk, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded bg-neutral-900 border border-white/10 space-y-1.5 text-xs"
                >
                  <div className="flex items-center justify-between font-mono text-[10px] text-neutral-400">
                    <span className="text-emerald-400 font-semibold">
                      Source: {chunk.source || chunk.documentId || "Knowledge Base"}
                    </span>
                    {chunk.similarity !== undefined && (
                      <span>Similarity: {chunk.similarity.toFixed(3)}</span>
                    )}
                  </div>
                  <p className="text-neutral-300 leading-normal text-[11px]">
                    {chunk.content}
                  </p>
                </div>
              ))
            ) : (
              <div className="text-center py-6 text-neutral-500 italic">
                RAG retrieval context was not invoked or returned for this task
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
