import { useState } from "react";
import { Brain, ChevronDown, ChevronRight, Sparkles } from "lucide-react";
import type { ExecutionEvent } from "@/types/execution";

interface Props {
  events: ExecutionEvent[];
  running: boolean;
}

export function ThinkingAccordion({ events, running }: Props) {
  const [isOpen, setIsOpen] = useState(true);

  // Extract reasoning steps from events
  const thinkingEvents = events.filter(
    (ev) =>
      ev.rawType === "PLANNING_STARTED" ||
      ev.rawType === "PLANNING_COMPLETED" ||
      ev.rawType === "TASK_STARTED" ||
      ev.rawType === "TASK_COMPLETED" ||
      ev.rawType === "RAG_STARTED" ||
      ev.rawType === "RAG_COMPLETED" ||
      ev.rawType === "EVALUATION_STARTED" ||
      ev.rawType === "EVALUATION_COMPLETED" ||
      ev.rawType === "SYNTHESIS_STARTED"
  );

  if (thinkingEvents.length === 0 && !running) return null;

  return (
    <div className="rounded-xl border border-white/10 bg-neutral-950/80 overflow-hidden shadow-sm transition-all my-2">
      {/* Accordion Header Bar */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 bg-neutral-900/60 hover:bg-neutral-900 text-left transition-colors"
      >
        <div className="flex items-center gap-2 font-mono text-xs text-neutral-300">
          <Brain size={14} className={running ? "text-emerald-400 animate-pulse" : "text-neutral-400"} />
          <span className="font-medium">
            {running ? "Thinking & Reasoning..." : `Thought Process (${thinkingEvents.length} steps)`}
          </span>
        </div>
        <div className="flex items-center gap-2 text-neutral-400">
          {running && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 animate-pulse">
              REASONING
            </span>
          )}
          {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </div>
      </button>

      {/* Thinking Steps Feed */}
      {isOpen && (
        <div className="p-3.5 border-t border-white/5 space-y-2 font-mono text-[11px] bg-black/40">
          {thinkingEvents.length > 0 ? (
            thinkingEvents.map((ev, index) => (
              <div key={ev.id || index} className="flex items-start gap-2.5 text-neutral-300">
                <span className="text-[10px] text-emerald-500/80 mt-0.5">›</span>
                <div className="flex-1 min-w-0">
                  <span className="text-neutral-400 text-[10px] font-semibold uppercase tracking-wider block">
                    {ev.label.replaceAll("_", " ")}
                  </span>
                  <p className="text-neutral-300 text-[11px] font-sans leading-relaxed">
                    {ev.message}
                  </p>
                </div>
              </div>
            ))
          ) : (
            <div className="flex items-center gap-2 text-neutral-500 text-xs italic">
              <Sparkles size={13} className="animate-spin text-emerald-400" />
              Formulating execution strategy & subtask breakdown...
            </div>
          )}
        </div>
      )}
    </div>
  );
}
