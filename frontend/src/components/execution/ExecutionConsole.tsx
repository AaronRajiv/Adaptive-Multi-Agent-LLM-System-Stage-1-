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
  Terminal,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ExecutionEvent, ExecutionModel, ExecutionTask } from "@/types/execution";
export type InspectorTab = "synthesis" | "node" | "rag";
interface Props {
  events: ExecutionEvent[];
  run?: ExecutionModel | undefined;
  selected?: ExecutionTask | undefined;
  tab: InspectorTab;
  setTab: (tab: InspectorTab) => void;
  running: boolean;
}
export function TaskDetails({ task }: { task: ExecutionTask }) {
  const conceptual = task.metadata?.["view"] === "Conceptual architecture; not execution telemetry";
  return (
    <div className="task-details">
      <div className="detail-title">
        <strong>{task.title}</strong>
        <span className={`state-text ${conceptual ? "" : `status-${task.status.toLowerCase()}`}`}>
          {conceptual ? "ARCHITECTURE" : task.status}
        </span>
      </div>
      <dl>
        <dt>Task ID</dt>
        <dd>{task.id}</dd>
        <dt>Capability</dt>
        <dd>{task.capability || task.type}</dd>
        <dt>Dependencies</dt>
        <dd>{task.dependencies.length ? task.dependencies.join(", ") : "Not provided"}</dd>
        {task.durationMs !== undefined && (
          <>
            <dt>Latency</dt>
            <dd>{(task.durationMs / 1000).toFixed(2)}s</dd>
          </>
        )}
        {task.agent && (
          <>
            <dt>Agent</dt>
            <dd>{task.agent}</dd>
          </>
        )}
      </dl>
      {task.description && <p>{task.description}</p>}
      {task.query && (
        <section>
          <h4>Retrieval query</h4>
          <p>{task.query}</p>
        </section>
      )}
      {task.input && (
        <section>
          <h4>Input</h4>
          <p>{task.input}</p>
        </section>
      )}
      {task.output && (
        <section>
          <h4>Output</h4>
          <div className="markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{task.output}</ReactMarkdown>
          </div>
        </section>
      )}
      {task.metadata && Object.keys(task.metadata).length > 0 && (
        <section>
          <h4>Metadata</h4>
          <pre>{JSON.stringify(task.metadata, null, 2)}</pre>
        </section>
      )}
    </div>
  );
}
export function FloatingInspector({ task, onClose }: { task: ExecutionTask; onClose: () => void }) {
  return (
    <aside className="floating-inspector">
      <header>
        <span>
          <ChevronRight size={13} /> NODE INSPECTOR
        </span>
        <Button size="icon" variant="ghost" aria-label="Close node inspector" onClick={onClose}>
          <X />
        </Button>
      </header>
      <div>
        <TaskDetails task={task} />
      </div>
    </aside>
  );
}
export function ExecutionConsole({ events, run, selected, tab, setTab, running }: Props) {
  const [copied, setCopied] = useState(false);
  const stream = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  useEffect(() => {
    if (stream.current && follow.current) stream.current.scrollTop = stream.current.scrollHeight;
  }, [events]);
  useEffect(() => {
    setCopied(false);
  }, [run?.runId]);
  const queries = Array.from(
    new Set(
      run?.tasks.map((task) => task.query).filter((query): query is string => Boolean(query)) ?? [],
    ),
  );
  return (
    <section className="execution-console">
      <div className="event-panel">
        <header className="console-header">
          <div>
            <Terminal size={13} />
            <span>Live Event Stream</span>
            <span className="subtle-tag">{run ? "POST-RUN" : running ? "REQUEST" : "IDLE"}</span>
          </div>
          <span className="event-count">{events.length} EVENTS</span>
        </header>
        <div
          className="event-body"
          ref={stream}
          role="log"
          aria-label="Execution events"
          aria-live="polite"
          onScroll={() => {
            const el = stream.current;
            if (el) follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          }}
        >
          {events.length ? (
            events.map((event) => (
              <div className="event-line" key={event.id}>
                <span className="event-time">{event.time || "TRACE"}</span>
                <div className="event-content">
                  <span
                    className={`event-label ${event.status === "FAILED" ? "text-destructive" : event.status === "COMPLETED" ? "text-success" : event.status === "RUNNING" ? "text-evaluation" : "text-muted-foreground"}`}
                  >
                    <i className="event-status-dot" />
                    {event.label.replaceAll("_", " ")}
                  </span>
                  <span className="event-message">{event.message}</span>
                </div>
                <span className="event-source">
                  {event.durationMs !== undefined ? `${event.durationMs}ms · ` : ""}
                  {event.source}
                </span>
              </div>
            ))
          ) : (
            <div className="event-idle">
              <span className="terminal-prompt">›</span>
              <span>Awaiting task submission</span>
              <span className="event-source">IDLE</span>
            </div>
          )}
        </div>
        <footer>
          <span>
            <Activity size={11} />
            {running ? "REQUEST IN FLIGHT" : run ? "POST-RUN TRACE" : "NO ACTIVE EXECUTION"}
          </span>
          <span>{run ? `${run.tasks.length} TASKS` : "LATENCY —"}</span>
          <span>{run ? `${run.events.length} TRACE ENTRIES` : "TOKENS —"}</span>
        </footer>
      </div>
      <div className="inspector-panel">
        <header className="console-header">
          <div className="inspector-tabs" role="tablist">
            {(
              [
                { id: "synthesis", label: "Final Synthesis", icon: FileText },
                { id: "node", label: "Node Inspector", icon: ArrowUpRight },
                { id: "rag", label: "RAG Provenance", icon: Database },
              ] as const
            ).map((item) => (
              <Button
                role="tab"
                aria-selected={tab === item.id}
                key={item.id}
                variant="ghost"
                size="sm"
                className={tab === item.id ? "tab-active" : ""}
                onClick={() => setTab(item.id)}
              >
                <item.icon size={12} />
                {item.label}
              </Button>
            ))}
          </div>
          {tab === "synthesis" && (
            <Button
              size="icon"
              variant="ghost"
              aria-label="Copy final answer"
              title="Copy final answer"
              disabled={!run?.finalAnswer}
              onClick={async () => {
                if (run?.finalAnswer) {
                  try {
                    await navigator.clipboard.writeText(run.finalAnswer);
                    setCopied(true);
                  } catch {
                    setCopied(false);
                  }
                }
              }}
            >
              {copied ? <Check /> : <Copy />}
            </Button>
          )}
        </header>
        <div className="inspector-body">
          {tab === "synthesis" &&
            (run?.finalAnswer ? (
              <>
                <div className="answer-meta">RUN / {run.runId}</div>
                <div className="markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{run.finalAnswer}</ReactMarkdown>
                </div>
              </>
            ) : (
              <div className="inspector-empty">
                <span className="empty-icon">
                  <FileText size={21} />
                </span>
                <div>
                  <strong>No final answer returned yet.</strong>
                  {running && <p>Waiting for the execution response.</p>}
                </div>
              </div>
            ))}
          {tab === "node" &&
            (selected ? (
              <TaskDetails task={selected} />
            ) : (
              <div className="inspector-empty">
                <span className="empty-icon">
                  <ArrowUpRight size={21} />
                </span>
                <div>
                  <strong>No node selected</strong>
                  <p>Awaiting node selection</p>
                </div>
              </div>
            ))}
          {tab === "rag" &&
            (run?.chunks.length || queries.length ? (
              <div className="provenance">
                {queries.map((query) => (
                  <div className="provenance-query" key={query}>
                    Query · {query}
                  </div>
                ))}
                {run?.chunks.map((c, i) => (
                  <article key={i}>
                    <strong>{c.source || c.documentId || `Retrieved chunk ${i + 1}`}</strong>
                    {c.documentId && <span>Document · {c.documentId}</span>}
                    <span>
                      {c.chunkIndex !== undefined ? `Chunk #${c.chunkIndex}` : ""}
                      {c.similarity !== undefined ? ` · Similarity ${c.similarity.toFixed(3)}` : ""}
                    </span>
                    <p>{c.content || "Chunk content not exposed"}</p>
                  </article>
                ))}
                {!run?.chunks.length && <p>Retrieved chunks not exposed for this run.</p>}
              </div>
            ) : (
              <div className="inspector-empty">
                <span className="empty-icon">
                  <Database size={21} />
                </span>
                <div>
                  <strong>RAG telemetry unavailable for this run.</strong>
                </div>
              </div>
            ))}
        </div>
      </div>
    </section>
  );
}
