import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import {
  Activity,
  ArrowRight,
  Check,
  ChevronDown,
  CircleHelp,
  Command,
  GitBranch,
  History,
  Layers3,
  LoaderCircle,
  Network,
  Play,
  Radio,
  RotateCcw,
  Trash2,
  X,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { ExecutionGraph } from "@/components/graph/ExecutionGraph";
import {
  ExecutionConsole,
  FloatingInspector,
  type InspectorTab,
} from "@/components/execution/ExecutionConsole";
import { adaptRun, architectureTasks } from "@/services/adapters";
import { executePipeline, fetchRecentRuns, fetchSystemStatus } from "@/services/api";
import { ResponseExecutionEventSource } from "@/services/executionEvents";
import type { ExecutionEvent, ExecutionModel, ExecutionTask } from "@/types/execution";
const presets = [
  {
    label: "EV vs Petrol Vehicles",
    task: "Compare electric and petrol vehicles across lifecycle emissions, total cost of ownership, infrastructure requirements, and long-term adoption.",
  },
  {
    label: "Renewable Energy Grid",
    task: "Analyze the challenges and opportunities of a renewable energy grid, including storage, reliability, infrastructure, and cost.",
  },
];
export function ControlPlane() {
  const [task, setTask] = useState("");
  const [view, setView] = useState<"architecture" | "execution">("architecture");
  const [run, setRun] = useState<ExecutionModel>();
  const [events, setEvents] = useState<ExecutionEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<ExecutionTask>();
  const [floating, setFloating] = useState(false);
  const [tab, setTab] = useState<InspectorTab>("synthesis");
  const [history, setHistory] = useState(false);
  const [cleared, setCleared] = useState(false);
  const requestId = useRef(0);
  const mounted = useRef(true);
  const status = useQuery({
    queryKey: ["system-status"],
    queryFn: fetchSystemStatus,
    refetchInterval: 15000,
    retry: false,
  });
  const recent = useQuery({ queryKey: ["recent-runs"], queryFn: fetchRecentRuns, retry: false });
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      requestId.current++;
    };
  }, []);
  const event = (
    label: string,
    message: string,
    status?: ExecutionEvent["status"],
  ): ExecutionEvent => ({
    id: `${Date.now()}-${label}`,
    label,
    message,
    source: "CLIENT",
    time: new Date().toLocaleTimeString("en-GB"),
    status,
  });
  const submit = useCallback(async () => {
    if (running || !task.trim()) return;
    const id = ++requestId.current;
    setRunning(true);
    setError("");
    setRun(undefined);
    setSelected(undefined);
    setFloating(false);
    setView("execution");
    setCleared(false);
    setTab("synthesis");
    setEvents([event("TASK_RECEIVED", "Request submitted to the API", "READY")]);
    try {
      const raw = await executePipeline(task.trim());
      if (!mounted.current || id !== requestId.current) return;
      const model = adaptRun(raw);
      setRun(model);
      const trace: ExecutionEvent[] = [];
      new ResponseExecutionEventSource(model).subscribe((e) => trace.push(e));
      setEvents((current) => [
        ...current,
        ...trace,
        event("RESPONSE_RECEIVED", "Execution response received", "COMPLETED"),
      ]);
      recent.refetch();
    } catch (err) {
      if (!mounted.current || id !== requestId.current) return;
      const message = err instanceof Error ? err.message : "Execution failed";
      setError(message);
      setEvents((current) => [...current, event("REQUEST_FAILED", message, "FAILED")]);
    } finally {
      if (mounted.current && id === requestId.current) setRunning(false);
    }
  }, [task, running, recent.refetch]);
  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        void submit();
      }
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [submit]);
  const clear = () => {
    setRun(undefined);
    setEvents([]);
    setError("");
    setSelected(undefined);
    setFloating(false);
    setCleared(true);
    setTab("synthesis");
    setView("execution");
  };
  const tasks = view === "architecture" ? architectureTasks : (run?.tasks ?? []);
  const online = status.data?.configured === true;
  return (
    <main className="control-plane">
      <header className="system-header">
        <div className="brand">
          <div className="brand-mark">
            <Network size={20} />
          </div>
          <div>
            <h1>
              Adaptive Multi-Agent <span>LLM System</span>
            </h1>
            <p>Resource-aware task orchestration</p>
          </div>
          <span className="version-label">STAGE 05</span>
        </div>
        <div className="preset-group">
          <span className="preset-label">PRESETS</span>
          {presets.map((p) => (
            <Button
              key={p.label}
              variant="ghost"
              size="sm"
              className="preset-button"
              disabled={running}
              onClick={() => setTask(p.task)}
            >
              {p.label}
              <ArrowRight size={10} />
            </Button>
          ))}
        </div>
        <div className="header-actions">
          <span
            className={`system-status ${running ? "executing" : online ? "online" : ""}`}
            title={status.data?.message}
          >
            <i />
            {running
              ? "EXECUTING"
              : status.isPending
                ? "CONNECTING"
                : online
                  ? "SYSTEM ONLINE"
                  : "OFFLINE"}
          </span>
          <span className="header-divider" />
          <Button
            variant="ghost"
            size="icon"
            title="Recent runs"
            aria-label="Recent runs"
            onClick={() => setHistory(!history)}
          >
            <History />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            title="Clear canvas"
            aria-label="Clear canvas"
            disabled={running}
            onClick={clear}
          >
            <Trash2 size={13} />
          </Button>
        </div>
      </header>
      <form
        className="task-command"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <span className="command-symbol">
          <Command size={17} />
        </span>
        <input
          aria-label="Analytical task"
          placeholder="Enter complex analytical task..."
          value={task}
          disabled={running}
          onChange={(e) => setTask(e.target.value)}
          autoComplete="off"
        />
        <Button
          type="submit"
          variant="outline"
          disabled={running || !task.trim()}
          className={`run-button ${running ? "is-running" : ""}`}
        >
          {running ? <LoaderCircle className="animate-spin" /> : <Play size={13} />}
          {running ? "Running Pipeline" : "Run Pipeline"}
          <kbd>⌘ ↵</kbd>
        </Button>
      </form>
      <section className="workspace">
        <header className="workspace-header">
          <div className="workspace-title">
            <GitBranch size={15} />
            <strong>{view === "architecture" ? "System Architecture" : "Execution Graph"}</strong>
            <span className="workspace-divider" />
            <span className="workspace-subtitle">
              {view === "architecture"
                ? "Dependency-aware orchestration"
                : run
                  ? `${run.tasks.length} tasks · ${run.runId.slice(0, 12)}`
                  : running
                    ? "Awaiting backend response"
                    : "No active run"}
            </span>
          </div>
          <div className="view-toggle" role="group" aria-label="Graph view">
            <Button
              size="sm"
              variant="ghost"
              aria-pressed={view === "architecture"}
              className={view === "architecture" ? "view-active" : ""}
              onClick={() => {
                setView("architecture");
                setSelected(undefined);
                setFloating(false);
              }}
            >
              <Layers3 size={12} />
              Architecture
            </Button>
            <Button
              size="sm"
              variant="ghost"
              aria-pressed={view === "execution"}
              className={view === "execution" ? "view-active" : ""}
              onClick={() => {
                setView("execution");
                setSelected(undefined);
                setFloating(false);
              }}
            >
              <Radio size={12} />
              Live Execution
            </Button>
          </div>
        </header>
        <div className="canvas-container">
          <div className="canvas-caption">
            <span className="caption-dot" />
            <span>
              {view === "architecture"
                ? "ARCHITECTURE"
                : running
                  ? "REQUEST IN FLIGHT"
                  : run
                    ? "RETURNED EXECUTION"
                    : "EXECUTION VIEW"}
            </span>
            <span className="caption-line" />
            {view === "architecture"
              ? "Conceptual system architecture — not a live execution."
              : run && !run.dependenciesAvailable
                ? "Dependencies not exposed by this API"
                : running
                  ? "No real-time telemetry exposed"
                  : run
                    ? "Post-run results · not live telemetry"
                    : "No current run"}
          </div>
          <ExecutionGraph
            tasks={tasks}
            conceptual={view === "architecture"}
            selectedId={selected?.id}
            onSelect={(t) => {
              setSelected(t);
              setTab("node");
              setFloating(true);
            }}
          />
          {running && (
            <div className="running-notice">
              <LoaderCircle size={16} className="animate-spin" />
              <span>Pipeline request in progress</span>
            </div>
          )}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="error-notice"
              role="alert"
            >
              <span>
                <X size={15} />
                <strong>Execution failed</strong>
              </span>
              <p>{error}</p>
              <Button variant="ghost" size="sm" onClick={() => void submit()}>
                <RotateCcw size={12} />
                Retry
              </Button>
            </motion.div>
          )}
          {selected && floating && (
            <FloatingInspector task={selected} onClose={() => setFloating(false)} />
          )}
          <div className="canvas-legend">
            <span>
              <i className="legend-pending" />
              Pending
            </span>
            <span>
              <i className="legend-running" />
              Running
            </span>
            <span>
              <i className="legend-completed" />
              Completed
            </span>
            <span>
              <i className="legend-failed" />
              Failed
            </span>
            <span className="legend-divider" />
            <span>
              <GitBranch size={11} />
              Dependency
            </span>
          </div>
        </div>
      </section>
      <ExecutionConsole
        events={events}
        run={run}
        selected={selected}
        tab={tab}
        setTab={setTab}
        running={running}
      />
      <footer className="system-footer">
        <span>
          <span className="footer-dot" />
          ADAPTIVE ORCHESTRATION ENGINE<span className="footer-slash">/</span>
          {running
            ? "EXECUTING"
            : cleared
              ? "CANVAS CLEARED"
              : error
                ? "FAILED"
                : run
                  ? "RESPONSE RECEIVED"
                  : "IDLE"}
        </span>
        <span className="footer-model">
          {online && status.data?.model && status.data.model !== "unknown"
            ? `${status.data.provider && status.data.provider !== "unknown" ? `${status.data.provider} / ` : ""}${status.data.model}`
            : "MODEL — UNAVAILABLE"}
          <span className="footer-slash">/</span>
          {online && typeof status.data?.evaluator_threshold === "number"
            ? `EVALUATOR THRESHOLD ${status.data.evaluator_threshold}`
            : "STATUS CHECK EVERY 15s"}
          <CircleHelp size={11} aria-label="Connection details" />
          <span title={status.data?.message}>{online ? "CONNECTED" : "OFFLINE"}</span>
        </span>
      </footer>
      {history && (
        <aside className="history-panel">
          <header>
            <strong>Recent runs</strong>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Close recent runs"
              onClick={() => setHistory(false)}
            >
              <X />
            </Button>
          </header>
          {recent.data?.length ? (
            recent.data.map((r) => (
              <Button
                variant="ghost"
                key={r.run_id}
                className="history-item"
                onClick={() => {
                  try {
                    const model = adaptRun(r);
                    setRun(model);
                    setTask(model.task);
                    setEvents(model.events);
                    setView("execution");
                    setSelected(undefined);
                    setFloating(false);
                    setError("");
                    setHistory(false);
                  } catch (e) {
                    setError(e instanceof Error ? e.message : "Unable to load run");
                  }
                }}
              >
                <span>
                  {r.user_task}
                  <small>{r.run_id}</small>
                </span>
                <ChevronDown size={13} />
              </Button>
            ))
          ) : (
            <p>{online ? "No recent runs returned." : "Recent runs unavailable while offline."}</p>
          )}
          <Button variant="ghost" size="sm" onClick={() => recent.refetch()}>
            <RotateCcw size={12} />
            Refresh
          </Button>
        </aside>
      )}
    </main>
  );
}
