import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { LiveArchitectureCanvas } from "@/components/architecture/LiveArchitectureCanvas";
import { TelemetryConsole, type TelemetryTab } from "@/components/telemetry/TelemetryConsole";

import { adaptRun } from "@/services/adapters";
import { executePipeline, fetchRecentRuns, fetchSystemStatus } from "@/services/api";
import { ServerExecutionEventSource } from "@/services/executionEvents";
import { processExecutionEvent, type LiveExecutionState } from "@/services/executionState";
import {
  createInitialArchitectureState,
  updateArchitectureFromEvent,
  updateArchitectureFromModel,
  type ArchitectureState,
} from "@/services/architectureState";
import type { ChatMessage, ExecutionEvent, ExecutionModel, ExecutionTask } from "@/types/execution";
import type { ArchitectureNodeId } from "@/types/architecture";
import type { ChatMessagePayload, DocumentPayload } from "@/types/api";

const PRESETS = [
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
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [run, setRun] = useState<ExecutionModel>();
  const [liveTasks, setLiveTasks] = useState<ExecutionTask[]>([]);
  const [events, setEvents] = useState<ExecutionEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [selectedTask, setSelectedTask] = useState<ExecutionTask>();
  const [selectedNodeId, setSelectedNodeId] = useState<ArchitectureNodeId>();
  const [telemetryTab, setTelemetryTab] = useState<TelemetryTab>("events");

  // QoL States: Collapsible Sidebar & Resizable Right Panel
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [rightPanelWidth, setRightPanelWidth] = useState(540);
  const isDraggingRef = useRef(false);

  const [archState, setArchState] = useState<ArchitectureState>(createInitialArchitectureState());

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

  // Draggable Resizer Handler for Right Panel
  const startResizing = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDraggingRef.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const newWidth = window.innerWidth - e.clientX;
      if (newWidth >= 360 && newWidth <= 880) {
        setRightPanelWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      if (isDraggingRef.current) {
        isDraggingRef.current = false;
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  const createClientEvent = (
    label: string,
    message: string,
    status?: ExecutionEvent["status"]
  ): ExecutionEvent => ({
    id: `${Date.now()}-${label}`,
    label,
    message,
    source: "CLIENT",
    time: new Date().toLocaleTimeString("en-GB"),
    status,
  });

  const submit = useCallback(
    async (files?: File[]) => {
      const promptText = task.trim();
      if (running || !promptText) return;
      const id = ++requestId.current;

      setRunning(true);
      setError("");
      setTask("");
      setSelectedTask(undefined);
      setSelectedNodeId(undefined);

      // Extract conversation history from previous completed turns
      const historyPayload: ChatMessagePayload[] = messages
        .filter((m) => Boolean(m.content))
        .map((m) => ({ role: m.role, content: m.content }));

      // Append user prompt message & assistant placeholder message
      const userMsgId = `user-${Date.now()}`;
      const assistantMsgId = `assistant-${Date.now()}`;
      const initialClientEv = createClientEvent("TASK_RECEIVED", "Request submitted to the API", "READY");

      const userMsg: ChatMessage = {
        id: userMsgId,
        role: "user",
        content: promptText,
        files: files ? [...files] : undefined,
      };

      const assistantMsg: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        events: [initialClientEv],
        running: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      // Read attached files text for RAG ingestion
      const docPayloads: DocumentPayload[] = [];
      if (files && files.length > 0) {
        for (const f of files) {
          try {
            if (f.name.toLowerCase().endsWith(".pdf")) {
              const dataUrl = await new Promise<string>((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve((reader.result as string) || "");
                reader.onerror = reject;
                reader.readAsDataURL(f);
              });
              if (dataUrl) {
                docPayloads.push({ filename: f.name, content: dataUrl });
              }
            } else {
              const text = await f.text();
              if (text.trim()) {
                docPayloads.push({ filename: f.name, content: text });
              }
            }
          } catch (e) {
            console.error("File reading error:", e);
          }
        }
      }

      const runId =
        typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
          ? crypto.randomUUID()
          : `run-${Date.now()}`;

      let liveState: LiveExecutionState = {
        running: true,
        tasks: [],
        events: [initialClientEv],
        chunks: [],
      };

      let currentArch = createInitialArchitectureState();
      const initialEv = createClientEvent("RUN_STARTED", "Pipeline execution started", "RUNNING");
      currentArch = updateArchitectureFromEvent(currentArch, initialEv);

      setEvents(liveState.events);
      setLiveTasks(liveState.tasks);
      setArchState(currentArch);

      const sseSource = new ServerExecutionEventSource(runId);
      const unsubscribe = sseSource.subscribe(
        (ev) => {
          if (!mounted.current || id !== requestId.current) return;

          liveState = processExecutionEvent(liveState, ev);
          currentArch = updateArchitectureFromEvent(currentArch, ev);

          setEvents([...liveState.events]);
          setLiveTasks([...liveState.tasks]);
          setArchState({ ...currentArch });

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId ? { ...msg, events: [...liveState.events] } : msg
            )
          );
        },
        () => {}
      );

      try {
        const raw = await executePipeline(promptText, runId, docPayloads, historyPayload);
        if (!mounted.current || id !== requestId.current) return;

        const model = adaptRun(raw);
        setRun(model);

        const compEv = createClientEvent("RUN_COMPLETED", "Pipeline execution completed successfully", "COMPLETED");
        currentArch = updateArchitectureFromEvent(currentArch, compEv);
        if (model.chunks && model.chunks.length > 0) {
          currentArch.ragUsed = true;
          const ragNode = currentArch.nodes.find((n) => n.id === "rag");
          if (ragNode) {
            ragNode.status = "COMPLETED";
            ragNode.detail = `${model.chunks.length} chunk(s) retrieved`;
          }
        }
        setArchState({ ...currentArch });

        setEvents((current) => [
          ...current,
          createClientEvent("RESPONSE_RECEIVED", "Execution response received", "COMPLETED"),
        ]);

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, content: model.finalAnswer, running: false, executionStrategy: model.executionStrategy }
              : msg
          )
        );

        recent.refetch();
      } catch (err) {
        if (!mounted.current || id !== requestId.current) return;
        const message = err instanceof Error ? err.message : "Execution failed";
        setError(message);

        const failEv = createClientEvent("RUN_FAILED", message, "FAILED");
        currentArch = updateArchitectureFromEvent(currentArch, failEv);
        setArchState({ ...currentArch });

        setEvents((current) => [...current, createClientEvent("REQUEST_FAILED", message, "FAILED")]);

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId ? { ...msg, error: message, running: false } : msg
          )
        );
      } finally {
        unsubscribe();
        if (mounted.current && id === requestId.current) setRunning(false);
      }
    },
    [task, running, messages, recent.refetch]
  );

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
    setMessages([]);
    setRun(undefined);
    setLiveTasks([]);
    setEvents([]);
    setError("");
    setSelectedTask(undefined);
    setSelectedNodeId(undefined);
    setArchState(createInitialArchitectureState());
  };

  const handleSelectRecentRun = (runId: string) => {
    const target = recent.data?.find((r) => r.run_id === runId);
    if (!target) return;
    try {
      const model = adaptRun(target);
      setRun(model);
      setTask("");
      setEvents(model.events);
      setLiveTasks(model.tasks);
      setError("");

      setMessages([
        { id: `hist-user-${runId}`, role: "user", content: model.task },
        { id: `hist-ast-${runId}`, role: "assistant", content: model.finalAnswer, events: model.events },
      ]);

      setArchState(updateArchitectureFromModel(model));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to load run");
    }
  };

  const displayTasks = run?.tasks ?? liveTasks;
  const online = status.data?.configured === true;

  return (
    <div className="flex h-screen w-screen bg-black overflow-hidden font-sans select-none">
      {/* 1. Collapsible Left Conversation Sidebar */}
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        running={running}
        online={online}
        statusMessage={status.data?.message}
        presets={PRESETS}
        onSelectPreset={(t) => setTask(t)}
        recentRuns={recent.data}
        onSelectRecentRun={handleSelectRecentRun}
        onRefreshRecent={() => recent.refetch()}
        onClear={clear}
      />

      {/* 2. Center Chat Panel */}
      <div className="flex-1 h-full min-w-[320px]">
        <ChatPanel
          messages={messages}
          task={task}
          setTask={setTask}
          onSubmit={(files) => void submit(files)}
          running={running}
          onClear={clear}
          presets={PRESETS}
        />
      </div>

      {/* Draggable Divider Handle */}
      <div
        onMouseDown={startResizing}
        className="w-1.5 h-full bg-white/5 hover:bg-emerald-500/50 cursor-col-resize transition-colors flex-shrink-0 flex items-center justify-center group z-30"
        title="Drag to resize panel width"
      >
        <div className="w-0.5 h-8 bg-neutral-600 group-hover:bg-emerald-400 rounded-full transition-colors" />
      </div>

      {/* 3. Resizable Right Observability Panel */}
      <div
        style={{ width: `${rightPanelWidth}px` }}
        className="h-full flex flex-col p-3 gap-3 bg-neutral-950/90 border-l border-white/10 flex-shrink-0"
      >
        {/* Live Architecture Canvas (Top 56%) */}
        <div className="h-[56%] min-h-[280px]">
          <LiveArchitectureCanvas
            nodes={archState.nodes}
            particles={archState.particles}
            ragUsed={archState.ragUsed}
            toolsUsed={archState.toolsUsed}
            selectedNodeId={selectedNodeId}
            onSelectNode={(nodeId) => {
              setSelectedNodeId(nodeId);
              if (nodeId === "task_graph" || nodeId === "agents") {
                setTelemetryTab("graph");
              } else if (nodeId === "rag") {
                setTelemetryTab("rag");
              } else {
                setTelemetryTab("node");
              }
            }}
          />
        </div>

        {/* Telemetry Console & Inspector (Bottom 44%) */}
        <div className="flex-1 min-h-[180px]">
          <TelemetryConsole
            events={events}
            run={run}
            selectedTask={selectedTask}
            activeTab={telemetryTab}
            setActiveTab={setTelemetryTab}
            running={running}
            tasks={displayTasks}
          />
        </div>
      </div>
    </div>
  );
}
