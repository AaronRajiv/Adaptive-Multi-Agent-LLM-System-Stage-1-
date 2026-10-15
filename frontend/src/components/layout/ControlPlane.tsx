import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { LiveArchitectureCanvas } from "@/components/architecture/LiveArchitectureCanvas";
import { TelemetryConsole, type TelemetryTab } from "@/components/telemetry/TelemetryConsole";
import { DocumentManagerModal } from "@/components/documents/DocumentManagerModal";

import { adaptRun } from "@/services/adapters";
import {
  executePipeline,
  fetchRecentRuns,
  fetchSystemStatus,
  uploadDocument,
  fetchDocumentStatus,
  cancelRun,
  fetchRun,
  fetchModels,
} from "@/services/api";
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

  // Document Manager & Ingestion states
  const [docManagerOpen, setDocManagerOpen] = useState(false);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const currentActiveRunId = useRef<string>("");

  const handleCancel = async () => {
    if (currentActiveRunId.current && running) {
      try {
        await cancelRun(currentActiveRunId.current);
        setRunning(false);
        setEvents((prev) => [
          ...prev,
          createClientEvent("RUN_CANCELLED", "Execution cancelled by user request", "FAILED"),
        ]);
      } catch (e) {
        console.error("Cancel failed:", e);
      }
    }
  };

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

  const modelsQuery = useQuery({
    queryKey: ["available-models"],
    queryFn: fetchModels,
    refetchInterval: 60000,
    retry: false,
  });

  const [selectedModel, setSelectedModel] = useState<string>(() => {
    try {
      return (
        localStorage.getItem("selected_llm_model") ||
        "nvidia/nemotron-3-super-120b-a12b"
      );
    } catch {
      return "nvidia/nemotron-3-super-120b-a12b";
    }
  });

  const handleSelectModel = (modelId: string) => {
    setSelectedModel(modelId);
    try {
      localStorage.setItem("selected_llm_model", modelId);
    } catch {}
  };

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
    async (files?: File[], overridePrompt?: string) => {
      const promptText = (overridePrompt ?? task).trim();
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

      // Generate runId early so SSE and child components have it immediately
      const runId =
        typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
          ? crypto.randomUUID()
          : `run-${Date.now()}`;
      currentActiveRunId.current = runId;

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

      const modelDisplayName =
        modelsQuery.data?.models.find((m) => m.id === selectedModel)?.name ||
        selectedModel;

      const assistantMsg: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        events: [initialClientEv],
        running: true,
        runId,
        modelName: modelDisplayName,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      let liveState: LiveExecutionState = {
        running: true,
        tasks: [],
        events: [initialClientEv],
        chunks: [],
      };

      const updateLiveEvents = (newEvent: ExecutionEvent) => {
        liveState = {
          ...liveState,
          events: [...liveState.events, newEvent],
        };
        setEvents([...liveState.events]);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId ? { ...msg, events: [...liveState.events] } : msg
          )
        );
      };

      let unsubscribe: () => void = () => {};
      let currentArch = createInitialArchitectureState();

      try {
        // Upload attached files to document service and obtain document_ids
        const attachedDocIds: string[] = [];
        if (files && files.length > 0) {
          for (const f of files) {
            updateLiveEvents(
              createClientEvent("DOC_UPLOAD", `Uploading '${f.name}' to document service...`, "RUNNING")
            );
            const uploadRes = await uploadDocument(f, f.name);
            attachedDocIds.push(uploadRes.document_id);

            if (uploadRes.status !== "ready") {
              const pollDocInterval = 800;
              let isDocReady = false;
              for (let d = 0; d < 300; d++) {
                await new Promise((r) => setTimeout(r, pollDocInterval));
                const docStatus = await fetchDocumentStatus(uploadRes.document_id);
                if (docStatus.status === "ready") {
                  updateLiveEvents(
                    createClientEvent(
                      "DOC_READY",
                      `'${f.name}' processed & ready (${docStatus.chunk_count} chunks indexed).`,
                      "COMPLETED"
                    )
                  );
                  isDocReady = true;
                  break;
                }
                if (docStatus.status === "failed") {
                  throw new Error(
                    `Document ingestion failed for '${f.name}': ${docStatus.error_message || "Unknown error"}`
                  );
                }
                if (docStatus.status === "extracting") {
                  updateLiveEvents(
                    createClientEvent("DOC_EXTRACTING", `Extracting text from '${f.name}'...`, "RUNNING")
                  );
                } else if (docStatus.status === "embedding") {
                  updateLiveEvents(
                    createClientEvent(
                      "DOC_EMBEDDING",
                      `Indexing '${f.name}' (${docStatus.embedded_chunk_count}/${docStatus.chunk_count} chunks)...`,
                      "RUNNING"
                    )
                  );
                }
              }
              if (!isDocReady) {
                throw new Error(`Document ingestion timed out for '${f.name}'.`);
              }
            } else {
              updateLiveEvents(
                createClientEvent(
                  "DOC_READY",
                  `'${f.name}' already ingested (deduplicated by SHA-256).`,
                  "COMPLETED"
                )
              );
            }
          }
        }

        const allDocIds = Array.from(new Set([...selectedDocIds, ...attachedDocIds]));

        const initialEv = createClientEvent("RUN_STARTED", "Pipeline execution started", "RUNNING");
        currentArch = updateArchitectureFromEvent(currentArch, initialEv);
        updateLiveEvents(initialEv);
        setLiveTasks(liveState.tasks);
        setArchState(currentArch);

        const sseSource = new ServerExecutionEventSource(runId);
        unsubscribe = sseSource.subscribe(
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

        const raw = await executePipeline(
          promptText,
          runId,
          undefined,
          historyPayload,
          allDocIds.length > 0 ? allDocIds : undefined,
          true,
          selectedModel
        );
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
              ? {
                  ...msg,
                  content: model.finalAnswer,
                  running: false,
                  executionStrategy: model.executionStrategy,
                  runId: model.runId,
                  validCitations: model.verification?.validCitations,
                  statusMessage: undefined,
                  modelName: modelDisplayName,
                }
              : msg
          )
        );

        recent.refetch();
      } catch (err) {
        if (!mounted.current || id !== requestId.current) return;
        // Honest recovery: transition to "Execution in progress, synchronizing..." and poll run status
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, statusMessage: "Execution in progress, synchronizing...", running: true }
              : msg
          )
        );

        let recovered = false;
        let consecutiveErrors = 0;
        try {
          // Poll GET /api/runs/{runId} to inspect backend state honestly
          for (let poll = 0; poll < 360; poll++) {
            await new Promise((r) => setTimeout(r, 2000));
            if (!mounted.current || id !== requestId.current) return;
            const check = await fetchRun(runId).catch(() => null);
            if (check) {
              consecutiveErrors = 0;
              if (check.status === "completed" && check.result) {
                const model = adaptRun(check.result);
                setRun(model);
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? {
                          ...msg,
                          content: model.finalAnswer,
                          running: false,
                          statusMessage: undefined,
                          runId: model.runId,
                          validCitations: model.verification?.validCitations,
                          executionStrategy: model.executionStrategy,
                        }
                      : msg
                  )
                );
                recovered = true;
                break;
              } else if (check.status === "failed") {
                const failMsg = check.error || "Backend pipeline execution failed.";
                setError(failMsg);
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? { ...msg, error: failMsg, running: false, statusMessage: undefined }
                      : msg
                  )
                );
                recovered = true;
                break;
              } else if (check.status === "cancelled") {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? { ...msg, error: "Execution cancelled.", running: false, statusMessage: undefined }
                      : msg
                  )
                );
                recovered = true;
                break;
              } else if (check.status === "running") {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? { ...msg, statusMessage: "Execution in progress on server...", running: true }
                      : msg
                  )
                );
              }
            } else {
              consecutiveErrors++;
              if (consecutiveErrors >= 20) {
                // Backend unreachable for 40 consecutive seconds
                break;
              }
            }
          }
        } catch {
          // Ignore polling errors
        }

        if (!recovered) {
          const message = "Lost connection to server. Run status unknown; verify network or backend.";
          setError(message);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? { ...msg, error: message, running: false, statusMessage: undefined }
                : msg
            )
          );
        }
      } finally {
        unsubscribe();
        if (mounted.current && id === requestId.current) setRunning(false);
      }
    },
    [task, running, messages, selectedDocIds, recent.refetch]
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
        onOpenDocumentManager={() => setDocManagerOpen(true)}
      />

      {/* 2. Center Chat Panel */}
      <div className="flex-1 h-full min-w-[320px]">
        <ChatPanel
          messages={messages}
          task={task}
          setTask={setTask}
          onSubmit={(files, overridePrompt) => void submit(files, overridePrompt)}
          running={running}
          onClear={clear}
          presets={PRESETS}
          onCancel={handleCancel}
          onOpenEvidence={() => setTelemetryTab("evidence")}
          models={modelsQuery.data?.models || []}
          selectedModel={selectedModel}
          onSelectModel={handleSelectModel}
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
              } else if (nodeId === "rag" || nodeId === "tools") {
                setTelemetryTab("evidence");
              } else if (nodeId === "planner" || nodeId === "api_gateway" || nodeId === "user_task") {
                setTelemetryTab("strategy");
              } else if (nodeId === "evaluation") {
                setTelemetryTab("replanning");
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

      {/* 4. Document Manager Modal */}
      <DocumentManagerModal
        open={docManagerOpen}
        onClose={() => setDocManagerOpen(false)}
        selectedDocIds={selectedDocIds}
        onToggleSelectDoc={(docId) =>
          setSelectedDocIds((prev) =>
            prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
          )
        }
      />
    </div>
  );
}
