import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Database,
  FileSearch,
  FileText,
  GitBranch,
  Layers,
  Minus,
  Network,
  Plus,
  Sparkles,
  User,
  Wrench,
  XCircle,
} from "lucide-react";
import type {
  ArchitectureEdge,
  ArchitectureNode,
  ArchitectureNodeId,
  ArchitectureParticle,
} from "@/types/architecture";
import { ARCHITECTURE_EDGES } from "@/services/architectureState";

interface Props {
  nodes: ArchitectureNode[];
  particles: ArchitectureParticle[];
  ragUsed: boolean;
  toolsUsed: boolean;
  onSelectNode?: ((nodeId: ArchitectureNodeId) => void) | undefined;
  selectedNodeId?: string | undefined;
}

const nodeIcons: Record<ArchitectureNodeId, React.ComponentType<{ size?: number; className?: string }>> = {
  user_task: User,
  api_gateway: Network,
  planner: Layers,
  task_graph: GitBranch,
  orchestrator: Cpu,
  rag: Database,
  agents: FileSearch,
  tools: Wrench,
  evaluation: Activity,
  synthesis: Sparkles,
  final_answer: FileText,
};

export function LiveArchitectureCanvas({
  nodes,
  particles,
  ragUsed,
  toolsUsed,
  onSelectNode,
  selectedNodeId,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [now, setNow] = useState(Date.now());
  const [zoom, setZoom] = useState(1.0);

  useEffect(() => {
    let animId: number;
    const tick = () => {
      setNow(Date.now());
      animId = requestAnimationFrame(tick);
    };
    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, []);

  const getNodeMap = () => {
    const map = new Map<ArchitectureNodeId, ArchitectureNode>();
    nodes.forEach((n) => map.set(n.id, n));
    return map;
  };

  const nodeMap = getNodeMap();

  const calculateEdgePath = (edge: ArchitectureEdge) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) return "";

    const x1 = source.x;
    const y1 = source.y;
    const x2 = target.x;
    const y2 = target.y;

    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);
    const cx1 = x1 + (x2 > x1 ? dx * 0.4 : -dx * 0.4);
    const cy1 = y1 + (y2 === y1 ? 0 : dy * 0.1);
    const cx2 = x2 - (x2 > x1 ? dx * 0.4 : -dx * 0.4);
    const cy2 = y2 - (y2 === y1 ? 0 : dy * 0.1);

    return `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`;
  };

  const getPointOnCurve = (edge: ArchitectureEdge, t: number) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) return { x: 0, y: 0 };

    const clampedT = Math.max(0, Math.min(1, t));
    const x1 = source.x;
    const y1 = source.y;
    const x2 = target.x;
    const y2 = target.y;

    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);
    const cx1 = x1 + (x2 > x1 ? dx * 0.4 : -dx * 0.4);
    const cy1 = y1 + (y2 === y1 ? 0 : dy * 0.1);
    const cx2 = x2 - (x2 > x1 ? dx * 0.4 : -dx * 0.4);
    const cy2 = y2 - (y2 === y1 ? 0 : dy * 0.1);

    const u = 1 - clampedT;
    const tt = clampedT * clampedT;
    const uu = u * u;
    const uuu = uu * u;
    const ttt = tt * clampedT;

    const x = uuu * x1 + 3 * uu * clampedT * cx1 + 3 * u * tt * cx2 + ttt * x2;
    const y = uuu * y1 + 3 * uu * clampedT * cy1 + 3 * u * tt * cy2 + ttt * y2;

    return { x, y };
  };

  const handleZoomIn = () => setZoom((prev) => Math.min(2.2, prev + 0.15));
  const handleZoomOut = () => setZoom((prev) => Math.max(0.6, prev - 0.15));
  const handleZoomReset = () => setZoom(1.0);

  return (
    <div className="relative w-full h-full min-h-[360px] bg-neutral-950 rounded-xl border border-white/10 overflow-hidden select-none flex flex-col backdrop-blur-md">
      {/* Header Controls */}
      <div className="flex items-center justify-between px-3.5 py-2 border-b border-white/10 bg-neutral-900/60 z-20">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[11px] font-medium tracking-wider text-neutral-300 uppercase font-mono">
            Architecture Data-Flow
          </span>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center bg-neutral-900/90 border border-white/10 rounded px-1 text-[11px] font-mono text-neutral-300">
            <button
              onClick={handleZoomOut}
              className="p-1 hover:text-white transition-colors"
              title="Zoom out"
            >
              <Minus size={11} />
            </button>
            <button
              onClick={handleZoomReset}
              className="px-1.5 hover:text-white text-[10px] transition-colors"
              title="Reset zoom"
            >
              {Math.round(zoom * 100)}%
            </button>
            <button
              onClick={handleZoomIn}
              className="p-1 hover:text-white transition-colors"
              title="Zoom in"
            >
              <Plus size={11} />
            </button>
          </div>

          <div className="hidden sm:flex items-center gap-3 text-[10px] text-neutral-400 font-mono">
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-neutral-600" /> IDLE
            </span>
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]" /> ACTIVE
            </span>
          </div>
        </div>
      </div>

      {/* Scalable Container for Architecture Canvas */}
      <div className="relative flex-1 w-full h-full overflow-hidden" ref={containerRef}>
        <div
          className="relative w-full h-full transition-transform duration-150 origin-center"
          style={{ transform: `scale(${zoom})` }}
        >
          {/* SVG Connection Paths & Glowing Particles */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-0" viewBox="0 0 100 100" preserveAspectRatio="none">
            <defs>
              <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="0.8" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            {/* QoL #4: Distinct, visible edge lines between spaced nodes */}
            {ARCHITECTURE_EDGES.map((edge) => {
              const pathD = calculateEdgePath(edge);
              const sourceNode = nodeMap.get(edge.source);
              const targetNode = nodeMap.get(edge.target);

              const isConditional = edge.isConditional;
              const isConditionalActive =
                (edge.source === "rag" || edge.target === "rag") ? ragUsed :
                (edge.source === "tools" || edge.target === "tools") ? toolsUsed : true;

              const isEdgeActive =
                sourceNode &&
                targetNode &&
                (sourceNode.status === "ACTIVE" || targetNode.status === "ACTIVE") &&
                isConditionalActive;

              const isEdgeCompleted =
                sourceNode &&
                targetNode &&
                sourceNode.status === "COMPLETED" &&
                targetNode.status === "COMPLETED" &&
                isConditionalActive;

              return (
                <g key={edge.id}>
                  <path
                    d={pathD}
                    fill="none"
                    stroke={
                      isConditional && !isConditionalActive
                        ? "rgba(255, 255, 255, 0.05)"
                        : isEdgeActive
                        ? "rgba(52, 211, 153, 0.85)"
                        : isEdgeCompleted
                        ? "rgba(52, 211, 153, 0.45)"
                        : "rgba(255, 255, 255, 0.2)"
                    }
                    strokeWidth={isEdgeActive ? "2.0" : isEdgeCompleted ? "1.5" : "1.2"}
                    strokeDasharray={isConditional && !isConditionalActive ? "2,2" : "none"}
                    className="transition-all duration-300"
                  />
                </g>
              );
            })}

            {/* Live Glowing Particles */}
            {particles.map((particle) => {
              const edge = ARCHITECTURE_EDGES.find(
                (e) => e.source === particle.source && e.target === particle.target
              );
              if (!edge) return null;

              const elapsed = now - particle.startTime;
              const progress = Math.min(1, Math.max(0, elapsed / particle.durationMs));
              if (progress <= 0 || progress >= 1) return null;

              const pt = getPointOnCurve(edge, progress);
              const tailPt = getPointOnCurve(edge, Math.max(0, progress - 0.09));

              return (
                <g key={particle.id} filter="url(#glow)">
                  <line
                    x1={tailPt.x}
                    y1={tailPt.y}
                    x2={pt.x}
                    y2={pt.y}
                    stroke={particle.color || "#10b981"}
                    strokeWidth="1.2"
                    strokeOpacity={0.8 * (1 - progress * 0.3)}
                    strokeLinecap="round"
                  />
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r="1.4"
                    fill="#ffffff"
                    stroke={particle.color || "#34d399"}
                    strokeWidth="0.5"
                  />
                </g>
              );
            })}
          </svg>

          {/* HTML Architecture Nodes */}
          <div className="absolute inset-0 pointer-events-auto">
            {nodes.map((node) => {
              const Icon = nodeIcons[node.id] || Cpu;
              const isSelected = selectedNodeId === node.id;
              const isConditionalInactive = node.isConditional && !(node.id === "rag" ? ragUsed : toolsUsed);

              let borderStyle = "border-white/15 bg-neutral-950/90 text-neutral-200 hover:border-white/30";
              let glowStyle = "";

              if (isConditionalInactive) {
                borderStyle = "border-white/5 bg-neutral-950/40 text-neutral-600 opacity-50";
              } else {
                switch (node.status) {
                  case "ACTIVE":
                    borderStyle = "border-emerald-400 bg-neutral-900/95 text-white shadow-[0_0_20px_rgba(52,211,153,0.25)]";
                    glowStyle = "ring-1 ring-emerald-400/50";
                    break;
                  case "COMPLETED":
                    borderStyle = "border-emerald-500/40 bg-neutral-950/95 text-neutral-200";
                    break;
                  case "FAILED":
                    borderStyle = "border-red-500 bg-neutral-900 text-red-200 shadow-[0_0_20px_rgba(239,68,68,0.25)]";
                    break;
                  case "BLOCKED":
                    borderStyle = "border-amber-500/70 bg-neutral-950 text-amber-300";
                    break;
                  case "IDLE":
                  default:
                    borderStyle = "border-white/15 bg-neutral-950/90 text-neutral-200 hover:border-white/30";
                    break;
                }
              }

              return (
                <div
                  key={node.id}
                  onClick={() => onSelectNode && onSelectNode(node.id)}
                  style={{
                    left: `${node.x}%`,
                    top: `${node.y}%`,
                    transform: "translate(-50%, -50%)",
                  }}
                  className={`absolute z-10 cursor-pointer transition-all duration-200 rounded-lg border px-3 py-2 min-w-[130px] max-w-[155px] ${borderStyle} ${glowStyle} ${
                    isSelected ? "ring-2 ring-emerald-400 border-transparent shadow-xl" : ""
                  }`}
                >
                  <div className="flex items-center justify-between gap-1 mb-1">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <Icon
                        size={13}
                        className={
                          node.status === "ACTIVE"
                            ? "text-emerald-400 animate-pulse"
                            : node.status === "COMPLETED"
                            ? "text-emerald-400"
                            : node.status === "FAILED"
                            ? "text-red-400"
                            : "text-neutral-400"
                        }
                      />
                      <span className="text-[11px] font-semibold tracking-tight truncate">
                        {node.label}
                      </span>
                    </div>

                    {node.status === "ACTIVE" && (
                      <span className="flex h-2 w-2 relative flex-shrink-0">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                      </span>
                    )}
                    {node.status === "COMPLETED" && (
                      <CheckCircle2 size={11} className="text-emerald-400 flex-shrink-0" />
                    )}
                    {node.status === "FAILED" && (
                      <XCircle size={11} className="text-red-400 flex-shrink-0" />
                    )}
                    {node.status === "BLOCKED" && (
                      <AlertTriangle size={11} className="text-amber-400 flex-shrink-0" />
                    )}
                  </div>

                  <p className="text-[9px] leading-tight text-neutral-400 truncate">
                    {node.subtitle}
                  </p>

                  {node.detail && (
                    <div className="mt-1 pt-1 border-t border-white/5 text-[8.5px] font-mono text-emerald-400 font-medium truncate">
                      {node.detail}
                    </div>
                  )}
                  {isConditionalInactive && (
                    <div className="mt-0.5 text-[8px] font-mono text-neutral-600 italic">
                      Unused
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
