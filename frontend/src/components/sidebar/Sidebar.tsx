import {
  ArrowRight,
  ChevronRight,
  History,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  RotateCcw,
  Sparkles,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  collapsed: boolean;
  onToggleCollapse: () => void;
  running: boolean;
  online: boolean;
  statusMessage?: string | undefined;
  presets: { label: string; task: string }[];
  onSelectPreset: (task: string) => void;
  recentRuns?: Array<{ run_id: string; user_task: string }> | undefined;
  onSelectRecentRun: (runId: string) => void;
  onRefreshRecent: () => void;
  onClear: () => void;
}

export function Sidebar({
  collapsed,
  onToggleCollapse,
  running,
  online,
  presets,
  onSelectPreset,
  recentRuns,
  onSelectRecentRun,
  onRefreshRecent,
  onClear,
}: Props) {
  return (
    <aside
      className={`${
        collapsed ? "w-14" : "w-64"
      } h-full bg-neutral-950 border-r border-white/10 flex flex-col justify-between select-none text-neutral-300 transition-all duration-300 relative z-30`}
    >
      {/* Header Bar */}
      <div className="p-3 border-b border-white/10 flex items-center justify-between">
        {!collapsed ? (
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-7 h-7 rounded bg-white/5 border border-white/10 flex items-center justify-center text-emerald-400 flex-shrink-0">
              <Network size={15} />
            </div>
            <div className="min-w-0">
              <h1 className="text-xs font-semibold text-neutral-100 tracking-tight leading-none truncate">
                Adaptive LLM
              </h1>
              <p className="text-[9.5px] text-neutral-400 mt-0.5 truncate">
                Multi-Agent System
              </p>
            </div>
          </div>
        ) : (
          <div
            className="w-8 h-8 rounded bg-white/5 border border-white/10 flex items-center justify-center text-emerald-400 mx-auto cursor-pointer"
            onClick={onToggleCollapse}
            title="Expand Sidebar"
          >
            <Network size={16} />
          </div>
        )}

        {/* Collapse Toggle Button */}
        <Button
          size="icon"
          variant="ghost"
          className="h-7 w-7 text-neutral-400 hover:text-white flex-shrink-0"
          onClick={onToggleCollapse}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
        </Button>
      </div>

      {/* Action & List Items */}
      <div className="flex-1 overflow-y-auto p-2 space-y-4">
        {/* New Run Button */}
        {!collapsed ? (
          <Button
            variant="outline"
            size="sm"
            className="w-full justify-start gap-2 bg-neutral-900 border-white/10 hover:border-white/20 text-neutral-200 text-xs"
            disabled={running}
            onClick={onClear}
          >
            <Plus size={14} className="text-emerald-400" />
            New Execution Run
          </Button>
        ) : (
          <Button
            size="icon"
            variant="outline"
            className="w-9 h-9 mx-auto flex items-center justify-center bg-neutral-900 border-white/10 hover:border-white/20 text-neutral-200"
            disabled={running}
            onClick={onClear}
            title="New Execution Run"
          >
            <Plus size={15} className="text-emerald-400" />
          </Button>
        )}

        {/* Presets List */}
        {!collapsed ? (
          <div className="space-y-1">
            <span className="text-[10px] font-mono uppercase tracking-wider text-neutral-400 px-1">
              Task Presets
            </span>
            <div className="space-y-1">
              {presets.map((p) => (
                <Button
                  key={p.label}
                  variant="ghost"
                  size="sm"
                  className="w-full justify-between text-left text-xs font-normal text-neutral-400 hover:text-neutral-100 hover:bg-white/5 h-auto py-1.5 px-2 rounded"
                  disabled={running}
                  onClick={() => onSelectPreset(p.task)}
                >
                  <span className="truncate">{p.label}</span>
                  <ArrowRight size={11} className="text-neutral-600 flex-shrink-0" />
                </Button>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 pt-2">
            {presets.map((p) => (
              <button
                key={p.label}
                disabled={running}
                onClick={() => onSelectPreset(p.task)}
                className="w-8 h-8 rounded bg-white/5 hover:bg-white/10 flex items-center justify-center text-neutral-400 hover:text-white"
                title={p.label}
              >
                <Sparkles size={13} />
              </button>
            ))}
          </div>
        )}

        {/* Recent Runs */}
        {!collapsed && (
          <div className="space-y-1">
            <div className="flex items-center justify-between px-1">
              <span className="text-[10px] font-mono uppercase tracking-wider text-neutral-400 flex items-center gap-1">
                <History size={11} /> Recent Runs
              </span>
              <button
                onClick={onRefreshRecent}
                className="text-neutral-500 hover:text-neutral-300 text-[10px] p-0.5"
                title="Refresh recent runs"
              >
                <RotateCcw size={10} />
              </button>
            </div>

            <div className="space-y-1 max-h-48 overflow-y-auto">
              {recentRuns && recentRuns.length > 0 ? (
                recentRuns.map((r) => (
                  <button
                    key={r.run_id}
                    onClick={() => onSelectRecentRun(r.run_id)}
                    className="w-full text-left p-2 rounded bg-white/[0.02] hover:bg-white/5 border border-transparent hover:border-white/10 transition-all group"
                  >
                    <p className="text-xs text-neutral-300 truncate font-medium group-hover:text-white">
                      {r.user_task}
                    </p>
                    <p className="text-[9px] font-mono text-neutral-400 mt-0.5">
                      {r.run_id.slice(0, 12)}
                    </p>
                  </button>
                ))
              ) : (
                <p className="text-[10px] text-neutral-400 px-1 py-1 italic">
                  {online ? "No recent runs logged" : "Offline"}
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Footer Status & Clear */}
      <div className="p-3 border-t border-white/10 flex items-center justify-between text-xs text-neutral-400">
        {!collapsed ? (
          <>
            <span className="flex items-center gap-1.5 text-[10px] font-mono text-neutral-400">
              <span
                className={`w-2 h-2 rounded-full ${
                  running
                    ? "bg-amber-400 animate-pulse"
                    : online
                    ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]"
                    : "bg-neutral-600"
                }`}
              />
              {running ? "EXECUTING" : online ? "ONLINE" : "OFFLINE"}
            </span>

            <Button
              variant="ghost"
              size="sm"
              disabled={running}
              onClick={onClear}
              className="text-xs text-neutral-400 hover:text-neutral-200 gap-1 p-0 h-auto"
            >
              <Trash2 size={13} />
            </Button>
          </>
        ) : (
          <div
            className={`w-2.5 h-2.5 mx-auto rounded-full ${
              running ? "bg-amber-400 animate-pulse" : online ? "bg-emerald-400" : "bg-neutral-600"
            }`}
            title={running ? "Executing" : online ? "System Online" : "Offline"}
          />
        )}
      </div>
    </aside>
  );
}
