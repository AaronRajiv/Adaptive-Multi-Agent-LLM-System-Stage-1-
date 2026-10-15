import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Cpu,
  Sparkles,
  Check,
  ChevronDown,
  Bot,
} from "lucide-react";
import type { ModelInfo } from "@/types/api";

interface ModelSelectorProps {
  models: ModelInfo[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  disabled?: boolean;
  direction?: "up" | "down" | "auto";
  compact?: boolean;
}

export function ModelSelector({
  models,
  selectedModelId,
  onSelectModel,
  disabled = false,
  direction = "auto",
  compact = false,
}: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const [menuStyle, setMenuStyle] = useState<React.CSSProperties>({});
  const [maxListHeight, setMaxListHeight] = useState<number>(260);

  // Compute exact viewport-bounded position
  const updatePosition = useCallback(() => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    // Responsive width bounded within screen
    const targetWidth = Math.min(340, vw - 24);

    // Clamp horizontal placement: prefer button's left edge, clamp within [12, vw - targetWidth - 12]
    let left = rect.left;
    if (left + targetWidth > vw - 12) {
      left = Math.max(12, vw - 12 - targetWidth);
    }
    if (left < 12) {
      left = 12;
    }

    const spaceBelow = vh - rect.bottom;
    const spaceAbove = rect.top;

    let shouldOpenUp: boolean;
    if (direction === "up") {
      shouldOpenUp = true;
    } else if (direction === "down") {
      shouldOpenUp = false;
    } else {
      // Auto: prefer down unless space below is constrained and space above is greater
      shouldOpenUp = spaceBelow < 280 && spaceAbove > spaceBelow;
    }

    const availableVerticalSpace = shouldOpenUp ? spaceAbove - 24 : spaceBelow - 24;
    // Header ~42px, footer ~32px, padding ~16px -> 90px chrome
    const calculatedListHeight = Math.max(100, Math.min(260, availableVerticalSpace - 90));
    setMaxListHeight(calculatedListHeight);

    const style: React.CSSProperties = {
      position: "fixed",
      width: `${targetWidth}px`,
      left: `${left}px`,
      zIndex: 9999,
    };

    if (shouldOpenUp) {
      style.bottom = `${vh - rect.top + 6}px`;
    } else {
      style.top = `${rect.bottom + 6}px`;
    }

    setMenuStyle(style);
  }, [direction]);

  // Recalculate whenever open state changes or window resizes/scrolls
  useEffect(() => {
    if (isOpen) {
      updatePosition();
      const onResizeOrScroll = () => updatePosition();
      window.addEventListener("resize", onResizeOrScroll);
      window.addEventListener("scroll", onResizeOrScroll, true);
      return () => {
        window.removeEventListener("resize", onResizeOrScroll);
        window.removeEventListener("scroll", onResizeOrScroll, true);
      };
    }
  }, [isOpen, updatePosition]);

  // Dismiss on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as Node;
      if (
        containerRef.current &&
        !containerRef.current.contains(target) &&
        dropdownRef.current &&
        !dropdownRef.current.contains(target)
      ) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [isOpen]);

  // Dismiss on Escape
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setIsOpen(false);
    }
    if (isOpen) {
      document.addEventListener("keydown", handleKeyDown);
      return () => document.removeEventListener("keydown", handleKeyDown);
    }
  }, [isOpen]);

  const activeModel =
    models.find((m) => m.id === selectedModelId) ||
    models.find((m) => m.recommended) ||
    models[0] || {
      id: "nvidia/nemotron-3-super-120b-a12b",
      name: "Nemotron 3 Super 120B",
      provider: "nvidia",
      description: "Flagship NVIDIA 120B MoE Reasoning Model",
      recommended: true,
      badge: "120B Flagship",
    };

  const getProviderTheme = (provider: string) => {
    switch (provider.toLowerCase()) {
      case "nvidia":
        return {
          bg: "bg-emerald-500/10",
          border: "border-emerald-500/30",
          text: "text-emerald-400",
          glow: "group-hover:border-emerald-500/50",
          dot: "bg-emerald-400",
        };
      case "gemini":
        return {
          bg: "bg-sky-500/10",
          border: "border-sky-500/30",
          text: "text-sky-400",
          glow: "group-hover:border-sky-500/50",
          dot: "bg-sky-400",
        };
      case "openai":
      case "meta":
        return {
          bg: "bg-purple-500/10",
          border: "border-purple-500/30",
          text: "text-purple-400",
          glow: "group-hover:border-purple-500/50",
          dot: "bg-purple-400",
        };
      default:
        return {
          bg: "bg-neutral-800",
          border: "border-neutral-700",
          text: "text-neutral-400",
          glow: "group-hover:border-neutral-600",
          dot: "bg-neutral-400",
        };
    }
  };

  const currentTheme = getProviderTheme(activeModel.provider);

  return (
    <div className="relative inline-block text-left" ref={containerRef}>
      {/* Trigger Button */}
      <button
        type="button"
        disabled={disabled}
        onClick={() => setIsOpen((prev) => !prev)}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        title={`Active Model: ${activeModel.name} (${activeModel.id})`}
        className={`group flex items-center gap-1.5 ${
          compact ? "px-2 py-1 text-[11px]" : "px-2.5 py-1 text-xs"
        } rounded-lg border font-medium transition-all ${currentTheme.bg} ${
          currentTheme.border
        } ${currentTheme.glow} hover:bg-white/[0.06] text-neutral-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm`}
      >
        <div className="flex items-center gap-1.5 min-w-0">
          <span
            className={`w-1.5 h-1.5 rounded-full ${currentTheme.dot} shadow-[0_0_6px_currentColor] flex-shrink-0`}
          />
          <Cpu size={compact ? 11 : 12} className={`${currentTheme.text} flex-shrink-0`} />
          <span className="font-semibold text-neutral-100 tracking-tight truncate max-w-[110px] sm:max-w-[160px]">
            {activeModel.name}
          </span>
        </div>

        {activeModel.badge && !compact && (
          <span className="hidden md:inline-flex items-center px-1.5 py-0.2 rounded text-[9px] font-mono uppercase tracking-wider bg-white/5 border border-white/10 text-neutral-300 flex-shrink-0">
            {activeModel.badge}
          </span>
        )}

        <ChevronDown
          size={11}
          className={`text-neutral-400 transition-transform duration-200 flex-shrink-0 ${
            isOpen ? "rotate-180 text-white" : ""
          }`}
        />
      </button>

      {/* Dropdown Menu - Viewport-bounded fixed modal */}
      {isOpen && (
        <div
          ref={dropdownRef}
          role="listbox"
          style={menuStyle}
          className="rounded-xl bg-neutral-950/98 border border-white/20 p-2 shadow-2xl backdrop-blur-2xl ring-1 ring-black/80 animate-in fade-in-0 zoom-in-95 duration-100"
        >
          {/* Header */}
          <div className="px-2.5 py-1.5 border-b border-white/10 flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-neutral-300 font-medium text-xs">
              <Bot size={13} className="text-emerald-400" />
              <span>Select LLM Model</span>
            </div>
            <span className="text-[10px] font-mono text-neutral-400 uppercase">
              {models.length} Options
            </span>
          </div>

          {/* Options List */}
          <div
            style={{ maxHeight: `${maxListHeight}px` }}
            className="overflow-y-auto space-y-1 py-1.5 pr-0.5"
          >
            {models.map((m) => {
              const isSelected = m.id === selectedModelId;
              const theme = getProviderTheme(m.provider);

              return (
                <button
                  key={m.id}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => {
                    onSelectModel(m.id);
                    setIsOpen(false);
                  }}
                  className={`w-full text-left p-2 rounded-lg transition-all flex items-start justify-between gap-2 ${
                    isSelected
                      ? "bg-white/[0.08] border border-white/20 shadow-inner"
                      : "hover:bg-white/[0.04] border border-transparent"
                  }`}
                >
                  <div className="flex items-start gap-2 min-w-0 flex-1">
                    <div
                      className={`w-5 h-5 rounded-md ${theme.bg} ${theme.border} border flex items-center justify-center flex-shrink-0 mt-0.5`}
                    >
                      <Cpu size={11} className={theme.text} />
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="text-xs font-semibold text-neutral-100 truncate">
                          {m.name}
                        </span>
                        {m.recommended && (
                          <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded text-[8px] font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                            <Sparkles size={8} />
                            Recommended
                          </span>
                        )}
                        {m.badge && !m.recommended && (
                          <span className="inline-flex items-center px-1.5 py-0.2 rounded text-[8px] font-mono uppercase tracking-wider bg-white/5 text-neutral-400 border border-white/10">
                            {m.badge}
                          </span>
                        )}
                      </div>

                      <p className="text-[10px] text-neutral-400 leading-snug line-clamp-2 mt-0.5">
                        {m.description}
                      </p>

                      <div className="flex items-center gap-1.5 mt-1 text-[9px] font-mono text-neutral-500">
                        <span className="capitalize">{m.provider}</span>
                        <span>•</span>
                        <span className="truncate max-w-[160px]">{m.id}</span>
                      </div>
                    </div>
                  </div>

                  {isSelected && (
                    <div className="w-4 h-4 rounded-full bg-emerald-500/20 border border-emerald-500/50 flex items-center justify-center text-emerald-400 flex-shrink-0 mt-1">
                      <Check size={10} />
                    </div>
                  )}
                </button>
              );
            })}
          </div>

          {/* Footer */}
          <div className="px-2.5 py-1 border-t border-white/10 flex items-center justify-between text-[9px] text-neutral-400 font-mono">
            <span>Powered by NVIDIA NIM</span>
            <span className="text-emerald-400">Low-latency</span>
          </div>
        </div>
      )}
    </div>
  );
}
