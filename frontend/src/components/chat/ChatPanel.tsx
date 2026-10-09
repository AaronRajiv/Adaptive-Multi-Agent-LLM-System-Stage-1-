import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowUp,
  Bot,
  Check,
  Copy,
  FileText,
  LoaderCircle,
  Paperclip,
  Plus,
  RotateCcw,
  Sparkles,
  User,
  X,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { TypewriterMarkdown } from "@/components/chat/TypewriterMarkdown";
import { ThinkingAccordion } from "@/components/chat/ThinkingAccordion";
import type { ChatMessage } from "@/types/execution";

interface Props {
  messages: ChatMessage[];
  task: string;
  setTask: (task: string) => void;
  onSubmit: (files?: File[]) => void;
  running: boolean;
  onClear: () => void;
  presets: { label: string; task: string }[];
}

export function ChatPanel({
  messages,
  task,
  setTask,
  onSubmit,
  running,
  presets,
}: Props) {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, messages[messages.length - 1]?.content, messages[messages.length - 1]?.events?.length, running]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      setAttachedFiles((prev) => [...prev, ...newFiles]);
    }
  };

  const removeFile = (index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmitForm = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (running || !task.trim()) return;
    onSubmit(attachedFiles);
    setAttachedFiles([]);
  };

  return (
    <div className="flex flex-col h-full bg-black text-neutral-200">
      {/* Scrollable Messages Feed */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
        {/* Welcome Screen */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center p-6 text-neutral-400 space-y-4 my-auto">
            <div className="w-12 h-12 rounded-full bg-neutral-900 border border-white/10 flex items-center justify-center text-emerald-400 shadow-md">
              <Sparkles size={22} />
            </div>
            <div className="space-y-1.5 max-w-md">
              <h3 className="text-sm font-medium text-neutral-200">
                Resource-Aware Multi-Agent System
              </h3>
              <p className="text-xs text-neutral-400 leading-relaxed">
                Submit an analytical task or attach context files (+). Multi-turn chat memory and live node architecture state visualization enabled.
              </p>
            </div>

            {/* Presets */}
            <div className="flex flex-wrap gap-2 justify-center pt-2">
              {presets.map((p) => (
                <Button
                  key={p.label}
                  variant="outline"
                  size="sm"
                  className="text-xs bg-neutral-900/90 border-white/10 hover:border-white/20 text-neutral-300 rounded-lg"
                  onClick={() => setTask(p.task)}
                >
                  {p.label}
                  <ArrowRight size={12} className="ml-1 text-neutral-400" />
                </Button>
              ))}
            </div>
          </div>
        )}

        {/* Render Conversation Thread Messages */}
        {messages.map((msg) =>
          msg.role === "user" ? (
            <div key={msg.id} className="flex items-start gap-3 max-w-3xl mx-auto">
              <div className="w-7 h-7 rounded-full bg-neutral-800 border border-white/10 flex items-center justify-center flex-shrink-0 text-neutral-300">
                <User size={13} />
              </div>
              <div className="flex-1 space-y-2">
                <div className="bg-neutral-900/90 border border-white/10 rounded-xl p-3.5 text-xs text-neutral-200 leading-relaxed shadow-sm">
                  {msg.content}
                </div>
                {msg.files && msg.files.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {msg.files.map((file, i) => (
                      <span
                        key={i}
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[10px] text-neutral-300 font-mono"
                      >
                        <FileText size={10} className="text-emerald-400" />
                        {file.name}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div key={msg.id} className="flex items-start gap-3 max-w-3xl mx-auto">
              <div className="w-7 h-7 rounded-full bg-neutral-900 border border-emerald-500/30 flex items-center justify-center flex-shrink-0 text-emerald-400">
                <Bot size={13} />
              </div>
              <div className="flex-1 space-y-3">
                {/* AI Thinking / Reasoning Accordion */}
                <ThinkingAccordion events={msg.events || []} running={Boolean(msg.running)} />

                {/* Error Notice */}
                {msg.error && (
                  <div className="p-3.5 rounded-xl bg-red-950/40 border border-red-500/30 text-xs text-red-300 space-y-2">
                    <div className="flex items-center gap-2 text-red-400 font-semibold">
                      <XCircle size={15} />
                      <span>Execution Failed</span>
                    </div>
                    <p className="text-neutral-300 leading-normal">{msg.error}</p>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-xs text-red-400 hover:text-red-300 p-0 h-auto"
                      onClick={() => handleSubmitForm()}
                    >
                      <RotateCcw size={12} className="mr-1" /> Retry Execution
                    </Button>
                  </div>
                )}

                {/* Markdown Response */}
                {msg.content && (
                  <div className="bg-neutral-950 border border-white/10 rounded-xl p-4 text-xs text-neutral-200 space-y-3 relative shadow-md">
                    <div className="flex items-center justify-between pb-2 border-b border-white/10 text-[10px] text-neutral-400 font-mono">
                      <span>
                        {msg.executionStrategy === "DIRECT"
                          ? "DIRECT RESPONSE"
                          : msg.executionStrategy === "SINGLE_AGENT"
                            ? "RESPONSE"
                            : "FINAL SYNTHESIS ANSWER"}
                      </span>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-6 w-6 text-neutral-400 hover:text-white"
                        onClick={async () => {
                          try {
                            await navigator.clipboard.writeText(msg.content);
                            setCopiedId(msg.id);
                            setTimeout(() => setCopiedId(null), 2000);
                          } catch {
                            setCopiedId(null);
                          }
                        }}
                      >
                        {copiedId === msg.id ? (
                          <Check size={12} className="text-emerald-400" />
                        ) : (
                          <Copy size={12} />
                        )}
                      </Button>
                    </div>
                    <TypewriterMarkdown content={msg.content} speedMs={10} />
                  </div>
                )}
              </div>
            </div>
          )
        )}

        <div ref={chatBottomRef} />
      </div>

      {/* Input Command Area */}
      <form className="p-3.5 border-t border-white/10 bg-neutral-950/90" onSubmit={handleSubmitForm}>
        {/* Attached Files Chips */}
        {attachedFiles.length > 0 && (
          <div className="flex flex-wrap gap-2 max-w-3xl mx-auto mb-2 px-1">
            {attachedFiles.map((file, idx) => (
              <span
                key={idx}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-neutral-900 border border-white/10 text-[11px] text-neutral-200 font-mono"
              >
                <Paperclip size={12} className="text-emerald-400" />
                <span className="truncate max-w-[140px]">{file.name}</span>
                <button
                  type="button"
                  onClick={() => removeFile(idx)}
                  className="text-neutral-500 hover:text-red-400 p-0.5"
                >
                  <X size={12} />
                </button>
              </span>
            ))}
          </div>
        )}

        {/* Hidden File Picker Input */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          multiple
          accept=".pdf,.txt,.md,.json,.doc,.docx"
          className="hidden"
        />

        {/* Input Bar */}
        <div className="flex items-center gap-2 bg-neutral-900 border border-white/10 focus-within:border-white/30 rounded-2xl px-3 py-2 transition-all max-w-3xl mx-auto shadow-inner">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={running}
            className="w-8 h-8 rounded-full bg-white/5 hover:bg-white/10 flex items-center justify-center text-neutral-400 hover:text-white transition-colors flex-shrink-0 disabled:opacity-40"
            title="Attach file context for RAG"
          >
            <Plus size={16} />
          </button>

          <input
            type="text"
            className="flex-1 bg-transparent border-0 outline-none text-xs text-neutral-100 placeholder:text-neutral-500"
            placeholder="Enter follow-up or complex analytical task..."
            value={task}
            disabled={running}
            onChange={(e) => setTask(e.target.value)}
          />

          <button
            type="submit"
            disabled={running || !task.trim()}
            className="w-8 h-8 rounded-full bg-white text-black hover:bg-neutral-200 flex items-center justify-center transition-all disabled:opacity-30 disabled:hover:bg-white flex-shrink-0"
            title="Send prompt"
          >
            {running ? (
              <LoaderCircle size={15} className="animate-spin text-black" />
            ) : (
              <ArrowUp size={16} className="text-black" />
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
