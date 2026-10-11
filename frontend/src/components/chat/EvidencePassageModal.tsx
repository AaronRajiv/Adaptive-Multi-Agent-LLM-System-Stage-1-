import { useEffect, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { fetchEvidencePassage, type EvidencePassage } from "@/services/api";
import { sanitizeHttpUrl } from "@/services/activityReducer";
import { FileText, Globe, ExternalLink, AlertTriangle, Loader2 } from "lucide-react";

interface Props {
  runId?: string;
  evidenceId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

export function EvidencePassageModal({ runId, evidenceId, isOpen, onClose }: Props) {
  const [loading, setLoading] = useState(false);
  const [passage, setPassage] = useState<EvidencePassage | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen || !evidenceId || !runId) {
      setPassage(null);
      setError(null);
      return;
    }

    let active = true;
    setLoading(true);
    setError(null);

    fetchEvidencePassage(runId, evidenceId)
      .then((data) => {
        if (!active) return;
        setPassage(data);
        setLoading(false);
      })
      .catch((err) => {
        if (!active) return;
        setError(err.message || "Failed to load evidence passage.");
        setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [runId, evidenceId, isOpen]);

  const safeUrl = sanitizeHttpUrl(passage?.url);

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl bg-neutral-950 border border-white/10 text-neutral-200 p-6 rounded-2xl shadow-2xl">
        <DialogHeader className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              {passage?.source_type === "web" ? (
                <Globe size={18} className="text-cyan-400 flex-shrink-0" />
              ) : (
                <FileText size={18} className="text-emerald-400 flex-shrink-0" />
              )}
              <DialogTitle className="text-sm font-semibold text-neutral-100 truncate">
                {passage?.title || evidenceId || "Evidence Passage"}
              </DialogTitle>
            </div>
            {passage?.citation && (
              <Badge variant="outline" className="text-[10px] font-mono border-white/20 text-neutral-300">
                {passage.citation}
              </Badge>
            )}
          </div>
          <DialogDescription className="text-xs text-neutral-400">
            {passage?.source_type === "web"
              ? "Verbatim content extracted from online technical reference."
              : "Verbatim regulatory article passage retrieved from document repository."}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex flex-col items-center justify-center p-8 space-y-2 text-neutral-400">
            <Loader2 size={24} className="animate-spin text-emerald-400" />
            <span className="text-xs">Loading stored evidence passage...</span>
          </div>
        ) : error ? (
          <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-500/30 text-amber-200 space-y-2 my-2">
            <div className="flex items-center gap-2 text-amber-400 font-semibold text-xs">
              <AlertTriangle size={15} />
              <span>Unverified Citation</span>
            </div>
            <p className="text-xs text-neutral-300 leading-relaxed">
              Citation <code className="text-amber-300 bg-black/40 px-1 py-0.5 rounded">{evidenceId}</code> did not resolve to an item in the verified evidence ledger.
            </p>
          </div>
        ) : passage ? (
          <div className="space-y-4 my-2">
            {/* Metadata Bar */}
            <div className="flex flex-wrap items-center gap-3 p-2.5 rounded-lg bg-neutral-900/80 border border-white/5 text-[11px] text-neutral-300 font-mono">
              {passage.article_path && (
                <span>Article: <strong className="text-emerald-400">{passage.article_path}</strong></span>
              )}
              {passage.page_start !== undefined && (
                <span>Page: <strong className="text-neutral-200">{passage.page_start}{passage.page_end && passage.page_end !== passage.page_start ? `-${passage.page_end}` : ""}</strong></span>
              )}
              {passage.document_label && (
                <span className="truncate max-w-[200px]" title={passage.document_label}>Doc: {passage.document_label}</span>
              )}
              {safeUrl && (
                <a
                  href={safeUrl}
                  target="_blank"
                  rel="noopener noreferrer nofollow"
                  className="inline-flex items-center gap-1 text-cyan-400 hover:text-cyan-300 hover:underline ml-auto"
                >
                  Visit Link <ExternalLink size={11} />
                </a>
              )}
            </div>

            {/* Verbatim Passage Text - rendered inertly as plain text */}
            <div className="space-y-1.5">
              <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">Verbatim Passage</span>
              <div className="p-3.5 max-h-72 overflow-y-auto rounded-xl bg-black border border-white/10 text-xs text-neutral-200 font-mono leading-relaxed whitespace-pre-wrap select-text">
                {passage.text || "No text available."}
              </div>
            </div>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
