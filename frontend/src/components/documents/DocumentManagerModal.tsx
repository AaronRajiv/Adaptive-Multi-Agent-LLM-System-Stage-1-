import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  FileText,
  UploadCloud,
  Trash2,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Loader2,
  X,
  Database,
  Layers,
  Hash,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { fetchDocuments, uploadDocument, deleteDocument } from "@/services/api";
import type { DocumentMetadata } from "@/types/api";

interface Props {
  open: boolean;
  onClose: () => void;
  selectedDocIds: string[];
  onToggleSelectDoc: (docId: string) => void;
}

export function DocumentManagerModal({
  open,
  onClose,
  selectedDocIds,
  onToggleSelectDoc,
}: Props) {
  const queryClient = useQueryClient();
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [customLabel, setCustomLabel] = useState("");

  const { data: documents = [], isLoading, refetch } = useQuery({
    queryKey: ["documents-list"],
    queryFn: fetchDocuments,
    enabled: open,
    refetchInterval: open ? 3000 : false,
  });

  const deleteMutation = useMutation({
    mutationFn: (docId: string) => deleteDocument(docId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents-list"] });
    },
  });

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadError("");

    try {
      await uploadDocument(file, customLabel.trim() || undefined);
      setCustomLabel("");
      queryClient.invalidateQueries({ queryKey: ["documents-list"] });
    } catch (err: any) {
      setUploadError(err?.message || "Failed to upload document");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="w-full max-w-2xl bg-neutral-950 border border-white/10 rounded-xl shadow-2xl flex flex-col max-h-[85vh] overflow-hidden text-neutral-200">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/10 bg-neutral-900/60">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-950/80 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <Database size={16} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">Document Manager & RAG Knowledge Store</h3>
              <p className="text-[11px] text-neutral-400">
                Pre-ingest regulation documents into the vector store for grounded retrieval
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-neutral-400 hover:text-white p-1 rounded-md hover:bg-white/5 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Upload Zone */}
        <div className="p-5 border-b border-white/10 bg-neutral-900/30 space-y-3">
          <div className="flex items-center gap-3">
            <input
              type="text"
              placeholder="Optional label (e.g., FIA 2026 Regulations)"
              value={customLabel}
              onChange={(e) => setCustomLabel(e.target.value)}
              className="flex-1 bg-neutral-900 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-neutral-200 placeholder:text-neutral-500 outline-none focus:border-emerald-500/50"
            />
            <label className="cursor-pointer">
              <input
                type="file"
                accept=".pdf,.txt,.md,.json"
                onChange={handleFileUpload}
                disabled={uploading}
                className="hidden"
              />
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium transition-colors cursor-pointer shadow-sm">
                {uploading ? (
                  <>
                    <Loader2 size={13} className="animate-spin" /> Ingesting...
                  </>
                ) : (
                  <>
                    <UploadCloud size={14} /> Upload Document
                  </>
                )}
              </span>
            </label>
          </div>
          {uploadError && (
            <div className="text-xs text-red-400 flex items-center gap-1.5">
              <AlertCircle size={13} /> {uploadError}
            </div>
          )}
        </div>

        {/* Document List */}
        <div className="flex-1 overflow-y-auto p-5 space-y-2.5">
          <div className="flex items-center justify-between text-[11px] text-neutral-400 font-mono pb-1">
            <span>STORED DOCUMENTS ({documents.length})</span>
            <button
              onClick={() => refetch()}
              className="flex items-center gap-1 hover:text-white transition-colors"
            >
              <RefreshCw size={11} className={isLoading ? "animate-spin" : ""} /> Refresh
            </button>
          </div>

          {documents.length === 0 ? (
            <div className="text-center py-10 border border-dashed border-white/10 rounded-lg text-neutral-500 text-xs">
              No documents stored yet. Upload FIA or technical PDFs to enable grounded RAG.
            </div>
          ) : (
            documents.map((doc: DocumentMetadata) => {
              const isSelected = selectedDocIds.includes(doc.document_id);
              return (
                <div
                  key={doc.document_id}
                  className={`p-3 rounded-lg border transition-all flex items-center justify-between gap-3 ${
                    isSelected
                      ? "bg-emerald-950/30 border-emerald-500/40"
                      : "bg-neutral-900/60 border-white/5 hover:border-white/10"
                  }`}
                >
                  <div className="flex items-start gap-3 min-w-0">
                    <button
                      type="button"
                      onClick={() => onToggleSelectDoc(doc.document_id)}
                      className={`w-5 h-5 rounded mt-0.5 flex items-center justify-center transition-colors border ${
                        isSelected
                          ? "bg-emerald-500 border-emerald-500 text-black"
                          : "border-white/20 hover:border-white/40"
                      }`}
                      title={isSelected ? "Deselect document" : "Select for active retrieval"}
                    >
                      {isSelected && <CheckCircle size={13} className="stroke-[3]" />}
                    </button>

                    <div className="space-y-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-xs text-neutral-100 truncate">
                          {doc.label || doc.filename}
                        </span>
                        <span
                          className={`text-[9.5px] font-mono px-1.5 py-0.2 rounded uppercase ${
                            doc.status === "ready"
                              ? "bg-emerald-950 text-emerald-400 border border-emerald-500/30"
                              : doc.status === "failed"
                              ? "bg-red-950 text-red-400 border border-red-500/30"
                              : "bg-amber-950 text-amber-400 border border-amber-500/30"
                          }`}
                        >
                          {doc.status}
                        </span>
                      </div>

                      <div className="flex flex-wrap items-center gap-3 text-[10px] font-mono text-neutral-400">
                        <span className="flex items-center gap-1">
                          <FileText size={10} /> {doc.filename}
                        </span>
                        <span className="flex items-center gap-1">
                          <Layers size={10} /> {doc.chunk_count} chunks
                        </span>
                        <span className="flex items-center gap-1">
                          <Hash size={10} /> {doc.sha256.slice(0, 8)}...
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 flex-shrink-0">
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => deleteMutation.mutate(doc.document_id)}
                      className="h-7 w-7 text-neutral-500 hover:text-red-400 hover:bg-red-950/30"
                      title="Delete document from knowledge store"
                    >
                      <Trash2 size={13} />
                    </Button>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-white/10 bg-neutral-900/60 flex items-center justify-between text-xs">
          <span className="text-neutral-400 text-[11px]">
            {selectedDocIds.length} document(s) selected for current task filter
          </span>
          <Button size="sm" onClick={onClose} className="bg-white text-black hover:bg-neutral-200 text-xs">
            Done
          </Button>
        </div>
      </div>
    </div>
  );
}
