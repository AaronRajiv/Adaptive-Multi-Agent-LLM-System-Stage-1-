import type {
  RunResponse,
  SystemStatusResponse,
  ApiErrorDetail,
  DocumentPayload,
  DocumentMetadata,
  DocumentUploadResponse,
  ChatMessagePayload,
} from "../types/api";

export const API_BASE_URL = import.meta.env["VITE_API_BASE_URL"] || "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public detail: ApiErrorDetail | string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request(path: string, init?: RequestInit, timeout = 15000): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(`${API_BASE_URL}${path}`, { ...init, signal: controller.signal });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      const detail = data?.detail;
      const message =
        typeof detail === "string"
          ? detail
          : detail?.message || `Request failed (HTTP ${res.status})`;
      throw new ApiError(message, res.status, detail || message);
    }
    if (data === null)
      throw new ApiError(
        "The backend returned an empty or invalid response.",
        res.status,
        "Invalid response",
      );
    return data;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    const message =
      error instanceof Error && error.name === "AbortError"
        ? "The request timed out. The backend may still be processing this task."
        : "Backend unavailable. Verify the API server and connection settings.";
    throw new ApiError(message, 0, message);
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchSystemStatus(): Promise<SystemStatusResponse> {
  try {
    return (await request("/api/status", undefined, 8000)) as SystemStatusResponse;
  } catch (error) {
    return {
      configured: false,
      provider: "unknown",
      model: "unknown",
      evaluator_threshold: 80,
      message: error instanceof Error ? error.message : "Backend unavailable",
    };
  }
}

export async function uploadDocument(
  file: File,
  label?: string
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (label) {
    formData.append("label", label);
  }

  const res = await fetch(`${API_BASE_URL}/api/documents`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => null);
    throw new ApiError(
      errorData?.detail || `Document upload failed (HTTP ${res.status})`,
      res.status,
      errorData?.detail || "Upload error"
    );
  }
  return (await res.json()) as DocumentUploadResponse;
}

export async function fetchDocumentStatus(
  documentId: string
): Promise<DocumentMetadata> {
  return (await request(`/api/documents/${documentId}/status`, undefined, 8000)) as DocumentMetadata;
}

export async function fetchDocuments(): Promise<DocumentMetadata[]> {
  try {
    const data = await request("/api/documents", undefined, 8000);
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

export async function deleteDocument(documentId: string): Promise<void> {
  await request(`/api/documents/${documentId}`, { method: "DELETE" }, 8000);
}

export async function cancelRun(runId: string): Promise<{ run_id: string; status: string; message: string }> {
  return (await request(`/api/runs/${runId}/cancel`, { method: "POST" }, 8000)) as {
    run_id: string;
    status: string;
    message: string;
  };
}

export async function fetchRunMetrics(runId: string): Promise<any> {
  return await request(`/api/runs/${runId}/metrics`, undefined, 8000);
}

export async function fetchRun(runId: string): Promise<any> {
  return await request(`/api/runs/${runId}`, undefined, 8000);
}

export async function executePipeline(
  task: string,
  runId?: string,
  documents?: DocumentPayload[],
  history?: ChatMessagePayload[],
  documentIds?: string[],
  asyncRun = true
): Promise<RunResponse> {
  const payload = {
    task,
    async_run: asyncRun,
    ...(runId ? { run_id: runId } : {}),
    ...(documentIds && documentIds.length > 0 ? { document_ids: documentIds } : {}),
    ...(documents && documents.length > 0 ? { documents } : {}),
    ...(history && history.length > 0 ? { history } : {}),
  };

  if (!asyncRun) {
    return (await request("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })) as RunResponse;
  }

  // 1. Kick off pipeline asynchronously (returns immediately)
  const initRes = (await request(
    "/api/run",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    15000
  )) as { run_id: string; status: string };

  const effectiveRunId = initRes.run_id || runId || "";

  // 2. Poll GET /api/runs/{run_id} until completed
  const pollInterval = 1000;
  const maxAttempts = 300;
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((resolve) => setTimeout(resolve, pollInterval));
    try {
      const runStatus = (await fetchRun(effectiveRunId)) as {
        run_id: string;
        status: string;
        result?: RunResponse;
        error?: string;
      };

      if (runStatus.status === "completed" && runStatus.result) {
        return runStatus.result;
      }
      if (runStatus.status === "failed") {
        throw new ApiError(
          runStatus.error || "Pipeline execution failed.",
          500,
          runStatus.error || "Failed"
        );
      }
    } catch (e) {
      if (e instanceof ApiError && e.statusCode === 404) {
        continue;
      }
      throw e;
    }
  }

  throw new ApiError(
    "Pipeline execution polling timed out after 300 seconds.",
    408,
    "Timeout"
  );
}

export async function fetchRecentRuns(): Promise<RunResponse[]> {
  try {
    const data = await request("/api/runs", undefined, 8000);
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}
