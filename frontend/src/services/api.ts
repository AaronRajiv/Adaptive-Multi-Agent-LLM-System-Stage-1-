import type { RunResponse, SystemStatusResponse, ApiErrorDetail } from "../types/api";
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
async function request(path: string, init?: RequestInit, timeout = 120000): Promise<unknown> {
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
export async function executePipeline(task: string): Promise<RunResponse> {
  return (await request("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task }),
  })) as RunResponse;
}
export async function fetchRecentRuns(): Promise<RunResponse[]> {
  try {
    const data = await request("/api/runs", undefined, 8000);
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}
