import type { RunResponse, SystemStatusResponse, ApiErrorDetail } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export class ApiError extends Error {
  detail: ApiErrorDetail | string;
  statusCode: number;

  constructor(message: string, statusCode: number, detail: ApiErrorDetail | string) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.detail = detail;
  }
}

export async function fetchSystemStatus(): Promise<SystemStatusResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/status`);
    if (!res.ok) {
      throw new Error(`Status check failed: HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err: unknown) {
    return {
      configured: false,
      provider: 'unknown',
      model: 'unknown',
      evaluator_threshold: 80,
      message: err instanceof Error ? `Cannot reach backend at ${API_BASE_URL}: ${err.message}` : 'Backend is unreachable',
    };
  }
}

export async function executePipeline(task: string): Promise<RunResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ task }),
    });
  } catch (networkErr: unknown) {
    throw new ApiError(
      networkErr instanceof Error ? networkErr.message : 'Network communication failed',
      0,
      'Backend is unreachable. Please verify the FastAPI server is running on port 8000.'
    );
  }

  const data = await res.json();

  if (!res.ok) {
    const detail = data?.detail;
    let message = `Pipeline execution failed (HTTP ${res.status})`;
    if (typeof detail === 'string') {
      message = detail;
    } else if (detail?.message) {
      message = detail.message;
    }
    throw new ApiError(message, res.status, detail);
  }

  return data as RunResponse;
}

export async function fetchRecentRuns(): Promise<RunResponse[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/runs`);
    if (!res.ok) return [];
    return await res.json();
  } catch {
    return [];
  }
}
