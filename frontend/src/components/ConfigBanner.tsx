import React from 'react';
import { AlertTriangle, Key } from 'lucide-react';
import type { SystemStatusResponse } from '../types';

interface ConfigBannerProps {
  status: SystemStatusResponse | null;
}

export const ConfigBanner: React.FC<ConfigBannerProps> = ({ status }) => {
  if (!status) {
    return (
      <div className="alert-box danger">
        <AlertTriangle size={20} />
        <div className="alert-content">
          <h4>Backend Server Unreachable</h4>
          <p>
            The React frontend cannot connect to <code>http://localhost:8000/api/status</code>. Please start the FastAPI backend using:
          </p>
          <p style={{ marginTop: '0.4rem' }}>
            <code>cd backend && ./venv/bin/uvicorn app.main:app --reload --port 8000</code>
          </p>
        </div>
      </div>
    );
  }

  if (!status.configured) {
    return (
      <div className="alert-box">
        <Key size={20} />
        <div className="alert-content">
          <h4>LLM Provider Not Configured ({status.provider.toUpperCase()})</h4>
          <p>
            {status.message}
          </p>
          <p style={{ marginTop: '0.4rem', color: 'var(--text-secondary)' }}>
            To run with live Gemini: edit <code>backend/.env</code> and set <code>LLM_API_KEY=your_key</code>.<br />
            To run in offline test simulation: set <code>LLM_PROVIDER=mock</code> in <code>backend/.env</code>.
          </p>
        </div>
      </div>
    );
  }

  return null;
};
