import React from 'react';
import { Layers } from 'lucide-react';
import type { SystemStatusResponse } from '../types';

interface HeaderProps {
  status: SystemStatusResponse | null;
}

export const Header: React.FC<HeaderProps> = ({ status }) => {
  const isOnline = status !== null;
  const isConfigured = status?.configured ?? false;
  const isMock = status?.provider.toLowerCase() === 'mock';

  let statusClass = 'offline';
  let statusText = 'Connecting to backend...';

  if (isOnline) {
    if (isConfigured) {
      statusClass = isMock ? 'warning' : 'online';
      statusText = isMock ? 'Mock Provider Active' : `${status.provider.toUpperCase()} (${status.model})`;
    } else {
      statusClass = 'warning';
      statusText = 'API Key Missing';
    }
  }

  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="brand-badge-row">
          <span className="stage-badge">
            <Layers size={13} style={{ display: 'inline', marginRight: '5px', verticalAlign: '-1px' }} />
            Stage 1: Fixed Baseline Pipeline
          </span>
        </div>
        <h1 className="app-title">Adaptive Multi-Agent LLM System (early)</h1>
        <p className="app-subtitle">
          Task Decomposition, Parametric Research, Analytical Synthesis & Quality Evaluation
        </p>
      </div>

      <div className="status-pill">
        <span className={`status-indicator-dot ${statusClass}`} />
        <span style={{ fontWeight: 500 }}>{statusText}</span>
        {status?.evaluator_threshold && (
          <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', borderLeft: '1px solid rgba(255,255,255,0.1)', paddingLeft: '0.5rem' }}>
            Eval Threshold: {status.evaluator_threshold}%
          </span>
        )}
      </div>
    </header>
  );
};
