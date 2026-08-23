import React from 'react';
import { Cpu } from 'lucide-react';

interface AnalystViewProps {
  analysis: string;
}

export const AnalystView: React.FC<AnalystViewProps> = ({ analysis }) => {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Cpu size={18} color="var(--accent-purple)" />
          Analyst Agent Synthesis & Reasoning
        </h3>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Multi-dimensional synthesis across research subtasks
        </span>
      </div>

      <div className="glass-panel" style={{ padding: '1.25rem', background: 'var(--bg-subtle)' }}>
        <div className="prose-content">{analysis}</div>
      </div>
    </div>
  );
};
