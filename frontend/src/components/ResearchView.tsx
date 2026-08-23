import React from 'react';
import type { ResearchResult } from '../types';
import { BookOpen, Info } from 'lucide-react';

interface ResearchViewProps {
  researchResults: ResearchResult[];
}

export const ResearchView: React.FC<ResearchViewProps> = ({ researchResults }) => {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <BookOpen size={18} color="var(--accent-cyan)" />
          Research Agent Findings ({researchResults.length} Subtask Investigations)
        </h3>
      </div>

      <div className="parametric-notice">
        <Info size={14} style={{ display: 'inline', marginRight: '5px', verticalAlign: '-2px' }} />
        <strong>Parametric Knowledge Baseline:</strong> Stage 1 Research Agent uses the LLM's parametric knowledge. External retrieval/RAG and web search will be introduced in a later stage.
      </div>

      <div className="research-items">
        {researchResults.map((res) => (
          <div key={res.subtask_id} className="research-card">
            <div className="research-card-header">
              <span className="subtask-id-badge">{res.subtask_id}</span>
              <strong style={{ fontSize: '0.92rem', color: '#f1f5f9' }}>{res.subtask_description}</strong>
            </div>
            <div className="research-card-body">{res.findings}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
