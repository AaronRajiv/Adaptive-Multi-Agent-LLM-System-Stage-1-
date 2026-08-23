import React from 'react';
import type { SubTask } from '../types';
import { ListTree } from 'lucide-react';

interface SubtaskViewProps {
  subtasks: SubTask[];
}

export const SubtaskView: React.FC<SubtaskViewProps> = ({ subtasks }) => {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ListTree size={18} color="var(--accent-primary)" />
          Planner Agent Decomposition ({subtasks.length} Subtasks)
        </h3>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Structured Pydantic validation passed
        </span>
      </div>

      <div className="subtasks-grid">
        {subtasks.map((st) => (
          <div key={st.id} className="subtask-card">
            <div className="subtask-top">
              <span className="subtask-id-badge">{st.id}</span>
              <span className={`type-pill ${st.type}`}>{st.type}</span>
            </div>
            <p className="subtask-desc">{st.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
