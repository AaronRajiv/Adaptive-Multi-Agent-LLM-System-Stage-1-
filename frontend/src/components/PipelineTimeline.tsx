import React from 'react';
import { Terminal } from 'lucide-react';
import type { StageTrace } from '../types';

interface PipelineTimelineProps {
  isRunning: boolean;
  traces: StageTrace[];
  totalDurationMs?: number;
}

const STAGES = [
  { id: 'Planner', label: '1. Planner Agent', role: 'Task Decomposition' },
  { id: 'Research', label: '2. Research Agent(s)', role: 'Parametric Investigation' },
  { id: 'Analysis', label: '3. Analyst Agent', role: 'Synthesis & Reasoning' },
  { id: 'Evaluation', label: '4. Evaluator Agent', role: 'Quality Scoring' },
  { id: 'Synthesis', label: '5. Synthesizer Agent', role: 'Final Answer Generation' },
];

export const PipelineTimeline: React.FC<PipelineTimelineProps> = ({
  isRunning,
  traces,
  totalDurationMs,
}) => {
  const getTraceForStage = (stageName: string) => {
    return traces.find((t) => t.stage_name.toLowerCase() === stageName.toLowerCase());
  };

  return (
    <div className="pipeline-progress-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="section-label">
          <Terminal size={16} color="var(--accent-cyan)" />
          Fixed Pipeline Workflow Execution
        </span>
        {totalDurationMs !== undefined && (
          <span style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
            Total Pipeline Latency: {(totalDurationMs / 1000).toFixed(2)}s
          </span>
        )}
      </div>

      {isRunning && (
        <div className="running-loader-banner">
          <div className="spinner" />
          <div>
            <strong>Executing Sequential Baseline Pipeline...</strong>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              Processing synchronously on backend: Planner → Research → Analyst → Evaluator → Synthesizer
            </p>
          </div>
        </div>
      )}

      <div className="pipeline-steps-grid">
        {STAGES.map((st) => {
          const trace = getTraceForStage(st.id);
          const isCompleted = trace && trace.status === 'SUCCESS';
          const isFailed = trace && trace.status === 'FAILED';

          return (
            <div
              key={st.id}
              className={`pipeline-step-card ${isCompleted ? 'completed' : ''} ${isRunning ? 'active' : ''}`}
            >
              <div className="step-card-header">
                <span className="step-name">{st.label}</span>
                {isCompleted && (
                  <span className="step-badge success">
                    ✓ {(trace.duration_ms / 1000).toFixed(2)}s
                  </span>
                )}
                {isFailed && (
                  <span className="step-badge" style={{ background: 'rgba(244, 63, 94, 0.2)', color: '#fda4af' }}>
                    FAILED
                  </span>
                )}
                {!trace && !isRunning && (
                  <span className="step-badge">Pending</span>
                )}
              </div>
              <span className="step-agent-title">{st.role}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
