import React from 'react';
import type { StageTrace } from '../types';
import { Activity, CheckCircle2, XCircle } from 'lucide-react';

interface ExecutionTraceLogProps {
  traces: StageTrace[];
  runId: string;
}

export const ExecutionTraceLog: React.FC<ExecutionTraceLogProps> = ({ traces, runId }) => {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={18} color="var(--accent-emerald)" />
          Observable Execution Trace & Telemetry
        </h3>
        <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
          Run UUID: {runId}
        </span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="trace-table">
          <thead>
            <tr>
              <th>Pipeline Stage</th>
              <th>Agent Component</th>
              <th>Measured Latency</th>
              <th>Status</th>
              <th>Stage Summary / Diagnostics</th>
            </tr>
          </thead>
          <tbody>
            {traces.map((trace, idx) => (
              <tr key={idx}>
                <td>
                  <strong>{trace.stage_name}</strong>
                </td>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: '#94a3b8' }}>
                  {trace.agent_name}
                </td>
                <td className="latency-tag">
                  {(trace.duration_ms / 1000).toFixed(2)}s ({trace.duration_ms.toFixed(0)}ms)
                </td>
                <td>
                  {trace.status === 'SUCCESS' ? (
                    <span style={{ color: 'var(--accent-emerald)', display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '0.8rem', fontWeight: 600 }}>
                      <CheckCircle2 size={13} />
                      SUCCESS
                    </span>
                  ) : (
                    <span style={{ color: 'var(--accent-rose)', display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '0.8rem', fontWeight: 600 }}>
                      <XCircle size={13} />
                      FAILED
                    </span>
                  )}
                </td>
                <td style={{ fontSize: '0.84rem', color: '#cbd5e1' }}>
                  {trace.summary || 'Completed stage successfully.'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
