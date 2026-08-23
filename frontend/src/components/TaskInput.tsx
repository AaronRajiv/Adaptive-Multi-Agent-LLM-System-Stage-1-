import React from 'react';
import { Play, Sparkles, RefreshCw } from 'lucide-react';

interface TaskInputProps {
  task: string;
  setTask: (t: string) => void;
  onRun: () => void;
  isRunning: boolean;
  onClear: () => void;
}

const DEMO_TASKS = [
  {
    label: 'EV vs Petrol Vehicle Comparison',
    prompt: 'Compare electric vehicles and petrol vehicles for a college student considering cost, maintenance, environmental impact and practicality.',
  },
  {
    label: 'Renewable Energy Investment (India)',
    prompt: 'Analyze whether India should increase investment in renewable energy considering economic, environmental and policy factors.',
  },
];

export const TaskInput: React.FC<TaskInputProps> = ({
  task,
  setTask,
  onRun,
  isRunning,
  onClear,
}) => {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      if (task.trim() && !isRunning) {
        onRun();
      }
    }
  };

  return (
    <section className="glass-panel input-section">
      <div className="input-header">
        <span className="section-label">
          <Sparkles size={16} color="var(--accent-primary)" />
          User Task Specification
        </span>

        <div className="demo-presets">
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Demo Presets:</span>
          {DEMO_TASKS.map((demo, idx) => (
            <button
              key={idx}
              type="button"
              className="demo-btn"
              onClick={() => setTask(demo.prompt)}
              disabled={isRunning}
            >
              {demo.label}
            </button>
          ))}
        </div>
      </div>

      <textarea
        className="task-textarea"
        placeholder="Enter a complex analytical task here... (e.g. multi-dimensional comparisons, strategic policy evaluations, technology feasibility studies)"
        value={task}
        onChange={(e) => setTask(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isRunning}
        rows={4}
      />

      <div className="input-actions">
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <button
            type="button"
            className="btn-secondary"
            onClick={onClear}
            disabled={isRunning || !task}
          >
            Clear
          </button>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Press <kbd style={{ background: 'rgba(255,255,255,0.06)', padding: '0.1rem 0.35rem', borderRadius: '4px' }}>⌘ + Enter</kbd> to run
          </span>
        </div>

        <button
          type="button"
          className="primary-btn"
          onClick={onRun}
          disabled={isRunning || task.trim().length < 3}
        >
          {isRunning ? (
            <>
              <RefreshCw size={17} className="spinner" />
              Executing Baseline Pipeline...
            </>
          ) : (
            <>
              <Play size={17} />
              Run Fixed Pipeline
            </>
          )}
        </button>
      </div>
    </section>
  );
};
