import React, { useState, useEffect } from 'react';
import {
  Header,
} from './components/Header';
import { ConfigBanner } from './components/ConfigBanner';
import { TaskInput } from './components/TaskInput';
import { PipelineTimeline } from './components/PipelineTimeline';
import { SubtaskView } from './components/SubtaskView';
import { ResearchView } from './components/ResearchView';
import { AnalystView } from './components/AnalystView';
import { EvaluatorView } from './components/EvaluatorView';
import { FinalAnswerView } from './components/FinalAnswerView';
import { ExecutionTraceLog } from './components/ExecutionTraceLog';
import type {
  RunResponse,
  SystemStatusResponse,
} from './types';
import {
  fetchSystemStatus,
  executePipeline,
  ApiError,
} from './services/api';
import {
  ListTree,
  BookOpen,
  Cpu,
  Award,
  Activity,
  AlertTriangle,
  Layers,
} from 'lucide-react';

type TabKey = 'subtasks' | 'research' | 'analyst' | 'evaluator' | 'trace';

export const App: React.FC = () => {
  const [task, setTask] = useState<string>('');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [runResult, setRunResult] = useState<RunResponse | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabKey>('subtasks');

  const checkStatus = async () => {
    const status = await fetchSystemStatus();
    setSystemStatus(status);
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleRun = async () => {
    if (!task.trim() || isRunning) return;

    setError(null);
    setIsRunning(true);

    try {
      const response = await executePipeline(task);
      setRunResult(response);
      setActiveTab('subtasks');
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        if (typeof err.detail === 'object' && err.detail?.message) {
          setError(err.detail.message);
        } else {
          setError(err.message);
        }
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred while executing the baseline pipeline.');
      }
    } finally {
      setIsRunning(false);
      checkStatus();
    }
  };

  const handleClear = () => {
    setTask('');
    setRunResult(null);
    setError(null);
  };

  const totalDurationMs = runResult?.execution_trace.reduce(
    (acc, curr) => acc + curr.duration_ms,
    0
  );

  return (
    <div className="app-container">
      {/* Header */}
      <Header status={systemStatus} />

      {/* Configuration Status Notice */}
      <ConfigBanner status={systemStatus} />

      {/* Error Alert Box */}
      {error && (
        <div className="alert-box danger">
          <AlertTriangle size={20} />
          <div className="alert-content">
            <h4>Execution Error</h4>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Input Section */}
      <TaskInput
        task={task}
        setTask={setTask}
        onRun={handleRun}
        isRunning={isRunning}
        onClear={handleClear}
      />

      {/* Pipeline Stage Visualizer */}
      {(isRunning || runResult) && (
        <section className="glass-panel">
          <PipelineTimeline
            isRunning={isRunning}
            traces={runResult?.execution_trace || []}
            totalDurationMs={totalDurationMs}
          />
        </section>
      )}

      {/* Final Synthesized Output */}
      {runResult && (
        <FinalAnswerView
          finalAnswer={runResult.final_answer}
          runId={runResult.run_id}
          createdAt={runResult.created_at}
        />
      )}

      {/* Multi-Agent Intermediate Outputs / Observable Trace Panel */}
      {runResult && (
        <section className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
            <span className="section-label">
              <Layers size={16} color="var(--accent-primary)" />
              Agent Intermediate Outputs & Execution Trace
            </span>
          </div>

          <div className="tabs-header">
            <button
              type="button"
              className={`tab-btn ${activeTab === 'subtasks' ? 'active' : ''}`}
              onClick={() => setActiveTab('subtasks')}
            >
              <ListTree size={16} />
              1. Planner Subtasks ({runResult.subtasks.length})
            </button>

            <button
              type="button"
              className={`tab-btn ${activeTab === 'research' ? 'active' : ''}`}
              onClick={() => setActiveTab('research')}
            >
              <BookOpen size={16} />
              2. Research Findings ({runResult.research_results.length})
            </button>

            <button
              type="button"
              className={`tab-btn ${activeTab === 'analyst' ? 'active' : ''}`}
              onClick={() => setActiveTab('analyst')}
            >
              <Cpu size={16} />
              3. Analyst Synthesis
            </button>

            <button
              type="button"
              className={`tab-btn ${activeTab === 'evaluator' ? 'active' : ''}`}
              onClick={() => setActiveTab('evaluator')}
            >
              <Award size={16} />
              4. Evaluator Audit ({runResult.evaluation.score}/100)
            </button>

            <button
              type="button"
              className={`tab-btn ${activeTab === 'trace' ? 'active' : ''}`}
              onClick={() => setActiveTab('trace')}
            >
              <Activity size={16} />
              5. Execution Telemetry
            </button>
          </div>

          <div style={{ marginTop: '0.5rem' }}>
            {activeTab === 'subtasks' && <SubtaskView subtasks={runResult.subtasks} />}
            {activeTab === 'research' && <ResearchView researchResults={runResult.research_results} />}
            {activeTab === 'analyst' && <AnalystView analysis={runResult.analysis} />}
            {activeTab === 'evaluator' && (
              <EvaluatorView
                evaluation={runResult.evaluation}
                threshold={systemStatus?.evaluator_threshold || 80}
              />
            )}
            {activeTab === 'trace' && (
              <ExecutionTraceLog
                traces={runResult.execution_trace}
                runId={runResult.run_id}
              />
            )}
          </div>
        </section>
      )}
    </div>
  );
};

export default App;
