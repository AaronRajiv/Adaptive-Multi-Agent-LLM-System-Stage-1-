import React from 'react';
import type { EvaluationResult } from '../types';
import { Award, CheckCircle2, XCircle } from 'lucide-react';

interface EvaluatorViewProps {
  evaluation: EvaluationResult;
  threshold?: number;
}

export const EvaluatorView: React.FC<EvaluatorViewProps> = ({ evaluation, threshold = 80 }) => {
  const isPass = evaluation.status === 'PASS';

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Award size={18} color="var(--accent-amber)" />
          Evaluator Agent Quality Audit
        </h3>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Assessed on Correctness, Completeness, Relevance, & Clarity (Threshold: {threshold}%)
        </span>
      </div>

      <div className="evaluator-container">
        <div className="score-display-card">
          <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
            Quality Score
          </span>
          <div className={`score-number ${isPass ? 'pass' : 'fail'}`}>
            {evaluation.score}
            <span style={{ fontSize: '1.2rem', color: 'var(--text-muted)', fontWeight: 400 }}>/100</span>
          </div>
          <span className={`status-badge-lg ${isPass ? 'pass' : 'fail'}`}>
            {isPass ? (
              <>
                <CheckCircle2 size={15} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-2px' }} />
                STATUS: PASS
              </>
            ) : (
              <>
                <XCircle size={15} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-2px' }} />
                STATUS: FAIL
              </>
            )}
          </span>
        </div>

        <div className="evaluator-feedback-card">
          <h4>Evaluator Critique & Feedback</h4>
          <p style={{ color: '#e2e8f0', fontSize: '0.94rem', lineHeight: '1.7', whiteSpace: 'pre-wrap' }}>
            {evaluation.feedback}
          </p>
        </div>
      </div>
    </div>
  );
};
