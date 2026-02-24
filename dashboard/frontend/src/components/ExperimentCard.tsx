import { useNavigate } from 'react-router-dom';
import { useLaunchExperiment, useStopExperiment } from '../hooks/useExperiments';
import type { ExperimentRecord } from '../types';
import { StatusBadge } from './StatusBadge';

export function ExperimentCard({ exp }: { exp: ExperimentRecord }) {
  const navigate = useNavigate();
  const launch = useLaunchExperiment();
  const stop = useStopExperiment();

  const isRunning = exp.status === 'running';
  const canLaunch = exp.status === 'pending' || exp.status === 'completed' || exp.status === 'failed' || exp.status === 'stopped';

  return (
    <div
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: 8,
        padding: 16,
        marginBottom: 12,
        background: '#fff',
        cursor: 'pointer',
      }}
      onClick={() => navigate(`/experiments/${exp.id}`)}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <strong style={{ fontSize: 16 }}>{exp.name}</strong>
          <div style={{ color: '#6b7280', fontSize: 12, marginTop: 2 }}>id: {exp.id}</div>
        </div>
        <StatusBadge status={exp.status} />
      </div>
      <div style={{ marginTop: 8, fontSize: 13, color: '#374151' }}>
        {exp.model.num_layers}L · d_model={exp.model.d_model} · {exp.max_iters} iters ·{' '}
        {exp.device}
      </div>
      <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
        {canLaunch && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              launch.mutate(exp.id);
            }}
            style={{ background: '#2563eb', color: '#fff', border: 'none', borderRadius: 4, padding: '4px 12px', cursor: 'pointer' }}
          >
            Launch
          </button>
        )}
        {isRunning && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              stop.mutate(exp.id);
            }}
            style={{ background: '#dc2626', color: '#fff', border: 'none', borderRadius: 4, padding: '4px 12px', cursor: 'pointer' }}
          >
            Stop
          </button>
        )}
        <button
          onClick={(e) => {
            e.stopPropagation();
            navigate(`/experiments/${exp.id}`);
          }}
          style={{ background: '#f3f4f6', border: 'none', borderRadius: 4, padding: '4px 12px', cursor: 'pointer' }}
        >
          Details
        </button>
      </div>
    </div>
  );
}
