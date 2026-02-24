import { useNavigate, useParams } from 'react-router-dom';
import { GeneratePanel } from '../components/GeneratePanel';
import { LossCurveChart } from '../components/LossCurveChart';
import { LogViewer } from '../components/LogViewer';
import { StatusBadge } from '../components/StatusBadge';
import {
  useCheckpoints,
  useDeleteExperiment,
  useExperiment,
  useLaunchExperiment,
  useLog,
  useStopExperiment,
} from '../hooks/useExperiments';
import { useLiveMetrics } from '../hooks/useLiveMetrics';

export function ExperimentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { data: exp, isLoading } = useExperiment(id!);
  const launch = useLaunchExperiment();
  const stop = useStopExperiment();
  const del = useDeleteExperiment();
  const { data: checkpoints = [] } = useCheckpoints(id!);
  const { data: logData } = useLog(id!, exp?.status === 'running');

  const isRunning = exp?.status === 'running';
  const metrics = useLiveMetrics(id!, isRunning);

  if (isLoading) return <div style={{ color: '#6b7280' }}>Loading…</div>;
  if (!exp) return <div style={{ color: '#dc2626' }}>Experiment not found.</div>;

  const canLaunch = ['pending', 'completed', 'failed', 'stopped'].includes(exp.status);

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <button
          onClick={() => navigate('/')}
          style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 18, color: '#6b7280' }}
        >
          ←
        </button>
        <h1 style={{ margin: 0 }}>{exp.name}</h1>
        <StatusBadge status={exp.status} />
      </div>
      <div style={{ color: '#6b7280', fontSize: 13, marginBottom: 16 }}>
        id: {exp.id} · {exp.model.num_layers}L · d_model={exp.model.d_model} · {exp.device}
        {exp.pid && ` · pid=${exp.pid}`}
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {canLaunch && (
          <button
            onClick={() => launch.mutate(exp.id)}
            disabled={launch.isPending}
            style={{ background: '#2563eb', color: '#fff', border: 'none', borderRadius: 4, padding: '8px 16px', cursor: 'pointer' }}
          >
            {launch.isPending ? 'Launching…' : 'Launch'}
          </button>
        )}
        {isRunning && (
          <button
            onClick={() => stop.mutate(exp.id)}
            style={{ background: '#dc2626', color: '#fff', border: 'none', borderRadius: 4, padding: '8px 16px', cursor: 'pointer' }}
          >
            Stop
          </button>
        )}
        {!isRunning && (
          <button
            onClick={() => {
              if (confirm('Delete this experiment?')) {
                del.mutate(exp.id, { onSuccess: () => navigate('/') });
              }
            }}
            style={{ background: '#f3f4f6', border: '1px solid #d1d5db', borderRadius: 4, padding: '8px 16px', cursor: 'pointer' }}
          >
            Delete
          </button>
        )}
      </div>

      <h2>Loss Curves</h2>
      <LossCurveChart points={metrics} />

      <h2 style={{ marginTop: 24 }}>Log</h2>
      <LogViewer log={logData?.log ?? ''} />

      {checkpoints.length > 0 && (
        <GeneratePanel experimentId={exp.id} checkpoints={checkpoints} />
      )}
    </div>
  );
}
