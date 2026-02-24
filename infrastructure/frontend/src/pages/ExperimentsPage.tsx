import { Link } from 'react-router-dom';
import { ExperimentCard } from '../components/ExperimentCard';
import { useExperiments } from '../hooks/useExperiments';

export function ExperimentsPage() {
  const { data: experiments, isLoading, error } = useExperiments();

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <h1 style={{ margin: 0 }}>Experiments</h1>
        <div style={{ display: 'flex', gap: 12 }}>
          <Link to="/compare" style={{ textDecoration: 'none' }}>
            <button style={{ background: '#f3f4f6', border: '1px solid #d1d5db', borderRadius: 4, padding: '8px 16px', cursor: 'pointer' }}>
              Compare
            </button>
          </Link>
          <Link to="/new" style={{ textDecoration: 'none' }}>
            <button style={{ background: '#2563eb', color: '#fff', border: 'none', borderRadius: 4, padding: '8px 16px', cursor: 'pointer' }}>
              + New Experiment
            </button>
          </Link>
        </div>
      </div>
      {isLoading && <div style={{ color: '#6b7280' }}>Loading…</div>}
      {error && <div style={{ color: '#dc2626' }}>Error loading experiments.</div>}
      {experiments && experiments.length === 0 && (
        <div style={{ color: '#6b7280', textAlign: 'center', padding: 48 }}>
          No experiments yet. <Link to="/new">Create one</Link>.
        </div>
      )}
      {experiments?.map((exp) => <ExperimentCard key={exp.id} exp={exp} />)}
    </div>
  );
}
