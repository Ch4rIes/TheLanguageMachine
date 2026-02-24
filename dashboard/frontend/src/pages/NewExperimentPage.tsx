import { useNavigate } from 'react-router-dom';
import { ExperimentForm } from '../components/ExperimentForm';
import { useCreateExperiment } from '../hooks/useExperiments';
import type { ExperimentCreate } from '../types';

export function NewExperimentPage() {
  const navigate = useNavigate();
  const create = useCreateExperiment();

  const handleSubmit = (data: ExperimentCreate) => {
    create.mutate(data, {
      onSuccess: (exp) => navigate(`/experiments/${exp.id}`),
    });
  };

  return (
    <div>
      <h1 style={{ marginBottom: 24 }}>New Experiment</h1>
      {create.isError && (
        <div style={{ color: '#dc2626', marginBottom: 16 }}>
          Error: {(create.error as Error).message}
        </div>
      )}
      <ExperimentForm onSubmit={handleSubmit} loading={create.isPending} />
    </div>
  );
}
