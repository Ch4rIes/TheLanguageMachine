import type { ExperimentRecord } from '../types';

const colors: Record<ExperimentRecord['status'], string> = {
  pending: '#6b7280',
  running: '#2563eb',
  completed: '#16a34a',
  failed: '#dc2626',
  stopped: '#d97706',
};

export function StatusBadge({ status }: { status: ExperimentRecord['status'] }) {
  return (
    <span
      style={{
        backgroundColor: colors[status] ?? '#6b7280',
        color: '#fff',
        borderRadius: 4,
        padding: '2px 8px',
        fontSize: 12,
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: 1,
      }}
    >
      {status}
    </span>
  );
}
