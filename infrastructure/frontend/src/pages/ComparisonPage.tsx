import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ComparisonChart } from '../components/ComparisonChart';
import { StatusBadge } from '../components/StatusBadge';
import { metricsApi } from '../api';
import { useExperiments } from '../hooks/useExperiments';
import type { MetricPoint } from '../types';

export function ComparisonPage() {
  const { data: experiments = [] } = useExperiments();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [metric, setMetric] = useState<'train_loss' | 'val_loss'>('train_loss');

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // Fetch metrics for selected experiments
  const metricsQueries = useQuery({
    queryKey: ['comparison-metrics', Array.from(selected).sort()],
    queryFn: async () => {
      const results: { id: string; name: string; points: MetricPoint[] }[] = [];
      for (const id of selected) {
        const exp = experiments.find((e) => e.id === id);
        if (!exp) continue;
        const points = await metricsApi.snapshot(id);
        results.push({ id, name: exp.name, points });
      }
      return results;
    },
    enabled: selected.size > 0 && experiments.length > 0,
  });

  return (
    <div>
      <h1 style={{ marginBottom: 24 }}>Compare Experiments</h1>

      <div style={{ display: 'flex', gap: 16, marginBottom: 24, flexWrap: 'wrap' }}>
        {experiments.map((exp) => (
          <label
            key={exp.id}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              cursor: 'pointer',
              padding: '8px 12px',
              border: selected.has(exp.id) ? '2px solid #2563eb' : '1px solid #d1d5db',
              borderRadius: 6,
              background: selected.has(exp.id) ? '#eff6ff' : '#fff',
            }}
          >
            <input
              type="checkbox"
              checked={selected.has(exp.id)}
              onChange={() => toggle(exp.id)}
            />
            <span style={{ fontWeight: 600 }}>{exp.name}</span>
            <StatusBadge status={exp.status} />
          </label>
        ))}
      </div>

      <div style={{ marginBottom: 16 }}>
        <label style={{ fontWeight: 600, marginRight: 12 }}>Metric:</label>
        <select
          value={metric}
          onChange={(e) => setMetric(e.target.value as 'train_loss' | 'val_loss')}
          style={{ padding: '4px 8px', borderRadius: 4, border: '1px solid #d1d5db' }}
        >
          <option value="train_loss">Train Loss</option>
          <option value="val_loss">Val Loss</option>
        </select>
      </div>

      {selected.size === 0 && (
        <div style={{ color: '#6b7280', textAlign: 'center', padding: 48 }}>
          Select experiments above to compare.
        </div>
      )}

      {metricsQueries.isLoading && <div style={{ color: '#6b7280' }}>Loading metrics…</div>}

      {metricsQueries.data && (
        <ComparisonChart experiments={metricsQueries.data} metric={metric} />
      )}
    </div>
  );
}
