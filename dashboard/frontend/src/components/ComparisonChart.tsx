import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { MetricPoint } from '../types';

const PALETTE = ['#2563eb', '#16a34a', '#dc2626', '#d97706', '#7c3aed', '#0891b2'];

interface ExperimentMetrics {
  id: string;
  name: string;
  points: MetricPoint[];
}

interface Props {
  experiments: ExperimentMetrics[];
  metric: 'train_loss' | 'val_loss';
}

export function ComparisonChart({ experiments, metric }: Props) {
  // Build a unified iteration axis
  const iterSet = new Set<number>();
  const expMaps: Map<number, number | undefined>[] = [];

  for (const exp of experiments) {
    const m = new Map<number, number | undefined>();
    for (const p of exp.points) {
      const v = p[metric];
      if (v !== undefined) {
        iterSet.add(p.iteration);
        m.set(p.iteration, v);
      }
    }
    expMaps.push(m);
  }

  const iters = Array.from(iterSet).sort((a, b) => a - b);
  const data = iters.map((iter) => {
    const row: Record<string, number | undefined> = { iteration: iter };
    experiments.forEach((exp, i) => {
      row[exp.id] = expMaps[i].get(iter);
    });
    return row;
  });

  if (data.length === 0) {
    return <div style={{ color: '#6b7280', padding: 16 }}>No data for selected experiments.</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'insideBottom', offset: -4 }} />
        <YAxis />
        <Tooltip />
        <Legend verticalAlign="top" />
        {experiments.map((exp, i) => (
          <Line
            key={exp.id}
            type="monotone"
            dataKey={exp.id}
            stroke={PALETTE[i % PALETTE.length]}
            dot={false}
            connectNulls={false}
            name={exp.name}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
