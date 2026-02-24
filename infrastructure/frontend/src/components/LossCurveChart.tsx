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

interface Props {
  points: MetricPoint[];
}

// Merge train/val points by iteration for recharts
function mergePoints(points: MetricPoint[]) {
  const map = new Map<number, { iteration: number; train_loss?: number; val_loss?: number }>();
  for (const p of points) {
    const existing = map.get(p.iteration) ?? { iteration: p.iteration };
    if (p.train_loss !== undefined) existing.train_loss = p.train_loss;
    if (p.val_loss !== undefined) existing.val_loss = p.val_loss;
    map.set(p.iteration, existing);
  }
  return Array.from(map.values()).sort((a, b) => a.iteration - b.iteration);
}

export function LossCurveChart({ points }: Props) {
  const data = mergePoints(points);

  if (data.length === 0) {
    return <div style={{ color: '#6b7280', padding: 16 }}>No metrics yet.</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'insideBottom', offset: -4 }} />
        <YAxis />
        <Tooltip />
        <Legend verticalAlign="top" />
        <Line
          type="monotone"
          dataKey="train_loss"
          stroke="#2563eb"
          dot={false}
          connectNulls={false}
          name="Train Loss"
        />
        <Line
          type="monotone"
          dataKey="val_loss"
          stroke="#dc2626"
          strokeDasharray="5 5"
          dot={false}
          connectNulls={false}
          name="Val Loss"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
