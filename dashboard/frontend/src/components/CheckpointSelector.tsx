import type { Checkpoint } from '../types';

interface Props {
  checkpoints: Checkpoint[];
  selected: string;
  onChange: (path: string) => void;
}

export function CheckpointSelector({ checkpoints, selected, onChange }: Props) {
  return (
    <select
      value={selected}
      onChange={(e) => onChange(e.target.value)}
      style={{ padding: '6px 10px', borderRadius: 4, border: '1px solid #d1d5db', fontSize: 14 }}
    >
      <option value="">Select checkpoint…</option>
      {checkpoints.map((c) => (
        <option key={c.path} value={c.path}>
          {c.name}
        </option>
      ))}
    </select>
  );
}
