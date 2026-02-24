import { useState } from 'react';
import type { ExperimentCreate } from '../types';

interface Props {
  onSubmit: (data: ExperimentCreate) => void;
  loading?: boolean;
}

const defaultValues: ExperimentCreate = {
  name: '',
  train_data_path: '',
  val_data_path: '',
  batch_size: 32,
  max_iters: 10000,
  grad_clip_norm: 1.0,
  log_interval: 10,
  val_interval: 100,
  checkpoint_interval: 1000,
  device: 'cpu',
  tokenizer_path: '',
  model: {
    vocab_size: 32000,
    context_length: 256,
    num_layers: 6,
    d_model: 512,
    num_heads: 8,
    d_ff: 1024,
    theta: 10000.0,
  },
  optimizer: {
    lr: 1e-3,
    betas: [0.9, 0.999],
    eps: 1e-8,
    weight_decay: 0.01,
  },
  scheduler: {
    warmup_iters: 100,
    cosine_cycle_iters: 10000,
    min_lr_ratio: 0.1,
  },
};

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 14 }}>
      <span style={{ fontWeight: 600, color: '#374151' }}>{label}</span>
      {children}
    </label>
  );
}

const inputStyle = {
  padding: '6px 10px',
  borderRadius: 4,
  border: '1px solid #d1d5db',
  fontSize: 14,
};

export function ExperimentForm({ onSubmit, loading }: Props) {
  const [form, setForm] = useState<ExperimentCreate>(defaultValues);

  const set = (path: string[], value: unknown) => {
    setForm((prev) => {
      const next = structuredClone(prev) as Record<string, unknown>;
      let cur = next;
      for (let i = 0; i < path.length - 1; i++) {
        cur = cur[path[i]] as Record<string, unknown>;
      }
      cur[path[path.length - 1]] = value;
      return next as ExperimentCreate;
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 640 }}>
      <h3 style={{ margin: 0 }}>Basic</h3>
      <Field label="Experiment name *">
        <input required style={inputStyle} value={form.name} onChange={(e) => set(['name'], e.target.value)} />
      </Field>
      <Field label="Train data path *">
        <input required style={inputStyle} value={form.train_data_path} onChange={(e) => set(['train_data_path'], e.target.value)} placeholder="data/train.bin" />
      </Field>
      <Field label="Val data path">
        <input style={inputStyle} value={form.val_data_path ?? ''} onChange={(e) => set(['val_data_path'], e.target.value || undefined)} placeholder="data/val.bin" />
      </Field>
      <Field label="Tokenizer path (JSON)">
        <input style={inputStyle} value={form.tokenizer_path ?? ''} onChange={(e) => set(['tokenizer_path'], e.target.value || undefined)} placeholder="/path/to/tokenizer.json" />
      </Field>
      <Field label="Device">
        <select style={inputStyle} value={form.device} onChange={(e) => set(['device'], e.target.value)}>
          <option value="cpu">cpu</option>
          <option value="cuda:0">cuda:0</option>
          <option value="mps">mps</option>
        </select>
      </Field>

      <h3 style={{ margin: 0 }}>Training</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
        <Field label="Batch size"><input type="number" style={inputStyle} value={form.batch_size} onChange={(e) => set(['batch_size'], Number(e.target.value))} /></Field>
        <Field label="Max iters"><input type="number" style={inputStyle} value={form.max_iters} onChange={(e) => set(['max_iters'], Number(e.target.value))} /></Field>
        <Field label="Grad clip norm"><input type="number" step="0.1" style={inputStyle} value={form.grad_clip_norm} onChange={(e) => set(['grad_clip_norm'], Number(e.target.value))} /></Field>
        <Field label="Log interval"><input type="number" style={inputStyle} value={form.log_interval} onChange={(e) => set(['log_interval'], Number(e.target.value))} /></Field>
        <Field label="Val interval"><input type="number" style={inputStyle} value={form.val_interval} onChange={(e) => set(['val_interval'], Number(e.target.value))} /></Field>
        <Field label="Checkpoint interval"><input type="number" style={inputStyle} value={form.checkpoint_interval} onChange={(e) => set(['checkpoint_interval'], Number(e.target.value))} /></Field>
      </div>

      <h3 style={{ margin: 0 }}>Model</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 12 }}>
        <Field label="Vocab size"><input type="number" style={inputStyle} value={form.model.vocab_size} onChange={(e) => set(['model', 'vocab_size'], Number(e.target.value))} /></Field>
        <Field label="Context length"><input type="number" style={inputStyle} value={form.model.context_length} onChange={(e) => set(['model', 'context_length'], Number(e.target.value))} /></Field>
        <Field label="Num layers"><input type="number" style={inputStyle} value={form.model.num_layers} onChange={(e) => set(['model', 'num_layers'], Number(e.target.value))} /></Field>
        <Field label="d_model"><input type="number" style={inputStyle} value={form.model.d_model} onChange={(e) => set(['model', 'd_model'], Number(e.target.value))} /></Field>
        <Field label="Num heads"><input type="number" style={inputStyle} value={form.model.num_heads} onChange={(e) => set(['model', 'num_heads'], Number(e.target.value))} /></Field>
        <Field label="d_ff"><input type="number" style={inputStyle} value={form.model.d_ff} onChange={(e) => set(['model', 'd_ff'], Number(e.target.value))} /></Field>
        <Field label="Theta"><input type="number" style={inputStyle} value={form.model.theta} onChange={(e) => set(['model', 'theta'], Number(e.target.value))} /></Field>
      </div>

      <h3 style={{ margin: 0 }}>Optimizer</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 12 }}>
        <Field label="LR"><input type="number" step="0.0001" style={inputStyle} value={form.optimizer.lr} onChange={(e) => set(['optimizer', 'lr'], Number(e.target.value))} /></Field>
        <Field label="Beta1"><input type="number" step="0.01" style={inputStyle} value={form.optimizer.betas[0]} onChange={(e) => set(['optimizer', 'betas'], [Number(e.target.value), form.optimizer.betas[1]])} /></Field>
        <Field label="Beta2"><input type="number" step="0.001" style={inputStyle} value={form.optimizer.betas[1]} onChange={(e) => set(['optimizer', 'betas'], [form.optimizer.betas[0], Number(e.target.value)])} /></Field>
        <Field label="Weight decay"><input type="number" step="0.001" style={inputStyle} value={form.optimizer.weight_decay} onChange={(e) => set(['optimizer', 'weight_decay'], Number(e.target.value))} /></Field>
      </div>

      <h3 style={{ margin: 0 }}>Scheduler</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
        <Field label="Warmup iters"><input type="number" style={inputStyle} value={form.scheduler.warmup_iters} onChange={(e) => set(['scheduler', 'warmup_iters'], Number(e.target.value))} /></Field>
        <Field label="Cosine cycle iters"><input type="number" style={inputStyle} value={form.scheduler.cosine_cycle_iters} onChange={(e) => set(['scheduler', 'cosine_cycle_iters'], Number(e.target.value))} /></Field>
        <Field label="Min LR ratio"><input type="number" step="0.01" style={inputStyle} value={form.scheduler.min_lr_ratio} onChange={(e) => set(['scheduler', 'min_lr_ratio'], Number(e.target.value))} /></Field>
      </div>

      <button
        type="submit"
        disabled={loading}
        style={{
          alignSelf: 'flex-start',
          background: '#2563eb',
          color: '#fff',
          border: 'none',
          borderRadius: 4,
          padding: '10px 24px',
          fontSize: 15,
          cursor: loading ? 'not-allowed' : 'pointer',
          opacity: loading ? 0.6 : 1,
        }}
      >
        {loading ? 'Creating…' : 'Create Experiment'}
      </button>
    </form>
  );
}
