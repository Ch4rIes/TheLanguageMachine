import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { profilerApi } from '../api';
import type { StepProfilerRequest, StepProfilerResponse, StepProfilerResult, StepProfilerRun } from '../types';

const inputStyle: React.CSSProperties = {
  padding: '6px 10px',
  borderRadius: 4,
  border: '1px solid #d1d5db',
  fontSize: 13,
  width: '100%',
};

const labelStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 4,
  fontSize: 13,
};

const panelStyle: React.CSSProperties = {
  border: '1px solid #e5e7eb',
  borderRadius: 6,
  padding: 16,
  background: '#fff',
};

const defaultProfile: StepProfilerRequest = {
  device: 'cpu',
  batch_size: 8,
  warmup_steps: 5,
  profile_steps: 20,
  vocab_size: 10000,
  context_length: 128,
  num_layers: 4,
  d_model: 256,
  num_heads: 4,
  d_ff: 1024,
  theta: 10000.0,
  lr: 1e-3,
  weight_decay: 0.01,
};

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={labelStyle}>
      <span style={{ fontWeight: 600, color: '#374151' }}>{label}</span>
      {children}
    </label>
  );
}

function formatNumber(value?: number | null, digits = 1) {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatMode(mode: StepProfilerResult['mode']) {
  return mode
    .replace('forward_backward_optimizer', 'forward + backward + optimizer')
    .replace('forward_backward', 'forward + backward')
    .replace('forward', 'forward');
}

function HardwareBlock({ data }: { data: StepProfilerResponse }) {
  const { hardware } = data;
  const rows = [
    ['Device', hardware.device],
    ['CPU', hardware.cpu.model],
    ['CPU cores', `${hardware.cpu.physical_cores ?? '—'} physical / ${hardware.cpu.logical_cores ?? '—'} logical`],
    ['RAM', `${formatNumber(hardware.cpu.memory_gb)} GB`],
    ['GPU', hardware.gpu ? hardware.gpu.name : '—'],
    ['GPU memory', hardware.gpu?.total_memory_mb ? `${formatNumber(hardware.gpu.total_memory_mb / 1024)} GB` : '—'],
    ['Torch', hardware.torch],
    ['Platform', hardware.platform],
  ];

  return (
    <section style={panelStyle}>
      <h2 style={{ margin: '0 0 12px' }}>Hardware</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 10 }}>
        {rows.map(([label, value]) => (
          <div key={label}>
            <div style={{ fontSize: 12, color: '#6b7280', fontWeight: 600 }}>{label}</div>
            <div style={{ fontSize: 14, color: '#111827', overflowWrap: 'anywhere' }}>{value}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function ResultsTable({ results }: { results: StepProfilerResult[] }) {
  return (
    <section style={panelStyle}>
      <h2 style={{ margin: '0 0 12px' }}>Step Profile</h2>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, minWidth: 760 }}>
          <thead>
            <tr style={{ textAlign: 'left', color: '#4b5563', borderBottom: '1px solid #e5e7eb' }}>
              <th style={{ padding: '8px 6px' }}>Mode</th>
              <th style={{ padding: '8px 6px' }}>Mean</th>
              <th style={{ padding: '8px 6px' }}>p50</th>
              <th style={{ padding: '8px 6px' }}>p95</th>
              <th style={{ padding: '8px 6px' }}>Tokens/s</th>
              <th style={{ padding: '8px 6px' }}>Params</th>
              <th style={{ padding: '8px 6px' }}>Peak alloc.</th>
              <th style={{ padding: '8px 6px' }}>Peak reserved</th>
              <th style={{ padding: '8px 6px' }}>Loss</th>
            </tr>
          </thead>
          <tbody>
            {results.map((row) => (
              <tr key={row.mode} style={{ borderBottom: '1px solid #f3f4f6' }}>
                <td style={{ padding: '8px 6px', fontWeight: 600 }}>{formatMode(row.mode)}</td>
                <td style={{ padding: '8px 6px' }}>{formatNumber(row.mean_ms)} ms</td>
                <td style={{ padding: '8px 6px' }}>{formatNumber(row.p50_ms)} ms</td>
                <td style={{ padding: '8px 6px' }}>{formatNumber(row.p95_ms)} ms</td>
                <td style={{ padding: '8px 6px' }}>{formatNumber(row.tokens_per_sec, 0)}</td>
                <td style={{ padding: '8px 6px' }}>{formatNumber(row.parameter_count, 0)}</td>
                <td style={{ padding: '8px 6px' }}>{formatNumber(row.peak_allocated_mb)} MB</td>
                <td style={{ padding: '8px 6px' }}>{formatNumber(row.peak_reserved_mb)} MB</td>
                <td style={{ padding: '8px 6px' }}>{formatNumber(row.last_loss, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function runLabel(run: StepProfilerRun) {
  const date = new Date(run.created_at * 1000);
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function HistoryPanel({
  runs,
  activeId,
  onView,
  onLoadConfig,
}: {
  runs: StepProfilerRun[];
  activeId?: string;
  onView: (run: StepProfilerRun) => void;
  onLoadConfig: (run: StepProfilerRun) => void;
}) {
  return (
    <section style={panelStyle}>
      <h2 style={{ margin: '0 0 12px' }}>History</h2>
      {runs.length === 0 ? (
        <div style={{ color: '#6b7280', fontSize: 13 }}>No profiler runs yet.</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, minWidth: 760 }}>
            <thead>
              <tr style={{ textAlign: 'left', color: '#4b5563', borderBottom: '1px solid #e5e7eb' }}>
                <th style={{ padding: '8px 6px' }}>Run</th>
                <th style={{ padding: '8px 6px' }}>Device</th>
                <th style={{ padding: '8px 6px' }}>Model</th>
                <th style={{ padding: '8px 6px' }}>Batch</th>
                <th style={{ padding: '8px 6px' }}>Best tokens/s</th>
                <th style={{ padding: '8px 6px' }}>Params</th>
                <th style={{ padding: '8px 6px' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => {
                const best = Math.max(...run.results.map((result) => result.tokens_per_sec));
                const params = run.results[0]?.parameter_count;
                const selected = run.id === activeId;
                return (
                  <tr key={run.id} style={{ borderBottom: '1px solid #f3f4f6', background: selected ? '#eff6ff' : '#fff' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{runLabel(run)}</td>
                    <td style={{ padding: '8px 6px' }}>{run.hardware.device}</td>
                    <td style={{ padding: '8px 6px' }}>
                      {run.config.num_layers}L · d={run.config.d_model} · ctx={run.config.context_length}
                    </td>
                    <td style={{ padding: '8px 6px' }}>{run.config.batch_size}</td>
                    <td style={{ padding: '8px 6px' }}>{formatNumber(best, 0)}</td>
                    <td style={{ padding: '8px 6px' }}>{formatNumber(params, 0)}</td>
                    <td style={{ padding: '8px 6px' }}>
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        <button
                          type="button"
                          onClick={() => onView(run)}
                          style={{ border: '1px solid #d1d5db', background: '#fff', borderRadius: 4, padding: '4px 8px', cursor: 'pointer' }}
                        >
                          View
                        </button>
                        <button
                          type="button"
                          onClick={() => onLoadConfig(run)}
                          style={{ border: '1px solid #d1d5db', background: '#f9fafb', borderRadius: 4, padding: '4px 8px', cursor: 'pointer' }}
                        >
                          Load config
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function extractError(error: unknown) {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { data?: { detail?: string } } }).response;
    if (response?.data?.detail) return response.data.detail;
  }
  return error instanceof Error ? error.message : 'Profile failed';
}

export function ProfilerPage() {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<StepProfilerRequest>(defaultProfile);
  const [activeRun, setActiveRun] = useState<StepProfilerRun | null>(null);
  const { data: runs = [] } = useQuery({
    queryKey: ['profile-runs'],
    queryFn: () => profilerApi.listRuns(50),
  });
  const mutation = useMutation({
    mutationFn: profilerApi.runStep,
    onSuccess: (run) => {
      setActiveRun(run);
      queryClient.invalidateQueries({ queryKey: ['profile-runs'] });
    },
  });

  useEffect(() => {
    if (!activeRun && runs.length > 0) {
      setActiveRun(runs[0]);
    }
  }, [activeRun, runs]);

  const setField = (key: keyof StepProfilerRequest, value: string | number) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const numberField = (key: keyof StepProfilerRequest) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setField(key, Number(event.target.value));
  };

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    mutation.mutate(form);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <h1 style={{ margin: 0 }}>Profiler</h1>

      <form onSubmit={submit} style={{ ...panelStyle, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
          <Field label="Device">
            <select style={inputStyle} value={form.device} onChange={(e) => setField('device', e.target.value)}>
              <option value="cpu">cpu</option>
              <option value="cuda">cuda</option>
              <option value="cuda:0">cuda:0</option>
              <option value="mps">mps</option>
            </select>
          </Field>
          <Field label="Batch size">
            <input type="number" min={1} style={inputStyle} value={form.batch_size} onChange={numberField('batch_size')} />
          </Field>
          <Field label="Warmup steps">
            <input type="number" min={0} style={inputStyle} value={form.warmup_steps} onChange={numberField('warmup_steps')} />
          </Field>
          <Field label="Profile steps">
            <input type="number" min={1} style={inputStyle} value={form.profile_steps} onChange={numberField('profile_steps')} />
          </Field>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
          <Field label="Vocab size">
            <input type="number" min={1} style={inputStyle} value={form.vocab_size} onChange={numberField('vocab_size')} />
          </Field>
          <Field label="Context length">
            <input type="number" min={1} style={inputStyle} value={form.context_length} onChange={numberField('context_length')} />
          </Field>
          <Field label="Layers">
            <input type="number" min={1} style={inputStyle} value={form.num_layers} onChange={numberField('num_layers')} />
          </Field>
          <Field label="d_model">
            <input type="number" min={1} style={inputStyle} value={form.d_model} onChange={numberField('d_model')} />
          </Field>
          <Field label="Heads">
            <input type="number" min={1} style={inputStyle} value={form.num_heads} onChange={numberField('num_heads')} />
          </Field>
          <Field label="d_ff">
            <input type="number" min={1} style={inputStyle} value={form.d_ff} onChange={numberField('d_ff')} />
          </Field>
          <Field label="Theta">
            <input type="number" min={1} style={inputStyle} value={form.theta} onChange={numberField('theta')} />
          </Field>
          <Field label="LR">
            <input type="number" min={0} step="0.0001" style={inputStyle} value={form.lr} onChange={numberField('lr')} />
          </Field>
        </div>

        <button
          type="submit"
          disabled={mutation.isPending}
          style={{
            alignSelf: 'flex-start',
            background: '#2563eb',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            padding: '9px 18px',
            fontSize: 14,
            cursor: mutation.isPending ? 'not-allowed' : 'pointer',
            opacity: mutation.isPending ? 0.65 : 1,
          }}
        >
          {mutation.isPending ? 'Running…' : 'Run Profile'}
        </button>
      </form>

      {mutation.isError && (
        <div style={{ color: '#b91c1c', background: '#fee2e2', borderRadius: 6, padding: 12, whiteSpace: 'pre-wrap' }}>
          {extractError(mutation.error)}
        </div>
      )}

      {mutation.data && (
        <div style={{ color: '#166534', background: '#dcfce7', borderRadius: 6, padding: 12, fontSize: 13 }}>
          Saved profile run {mutation.data.id}.
        </div>
      )}

      <HistoryPanel
        runs={runs}
        activeId={activeRun?.id}
        onView={setActiveRun}
        onLoadConfig={(run) => {
          setForm(run.config);
          setActiveRun(run);
        }}
      />

      {activeRun && (
        <>
          <HardwareBlock data={activeRun} />
          <ResultsTable results={activeRun.results} />
        </>
      )}
    </div>
  );
}
