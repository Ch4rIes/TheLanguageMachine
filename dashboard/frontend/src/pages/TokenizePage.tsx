import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { tokenizeApi } from '../api/tokenize';
import type { TokenizeTask } from '../api/tokenize';

// ── shared styles ────────────────────────────────────────────────────────────
const input: React.CSSProperties = {
  padding: '6px 10px',
  borderRadius: 4,
  border: '1px solid #d1d5db',
  fontSize: 13,
  width: '100%',
};
const btn = (variant: 'primary' | 'ghost' = 'primary'): React.CSSProperties => ({
  background: variant === 'primary' ? '#2563eb' : '#f3f4f6',
  color: variant === 'primary' ? '#fff' : '#374151',
  border: variant === 'primary' ? 'none' : '1px solid #d1d5db',
  borderRadius: 4,
  padding: '7px 16px',
  fontSize: 13,
  cursor: 'pointer',
});

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 13 }}>
      <span style={{ fontWeight: 600, color: '#374151' }}>{label}</span>
      {children}
    </label>
  );
}

// ── task log box ─────────────────────────────────────────────────────────────
function TaskLog({ task }: { task: TokenizeTask }) {
  const qc = useQueryClient();
  const { data: logData } = useQuery({
    queryKey: ['tok-task-log', task.id],
    queryFn: () => tokenizeApi.taskLog(task.id),
    refetchInterval: task.status === 'running' ? 2000 : false,
  });

  // Poll task status while running
  useQuery({
    queryKey: ['tok-task', task.id],
    queryFn: () => tokenizeApi.getTask(task.id),
    refetchInterval: task.status === 'running' ? 2000 : false,
    onSuccess: (t) => {
      if (t.status !== task.status) {
        qc.invalidateQueries({ queryKey: ['tok-tasks'] });
      }
    },
  } as Parameters<typeof useQuery>[0]);

  const statusColor = task.status === 'running' ? '#2563eb' : task.status === 'completed' ? '#16a34a' : '#dc2626';

  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>{task.label}</span>
        <span style={{ color: statusColor, fontWeight: 600, fontSize: 12 }}>{task.status.toUpperCase()}</span>
        {task.pid && <span style={{ color: '#6b7280', fontSize: 12 }}>pid={task.pid}</span>}
      </div>
      <pre style={{
        background: '#1e1e1e', color: '#d4d4d4', borderRadius: 6,
        padding: 12, fontSize: 12, overflowY: 'auto', maxHeight: 240,
        whiteSpace: 'pre-wrap', wordBreak: 'break-all', margin: 0,
      }}>
        {logData?.log || '(waiting for output…)'}
      </pre>
    </div>
  );
}

// ── Tab: Prepare Data ─────────────────────────────────────────────────────────
function PrepareDataTab() {
  const qc = useQueryClient();

  // Train tokenizer form
  const [trainForm, setTrainForm] = useState({
    input_path: 'data/tinystories_train.txt',
    vocab_size: 10000,
    output_path: 'tokenizers/my_tokenizer.json',
    special_tokens: '<|endoftext|>',
  });

  // Encode dataset form
  const [encTrainForm, setEncTrainForm] = useState({
    tokenizer_path: 'tokenizers/tinystories_10k.json',
    input_path: 'data/tinystories_train.txt',
    output_path: 'data/tinystories_train.bin',
  });
  const [encValForm, setEncValForm] = useState({
    tokenizer_path: 'tokenizers/tinystories_10k.json',
    input_path: 'data/tinystories_val.txt',
    output_path: 'data/tinystories_val.bin',
  });

  const { data: tasks = [] } = useQuery({
    queryKey: ['tok-tasks'],
    queryFn: tokenizeApi.listTasks,
    refetchInterval: 3000,
  });

  const trainMut = useMutation({
    mutationFn: () => tokenizeApi.trainTokenizer(trainForm),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tok-tasks'] }),
  });

  const encTrainMut = useMutation({
    mutationFn: () => tokenizeApi.encodeDataset(encTrainForm),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tok-tasks'] }),
  });

  const encValMut = useMutation({
    mutationFn: () => tokenizeApi.encodeDataset(encValForm),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tok-tasks'] }),
  });

  const setTf = (k: string, v: string | number) => setTrainForm((p) => ({ ...p, [k]: v }));
  const setEt = (k: string, v: string) => setEncTrainForm((p) => ({ ...p, [k]: v }));
  const setEv = (k: string, v: string) => setEncValForm((p) => ({ ...p, [k]: v }));

  // All tasks from running mutations + task list
  const activeTasks = tasks.slice().sort((a, b) => b.created_at - a.created_at);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
      {/* Download instructions */}
      <section>
        <h3 style={{ margin: '0 0 8px' }}>Step 1 — Download TinyStories</h3>
        <p style={{ margin: '0 0 8px', color: '#6b7280', fontSize: 13 }}>
          Run these from <code>assignment1-basics/</code>:
        </p>
        <pre style={{ background: '#1e1e1e', color: '#86efac', borderRadius: 6, padding: 12, fontSize: 12, margin: 0, overflowX: 'auto' }}>{`curl -L "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" \\
     -o data/tinystories_train.txt

curl -L "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" \\
     -o data/tinystories_val.txt`}</pre>
        <p style={{ margin: '8px 0 0', color: '#6b7280', fontSize: 12 }}>
          ~2 GB total. Or use the sample already at <code>tests/fixtures/tinystories_sample_5M.txt</code> for a quick test.
        </p>
      </section>

      {/* Train tokenizer */}
      <section>
        <h3 style={{ margin: '0 0 12px' }}>Step 2 — Train BPE Tokenizer</h3>
        <p style={{ margin: '0 0 10px', color: '#6b7280', fontSize: 13 }}>
          Skip if using the pre-built <code>tokenizers/tinystories_10k.json</code>.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, maxWidth: 640 }}>
          <Field label="Input text file">
            <input style={input} value={trainForm.input_path} onChange={(e) => setTf('input_path', e.target.value)} />
          </Field>
          <Field label="Vocab size">
            <input type="number" style={input} value={trainForm.vocab_size} onChange={(e) => setTf('vocab_size', Number(e.target.value))} />
          </Field>
          <Field label="Output tokenizer JSON">
            <input style={input} value={trainForm.output_path} onChange={(e) => setTf('output_path', e.target.value)} />
          </Field>
          <Field label="Special tokens (comma-separated)">
            <input style={input} value={trainForm.special_tokens} onChange={(e) => setTf('special_tokens', e.target.value)} />
          </Field>
        </div>
        <button style={{ ...btn(), marginTop: 10 }} onClick={() => trainMut.mutate()} disabled={trainMut.isPending}>
          {trainMut.isPending ? 'Spawning…' : 'Train Tokenizer'}
        </button>
        {trainMut.data && <TaskLog task={trainMut.data} />}
      </section>

      {/* Encode train split */}
      <section>
        <h3 style={{ margin: '0 0 12px' }}>Step 3 — Encode Datasets to .bin</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, maxWidth: 720 }}>
          <Field label="Tokenizer JSON">
            <input style={input} value={encTrainForm.tokenizer_path} onChange={(e) => setEt('tokenizer_path', e.target.value)} />
          </Field>
          <Field label="Input (train .txt)">
            <input style={input} value={encTrainForm.input_path} onChange={(e) => setEt('input_path', e.target.value)} />
          </Field>
          <Field label="Output (train .bin)">
            <input style={input} value={encTrainForm.output_path} onChange={(e) => setEt('output_path', e.target.value)} />
          </Field>
        </div>
        <button style={{ ...btn(), marginTop: 10 }} onClick={() => encTrainMut.mutate()} disabled={encTrainMut.isPending}>
          {encTrainMut.isPending ? 'Spawning…' : 'Encode Train Split'}
        </button>
        {encTrainMut.data && <TaskLog task={encTrainMut.data} />}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, maxWidth: 720, marginTop: 14 }}>
          <Field label="Tokenizer JSON">
            <input style={input} value={encValForm.tokenizer_path} onChange={(e) => setEv('tokenizer_path', e.target.value)} />
          </Field>
          <Field label="Input (val .txt)">
            <input style={input} value={encValForm.input_path} onChange={(e) => setEv('input_path', e.target.value)} />
          </Field>
          <Field label="Output (val .bin)">
            <input style={input} value={encValForm.output_path} onChange={(e) => setEv('output_path', e.target.value)} />
          </Field>
        </div>
        <button style={{ ...btn(), marginTop: 10 }} onClick={() => encValMut.mutate()} disabled={encValMut.isPending}>
          {encValMut.isPending ? 'Spawning…' : 'Encode Val Split'}
        </button>
        {encValMut.data && <TaskLog task={encValMut.data} />}
      </section>

      {/* Task history */}
      {activeTasks.length > 0 && (
        <section>
          <h3 style={{ margin: '0 0 12px' }}>Task History</h3>
          {activeTasks.map((t) => (
            <TaskLog key={t.id} task={t} />
          ))}
        </section>
      )}
    </div>
  );
}

// ── Tab: Inspect ─────────────────────────────────────────────────────────────
function InspectTab() {
  const { data: tokenizers = [] } = useQuery({
    queryKey: ['tok-list'],
    queryFn: tokenizeApi.listTokenizers,
  });

  const [tokPath, setTokPath] = useState('');
  const [encText, setEncText] = useState('Once upon a time');
  const [decIds, setDecIds] = useState('');

  // Auto-select first tokenizer
  useEffect(() => {
    if (tokenizers.length > 0 && !tokPath) setTokPath(tokenizers[0].path);
  }, [tokenizers, tokPath]);

  const infoQ = useQuery({
    queryKey: ['tok-info', tokPath],
    queryFn: () => tokenizeApi.info(tokPath),
    enabled: !!tokPath,
  });

  const encodeMut = useMutation({ mutationFn: () => tokenizeApi.encodeText(tokPath, encText) });
  const decodeMut = useMutation({
    mutationFn: () => {
      const ids = decIds.trim().split(/\s+/).map(Number).filter((n) => !isNaN(n));
      return tokenizeApi.decodeText(tokPath, ids);
    },
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Tokenizer selector */}
      <section>
        <h3 style={{ margin: '0 0 10px' }}>Select Tokenizer</h3>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <select
            value={tokPath}
            onChange={(e) => setTokPath(e.target.value)}
            style={{ ...input, width: 'auto', minWidth: 280 }}
          >
            <option value="">Choose…</option>
            {tokenizers.map((t) => (
              <option key={t.path} value={t.path}>{t.name} — {t.path}</option>
            ))}
          </select>
          <span style={{ color: '#6b7280', fontSize: 12 }}>or paste a path:</span>
          <input
            style={{ ...input, width: 320 }}
            value={tokPath}
            onChange={(e) => setTokPath(e.target.value)}
            placeholder="/abs/path/to/tokenizer.json"
          />
        </div>
      </section>

      {/* Info panel */}
      {infoQ.data && (
        <section>
          <h3 style={{ margin: '0 0 10px' }}>Tokenizer Info</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, auto)', gap: '8px 24px', fontSize: 13 }}>
            <span style={{ color: '#6b7280' }}>Vocab size</span>
            <span style={{ fontWeight: 700 }}>{infoQ.data.vocab_size.toLocaleString()}</span>
            <span />
            <span style={{ color: '#6b7280' }}>BPE merges</span>
            <span style={{ fontWeight: 700 }}>{infoQ.data.num_merges.toLocaleString()}</span>
            <span />
            <span style={{ color: '#6b7280' }}>Special tokens</span>
            <span style={{ fontWeight: 700 }}>{infoQ.data.num_special_tokens}</span>
            <span style={{ color: '#6b7280', fontSize: 12 }}>
              {infoQ.data.special_tokens.map((s) => `${s.text} (${s.id})`).join(', ')}
            </span>
          </div>
          {infoQ.data.sample_tokens.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <span style={{ color: '#6b7280', fontSize: 12 }}>Sample merged tokens: </span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
                {infoQ.data.sample_tokens.map((t) => (
                  <span key={t.id} style={{
                    background: '#f3f4f6', borderRadius: 4, padding: '2px 6px',
                    fontSize: 12, fontFamily: 'monospace', border: '1px solid #e5e7eb',
                  }}>
                    <span style={{ color: '#6b7280' }}>{t.id}: </span>
                    {JSON.stringify(t.text)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {/* Encode */}
      <section>
        <h3 style={{ margin: '0 0 10px' }}>Encode Text → Token IDs</h3>
        <textarea
          value={encText}
          onChange={(e) => setEncText(e.target.value)}
          rows={3}
          style={{ ...input, fontFamily: 'monospace', resize: 'vertical' }}
        />
        <button
          style={{ ...btn(), marginTop: 8 }}
          onClick={() => encodeMut.mutate()}
          disabled={!tokPath || encodeMut.isPending}
        >
          Encode
        </button>
        {encodeMut.data && (
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6 }}>
              {encodeMut.data.count} tokens
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {encodeMut.data.tokens.map((t, i) => (
                <span key={i} title={`id=${t.id}`} style={{
                  background: '#eff6ff', borderRadius: 4, padding: '2px 6px',
                  fontSize: 12, fontFamily: 'monospace', border: '1px solid #bfdbfe',
                  cursor: 'default',
                }}>
                  {JSON.stringify(t.text)}
                  <sup style={{ color: '#93c5fd', fontSize: 9, marginLeft: 2 }}>{t.id}</sup>
                </span>
              ))}
            </div>
            <div style={{ marginTop: 8 }}>
              <span style={{ color: '#6b7280', fontSize: 12 }}>IDs: </span>
              <code style={{ fontSize: 12 }}>{encodeMut.data.token_ids.join(' ')}</code>
            </div>
          </div>
        )}
        {encodeMut.isError && <div style={{ color: '#dc2626', fontSize: 12, marginTop: 6 }}>{String(encodeMut.error)}</div>}
      </section>

      {/* Decode */}
      <section>
        <h3 style={{ margin: '0 0 10px' }}>Decode Token IDs → Text</h3>
        <input
          style={input}
          value={decIds}
          onChange={(e) => setDecIds(e.target.value)}
          placeholder="Space-separated token IDs e.g.  79 110 99 101 ..."
        />
        <button
          style={{ ...btn(), marginTop: 8 }}
          onClick={() => decodeMut.mutate()}
          disabled={!tokPath || !decIds.trim() || decodeMut.isPending}
        >
          Decode
        </button>
        {decodeMut.data && (
          <pre style={{
            background: '#f3f4f6', borderRadius: 4, padding: 10, marginTop: 8,
            fontSize: 13, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          }}>
            {decodeMut.data.text}
          </pre>
        )}
        {decodeMut.isError && <div style={{ color: '#dc2626', fontSize: 12, marginTop: 6 }}>{String(decodeMut.error)}</div>}
      </section>
    </div>
  );
}

// ── Tab: Default Config ───────────────────────────────────────────────────────
function DefaultConfigTab() {
  const { data: configs = [], isLoading } = useQuery({
    queryKey: ['tok-configs'],
    queryFn: tokenizeApi.listConfigs,
  });

  const [selected, setSelected] = useState(0);
  const cfg = configs[selected];

  if (isLoading) return <div style={{ color: '#6b7280' }}>Loading…</div>;
  if (configs.length === 0) return <div style={{ color: '#6b7280' }}>No config files found in dashboard/configs/</div>;

  return (
    <div style={{ display: 'flex', gap: 20, flexDirection: 'column' }}>
      <div style={{ display: 'flex', gap: 8 }}>
        {configs.map((c, i) => (
          <button
            key={c.path}
            onClick={() => setSelected(i)}
            style={btn(i === selected ? 'primary' : 'ghost')}
          >
            {c.name}
          </button>
        ))}
      </div>

      {cfg && (
        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
          {/* Raw YAML */}
          <div style={{ flex: 1, minWidth: 300 }}>
            <h3 style={{ margin: '0 0 8px' }}>Config file</h3>
            <p style={{ margin: '0 0 6px', fontSize: 12, color: '#6b7280' }}>{cfg.path}</p>
            <pre style={{
              background: '#1e1e1e', color: '#d4d4d4', borderRadius: 6,
              padding: 12, fontSize: 12, overflowY: 'auto', maxHeight: 460,
              whiteSpace: 'pre', margin: 0,
            }}>
              {configToYaml(cfg.config)}
            </pre>
          </div>

          {/* Summary card */}
          <div style={{ minWidth: 220 }}>
            <h3 style={{ margin: '0 0 8px' }}>At a glance</h3>
            <SummaryCard config={cfg.config} />
            <div style={{ marginTop: 16 }}>
              <p style={{ fontSize: 12, color: '#6b7280', margin: '0 0 8px' }}>
                To train with this config, create a new experiment and fill in the values above,
                or launch directly from the terminal:
              </p>
              <pre style={{
                background: '#1e1e1e', color: '#86efac', borderRadius: 6,
                padding: 10, fontSize: 11, margin: 0, overflowX: 'auto',
              }}>
                {`cd assignment1-basics\npython -m cs336_basics.training_loop \\\n  ${cfg.path}`}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function configToYaml(cfg: Record<string, unknown>): string {
  // Simple YAML serializer for display
  const lines: string[] = [];
  for (const [k, v] of Object.entries(cfg)) {
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      lines.push(`${k}:`);
      for (const [sk, sv] of Object.entries(v as Record<string, unknown>)) {
        lines.push(`  ${sk}: ${JSON.stringify(sv)}`);
      }
    } else {
      lines.push(`${k}: ${JSON.stringify(v)}`);
    }
  }
  return lines.join('\n');
}

function SummaryCard({ config }: { config: Record<string, unknown> }) {
  const model = (config.model as Record<string, number>) ?? {};
  const opt = (config.optimizer as Record<string, unknown>) ?? {};
  const sched = (config.scheduler as Record<string, number>) ?? {};

  const numParams = estimateParams(model);

  const rows: [string, string][] = [
    ['Device', String(config.device ?? '')],
    ['Dataset (train)', String(config.train_data_path ?? '')],
    ['Max iters', String(config.max_iters ?? '')],
    ['Batch size', String(config.batch_size ?? '')],
    ['~Params', numParams],
    ['Vocab size', String(model.vocab_size ?? '')],
    ['Context length', String(model.context_length ?? '')],
    ['Layers / d_model', `${model.num_layers} / ${model.d_model}`],
    ['Heads / d_ff', `${model.num_heads} / ${model.d_ff}`],
    ['LR', String(opt.lr ?? '')],
    ['Warmup iters', String(sched.warmup_iters ?? '')],
  ];

  return (
    <table style={{ borderCollapse: 'collapse', fontSize: 13 }}>
      <tbody>
        {rows.map(([label, val]) => (
          <tr key={label}>
            <td style={{ color: '#6b7280', paddingRight: 16, paddingBottom: 4, whiteSpace: 'nowrap' }}>{label}</td>
            <td style={{ fontWeight: 600, paddingBottom: 4 }}>{val}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function estimateParams(m: Record<string, number>): string {
  if (!m.vocab_size) return '?';
  const { vocab_size, d_model, num_layers, num_heads, d_ff, context_length } = m;
  void num_heads; void context_length;
  // embedding + n_layers*(4*d_model^2 + 2*d_model*d_ff) + lm_head
  const embed = vocab_size * d_model;
  const per_layer = 4 * d_model * d_model + 2 * d_model * d_ff;
  const total = embed + num_layers * per_layer + vocab_size * d_model;
  if (total > 1e9) return `${(total / 1e9).toFixed(1)}B`;
  if (total > 1e6) return `${(total / 1e6).toFixed(1)}M`;
  return `${(total / 1e3).toFixed(0)}K`;
}

// ── Main page ─────────────────────────────────────────────────────────────────
const TABS = ['Prepare Data', 'Inspect', 'Default Configs'] as const;
type Tab = (typeof TABS)[number];

export function TokenizePage() {
  const [tab, setTab] = useState<Tab>('Prepare Data');

  return (
    <div>
      <h1 style={{ marginBottom: 20 }}>Tokenize</h1>

      <div style={{ display: 'flex', borderBottom: '2px solid #e5e7eb', marginBottom: 24, gap: 0 }}>
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              background: 'none',
              border: 'none',
              borderBottom: t === tab ? '2px solid #2563eb' : '2px solid transparent',
              marginBottom: -2,
              padding: '8px 20px',
              cursor: 'pointer',
              fontWeight: t === tab ? 700 : 400,
              color: t === tab ? '#2563eb' : '#6b7280',
              fontSize: 14,
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'Prepare Data' && <PrepareDataTab />}
      {tab === 'Inspect' && <InspectTab />}
      {tab === 'Default Configs' && <DefaultConfigTab />}
    </div>
  );
}
