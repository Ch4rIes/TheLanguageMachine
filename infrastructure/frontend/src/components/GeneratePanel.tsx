import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { generateApi } from '../api';
import type { Checkpoint } from '../types';
import { CheckpointSelector } from './CheckpointSelector';

interface Props {
  experimentId: string;
  checkpoints: Checkpoint[];
}

export function GeneratePanel({ experimentId, checkpoints }: Props) {
  const [ckptPath, setCkptPath] = useState('');
  const [prompt, setPrompt] = useState('');
  const [maxTokens, setMaxTokens] = useState(200);
  const [temperature, setTemperature] = useState(1.0);
  const [topP, setTopP] = useState(1.0);
  const [result, setResult] = useState('');

  const gen = useMutation({
    mutationFn: () =>
      generateApi.run({
        experiment_id: experimentId,
        checkpoint_path: ckptPath,
        prompt,
        max_new_tokens: maxTokens,
        temperature,
        top_p: topP,
      }),
    onSuccess: (data) => setResult(data.text),
  });

  return (
    <div style={{ marginTop: 24 }}>
      <h3>Text Generation</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxWidth: 640 }}>
        <label>
          Checkpoint:
          <div style={{ marginTop: 4 }}>
            <CheckpointSelector checkpoints={checkpoints} selected={ckptPath} onChange={setCkptPath} />
          </div>
        </label>
        <label>
          Prompt:
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            style={{ display: 'block', width: '100%', marginTop: 4, padding: 8, borderRadius: 4, border: '1px solid #d1d5db', fontFamily: 'monospace', fontSize: 13 }}
          />
        </label>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <label>
            Max tokens:
            <input
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              style={{ display: 'block', width: 80, marginTop: 4, padding: '4px 6px', borderRadius: 4, border: '1px solid #d1d5db' }}
            />
          </label>
          <label>
            Temperature:
            <input
              type="number"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              style={{ display: 'block', width: 80, marginTop: 4, padding: '4px 6px', borderRadius: 4, border: '1px solid #d1d5db' }}
            />
          </label>
          <label>
            Top-p:
            <input
              type="number"
              step="0.05"
              value={topP}
              onChange={(e) => setTopP(Number(e.target.value))}
              style={{ display: 'block', width: 80, marginTop: 4, padding: '4px 6px', borderRadius: 4, border: '1px solid #d1d5db' }}
            />
          </label>
        </div>
        <button
          onClick={() => gen.mutate()}
          disabled={!ckptPath || !prompt || gen.isPending}
          style={{
            alignSelf: 'flex-start',
            background: '#2563eb',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            padding: '8px 20px',
            cursor: !ckptPath || !prompt || gen.isPending ? 'not-allowed' : 'pointer',
            opacity: !ckptPath || !prompt || gen.isPending ? 0.6 : 1,
          }}
        >
          {gen.isPending ? 'Generating…' : 'Generate'}
        </button>
        {gen.isError && (
          <div style={{ color: '#dc2626' }}>
            Error: {(gen.error as Error).message}
          </div>
        )}
        {result && (
          <div>
            <strong>Output:</strong>
            <pre
              style={{
                background: '#f3f4f6',
                borderRadius: 4,
                padding: 12,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                fontSize: 13,
                marginTop: 8,
              }}
            >
              {result}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
