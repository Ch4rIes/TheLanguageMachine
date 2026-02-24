import axios from 'axios';

const api = axios.create({ baseURL: 'http://localhost:8000/api/tokenize' });

export interface TokenizerFile {
  path: string;
  name: string;
}

export interface TokenizerInfo {
  vocab_size: number;
  num_merges: number;
  num_special_tokens: number;
  special_tokens: { id: number; text: string }[];
  sample_tokens: { id: number; text: string }[];
}

export interface TokenizeResult {
  token_ids: number[];
  tokens: { id: number; bytes: number[]; text: string }[];
  count: number;
}

export interface TokenizeTask {
  id: string;
  label: string;
  status: 'running' | 'completed' | 'failed';
  created_at: number;
  finished_at: number | null;
  log_path: string;
  pid: number | null;
  returncode: number | null;
}

export interface DefaultConfig {
  path: string;
  name: string;
  config: Record<string, unknown>;
}

export const tokenizeApi = {
  listTokenizers: () => api.get<TokenizerFile[]>('/tokenizers').then((r) => r.data),
  listConfigs: () => api.get<DefaultConfig[]>('/configs').then((r) => r.data),
  info: (tokenizer_path: string) =>
    api.get<TokenizerInfo>('/info', { params: { tokenizer_path } }).then((r) => r.data),
  encodeText: (tokenizer_path: string, text: string) =>
    api.post<TokenizeResult>('/encode-text', { tokenizer_path, text }).then((r) => r.data),
  decodeText: (tokenizer_path: string, token_ids: number[]) =>
    api.post<{ text: string }>('/decode-text', { tokenizer_path, token_ids }).then((r) => r.data),
  trainTokenizer: (body: {
    input_path: string;
    vocab_size: number;
    output_path: string;
    special_tokens: string;
  }) => api.post<TokenizeTask>('/train', body).then((r) => r.data),
  encodeDataset: (body: {
    tokenizer_path: string;
    input_path: string;
    output_path: string;
  }) => api.post<TokenizeTask>('/encode-dataset', body).then((r) => r.data),
  listTasks: () => api.get<TokenizeTask[]>('/tasks').then((r) => r.data),
  getTask: (id: string) => api.get<TokenizeTask>(`/tasks/${id}`).then((r) => r.data),
  taskLog: (id: string) =>
    api.get<{ log: string }>(`/tasks/${id}/log`).then((r) => r.data),
};
