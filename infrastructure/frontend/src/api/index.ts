import axios from 'axios';
import type {
  Checkpoint,
  ExperimentCreate,
  ExperimentRecord,
  GenerateRequest,
  GenerateResponse,
  MetricPoint,
  StepProfilerRequest,
  StepProfilerRun,
} from '../types';

const api = axios.create({ baseURL: 'http://localhost:8000/api' });

export const experimentsApi = {
  list: () => api.get<ExperimentRecord[]>('/experiments').then((r) => r.data),
  get: (id: string) => api.get<ExperimentRecord>(`/experiments/${id}`).then((r) => r.data),
  create: (body: ExperimentCreate) =>
    api.post<ExperimentRecord>('/experiments', body).then((r) => r.data),
  patch: (id: string, body: Partial<Pick<ExperimentRecord, 'name' | 'train_data_path' | 'val_data_path' | 'tokenizer_path' | 'device'>>) =>
    api.patch<ExperimentRecord>(`/experiments/${id}`, body).then((r) => r.data),
  delete: (id: string) => api.delete(`/experiments/${id}`).then((r) => r.data),
  launch: (id: string) =>
    api.post<ExperimentRecord>(`/experiments/${id}/launch`).then((r) => r.data),
  stop: (id: string) => api.post(`/experiments/${id}/stop`).then((r) => r.data),
  status: (id: string) =>
    api.get<{ status: string; pid?: number }>(`/experiments/${id}/status`).then((r) => r.data),
  checkpoints: (id: string) =>
    api.get<Checkpoint[]>(`/experiments/${id}/checkpoints`).then((r) => r.data),
  log: (id: string, lines = 200) =>
    api.get<{ log: string }>(`/experiments/${id}/log`, { params: { lines } }).then((r) => r.data),
};

export const metricsApi = {
  snapshot: (id: string) =>
    api.get<MetricPoint[]>(`/experiments/${id}/metrics`).then((r) => r.data),
};

export const generateApi = {
  run: (body: GenerateRequest) =>
    api.post<GenerateResponse>('/generate', body).then((r) => r.data),
};

export const profilerApi = {
  runStep: (body: StepProfilerRequest) =>
    api.post<StepProfilerRun>('/profile/step', body).then((r) => r.data),
  listRuns: (limit = 50) =>
    api.get<StepProfilerRun[]>('/profile/runs', { params: { limit } }).then((r) => r.data),
  getRun: (id: string) =>
    api.get<StepProfilerRun>(`/profile/runs/${id}`).then((r) => r.data),
  deleteRun: (id: string) =>
    api.delete<{ deleted: string }>(`/profile/runs/${id}`).then((r) => r.data),
};
