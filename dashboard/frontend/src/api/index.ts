import axios from 'axios';
import type {
  Checkpoint,
  ExperimentCreate,
  ExperimentRecord,
  GenerateRequest,
  GenerateResponse,
  MetricPoint,
} from '../types';

const api = axios.create({ baseURL: 'http://localhost:8000/api' });

export const experimentsApi = {
  list: () => api.get<ExperimentRecord[]>('/experiments').then((r) => r.data),
  get: (id: string) => api.get<ExperimentRecord>(`/experiments/${id}`).then((r) => r.data),
  create: (body: ExperimentCreate) =>
    api.post<ExperimentRecord>('/experiments', body).then((r) => r.data),
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
