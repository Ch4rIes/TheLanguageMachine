export interface ModelParams {
  vocab_size: number;
  context_length: number;
  num_layers: number;
  d_model: number;
  num_heads: number;
  d_ff: number;
  theta: number;
}

export interface OptimizerParams {
  lr: number;
  betas: [number, number];
  eps: number;
  weight_decay: number;
}

export interface SchedulerParams {
  warmup_iters: number;
  cosine_cycle_iters: number;
  min_lr_ratio: number;
}

export interface ExperimentCreate {
  name: string;
  train_data_path: string;
  val_data_path?: string;
  batch_size: number;
  max_iters: number;
  grad_clip_norm: number;
  log_interval: number;
  val_interval: number;
  checkpoint_interval: number;
  device: string;
  tokenizer_path?: string;
  model: ModelParams;
  optimizer: OptimizerParams;
  scheduler: SchedulerParams;
}

export interface ExperimentRecord extends ExperimentCreate {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  created_at: number;
  started_at?: number;
  finished_at?: number;
  pid?: number;
  config_path: string;
  metrics_file: string;
  checkpoint_dir: string;
}

export interface MetricPoint {
  iteration: number;
  train_loss?: number;
  val_loss?: number;
  lr?: number;
  timestamp: number;
}

export interface Checkpoint {
  name: string;
  path: string;
}

export interface GenerateRequest {
  experiment_id: string;
  checkpoint_path: string;
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_p: number;
}

export interface GenerateResponse {
  text: string;
}
