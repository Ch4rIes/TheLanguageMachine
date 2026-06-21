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

export interface StepProfilerRequest {
  device: string;
  batch_size: number;
  warmup_steps: number;
  profile_steps: number;
  vocab_size: number;
  context_length: number;
  num_layers: number;
  d_model: number;
  num_heads: number;
  d_ff: number;
  theta: number;
  lr: number;
  weight_decay: number;
}

export interface StepProfilerResult {
  mode: 'forward' | 'forward_backward' | 'forward_backward_optimizer';
  steps: number;
  parameter_count: number;
  tokens_per_step: number;
  mean_ms: number;
  p50_ms: number;
  p95_ms: number;
  min_ms: number;
  max_ms: number;
  tokens_per_sec: number;
  allocated_mb?: number | null;
  reserved_mb?: number | null;
  peak_allocated_mb?: number | null;
  peak_reserved_mb?: number | null;
  last_loss?: number;
}

export interface StepProfilerResponse {
  config: StepProfilerRequest;
  hardware: {
    platform: string;
    machine: string;
    python: string;
    torch: string;
    device: string;
    cpu: {
      model: string;
      physical_cores?: number | null;
      logical_cores?: number | null;
      memory_gb: number;
    };
    gpu?: {
      backend: string;
      name: string;
      index?: number;
      total_memory_mb?: number | null;
      cuda_version?: string | null;
      capability?: string | null;
      multi_processor_count?: number;
    } | null;
  };
  results: StepProfilerResult[];
}

export interface StepProfilerRun extends StepProfilerResponse {
  id: string;
  created_at: number;
}
