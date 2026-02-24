from typing import List, Optional

from pydantic import BaseModel


class ModelParams(BaseModel):
    vocab_size: int = 32000
    context_length: int = 256
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 1024
    theta: float = 10000.0


class OptimizerParams(BaseModel):
    lr: float = 1e-3
    betas: List[float] = [0.9, 0.999]
    eps: float = 1e-8
    weight_decay: float = 0.01


class SchedulerParams(BaseModel):
    warmup_iters: int = 100
    cosine_cycle_iters: int = 10000
    min_lr_ratio: float = 0.1


class ExperimentCreate(BaseModel):
    name: str
    train_data_path: str
    val_data_path: Optional[str] = None
    batch_size: int = 32
    max_iters: int = 10000
    grad_clip_norm: float = 1.0
    log_interval: int = 10
    val_interval: int = 100
    checkpoint_interval: int = 1000
    device: str = "cpu"
    tokenizer_path: Optional[str] = None
    model: ModelParams = ModelParams()
    optimizer: OptimizerParams = OptimizerParams()
    scheduler: SchedulerParams = SchedulerParams()


class ExperimentRecord(ExperimentCreate):
    id: str
    status: str  # pending | running | completed | failed | stopped
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    pid: Optional[int] = None
    config_path: str = ""
    metrics_file: str = ""
    checkpoint_dir: str = ""


class MetricPoint(BaseModel):
    iteration: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    lr: Optional[float] = None
    timestamp: float


class GenerateRequest(BaseModel):
    experiment_id: str
    checkpoint_path: str
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_p: float = 1.0


class GenerateResponse(BaseModel):
    text: str
