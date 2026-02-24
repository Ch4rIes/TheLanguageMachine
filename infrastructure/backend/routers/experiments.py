import secrets
import time
from pathlib import Path
from typing import List

import yaml
from fastapi import APIRouter, HTTPException, Request

from models import ExperimentCreate, ExperimentRecord

DATA_DIR = Path(__file__).parent.parent.parent / "data"

router = APIRouter()


def _store(request: Request):
    return request.app.state.store


def _pm(request: Request):
    return request.app.state.proc_manager


@router.get("/experiments", response_model=List[ExperimentRecord])
def list_experiments(request: Request):
    return _store(request).list_all()


@router.post("/experiments", response_model=ExperimentRecord)
def create_experiment(body: ExperimentCreate, request: Request):
    exp_id = secrets.token_hex(6)
    exp_dir = DATA_DIR / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log_path = DATA_DIR / "logs" / f"{exp_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch()

    config_path = exp_dir / "config.yaml"
    metrics_file = exp_dir / "metrics.jsonl"

    # Build config YAML
    cfg_data = {
        "train_data_path": body.train_data_path,
        "val_data_path": body.val_data_path,
        "batch_size": body.batch_size,
        "max_iters": body.max_iters,
        "grad_clip_norm": body.grad_clip_norm,
        "log_interval": body.log_interval,
        "val_interval": body.val_interval,
        "checkpoint_interval": body.checkpoint_interval,
        "checkpoint_dir": str(ckpt_dir),
        "device": body.device,
        "model": body.model.model_dump(),
        "optimizer": body.optimizer.model_dump(),
        "scheduler": body.scheduler.model_dump(),
    }
    config_path.write_text(yaml.dump(cfg_data, default_flow_style=False))

    record = ExperimentRecord(
        id=exp_id,
        status="pending",
        created_at=time.time(),
        config_path=str(config_path),
        metrics_file=str(metrics_file),
        checkpoint_dir=str(ckpt_dir),
        **body.model_dump(),
    )
    _store(request).save(record)
    return record


@router.get("/experiments/{exp_id}", response_model=ExperimentRecord)
def get_experiment(exp_id: str, request: Request):
    rec = _store(request).get(exp_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return rec


@router.delete("/experiments/{exp_id}")
def delete_experiment(exp_id: str, request: Request):
    rec = _store(request).get(exp_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    if rec.status == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running experiment")
    _store(request).delete(exp_id)
    return {"deleted": exp_id}


@router.post("/experiments/{exp_id}/launch", response_model=ExperimentRecord)
def launch_experiment(exp_id: str, request: Request):
    rec = _store(request).get(exp_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    if rec.status == "running":
        raise HTTPException(status_code=409, detail="Already running")
    _pm(request).launch(rec)
    return _store(request).get(exp_id)


@router.post("/experiments/{exp_id}/stop")
def stop_experiment(exp_id: str, request: Request):
    rec = _store(request).get(exp_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    _pm(request).stop(exp_id)
    return {"stopped": exp_id}


@router.get("/experiments/{exp_id}/status")
def experiment_status(exp_id: str, request: Request):
    rec = _store(request).get(exp_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return {"status": rec.status, "pid": rec.pid}


@router.get("/experiments/{exp_id}/checkpoints")
def list_checkpoints(exp_id: str, request: Request):
    rec = _store(request).get(exp_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    ckpt_dir = Path(rec.checkpoint_dir)
    if not ckpt_dir.exists():
        return []
    pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return [{"name": p.name, "path": str(p)} for p in pts]


@router.get("/experiments/{exp_id}/log")
def experiment_log(exp_id: str, request: Request, lines: int = 200):
    log_path = DATA_DIR / "logs" / f"{exp_id}.log"
    if not log_path.exists():
        return {"log": ""}
    text = log_path.read_text(errors="replace")
    tail = "\n".join(text.splitlines()[-lines:])
    return {"log": tail}
