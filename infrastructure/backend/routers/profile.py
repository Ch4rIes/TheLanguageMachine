import json
import os
import secrets
import sys
import threading
import time
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException

from models import StepProfilerRequest, StepProfilerRun

ASSIGNMENT_DIR = Path(__file__).parent.parent.parent.parent / "core"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RUNS_FILE = DATA_DIR / "profile_runs.json"
_runs_lock = threading.Lock()

router = APIRouter()


def _ensure_core_path() -> None:
    p = str(ASSIGNMENT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _read_runs() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RUNS_FILE.exists():
        return {}
    try:
        return json.loads(RUNS_FILE.read_text())
    except Exception:
        return {}


def _write_runs(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RUNS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, RUNS_FILE)


def _save_run(profile: dict) -> StepProfilerRun:
    run = StepProfilerRun(
        id=secrets.token_hex(6),
        created_at=time.time(),
        config=profile["config"],
        hardware=profile["hardware"],
        results=profile["results"],
    )
    with _runs_lock:
        data = _read_runs()
        data[run.id] = run.model_dump()
        _write_runs(data)
    return run


@router.post("/profile/step", response_model=StepProfilerRun)
def profile_step(body: StepProfilerRequest):
    _ensure_core_path()
    try:
        from language_machine.profile_step import StepProfileConfig, run_step_profile  # noqa: PLC0415

        profile = run_step_profile(StepProfileConfig(**body.model_dump()))
        return _save_run(profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{exc}\n{traceback.format_exc()}")


@router.get("/profile/runs", response_model=list[StepProfilerRun])
def list_profile_runs(limit: int = 50):
    with _runs_lock:
        runs = [StepProfilerRun(**value) for value in _read_runs().values()]
    runs.sort(key=lambda run: run.created_at, reverse=True)
    return runs[: max(1, min(limit, 200))]


@router.get("/profile/runs/{run_id}", response_model=StepProfilerRun)
def get_profile_run(run_id: str):
    with _runs_lock:
        run = _read_runs().get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Profile run not found")
    return StepProfilerRun(**run)


@router.delete("/profile/runs/{run_id}")
def delete_profile_run(run_id: str):
    with _runs_lock:
        data = _read_runs()
        if run_id not in data:
            raise HTTPException(status_code=404, detail="Profile run not found")
        del data[run_id]
        _write_runs(data)
    return {"deleted": run_id}
