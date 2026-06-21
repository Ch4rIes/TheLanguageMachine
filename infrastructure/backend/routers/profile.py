import json
import os
import secrets
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, HTTPException

from models import StepProfilerRequest, StepProfilerRun

ASSIGNMENT_DIR = Path(__file__).parent.parent.parent.parent / "core"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RUNS_FILE = DATA_DIR / "profile_runs.json"
_runs_lock = threading.Lock()

router = APIRouter()


def _remote_profile(body: StepProfilerRequest) -> dict | None:
    runpod_endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
    runpod_api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if runpod_endpoint_id or runpod_api_key:
        return _runpod_profile(body, runpod_endpoint_id, runpod_api_key)

    worker_url = os.environ.get("PROFILE_WORKER_URL", "").strip().rstrip("/")
    if not worker_url:
        return None

    timeout = float(os.environ.get("PROFILE_WORKER_TIMEOUT", "600"))
    endpoint = "/profile/step" if worker_url.endswith("/api") else "/api/profile/step"
    url = f"{worker_url}{endpoint}"
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("PROFILE_WORKER_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = json.dumps(body.model_dump()).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=f"Profile worker failed: {detail}")
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Profile worker unavailable: {exc.reason}")


def _runpod_profile(body: StepProfilerRequest, endpoint_id: str, api_key: str) -> dict:
    if not endpoint_id or not api_key:
        raise HTTPException(status_code=500, detail="Both RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY must be set")

    timeout = float(os.environ.get("RUNPOD_TIMEOUT", os.environ.get("PROFILE_WORKER_TIMEOUT", "600")))
    execution_timeout_ms = int(float(os.environ.get("RUNPOD_EXECUTION_TIMEOUT_MS", str(timeout * 1000))))
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    payload = {
        "input": body.model_dump(),
        "policy": {
            "executionTimeout": execution_timeout_ms,
            "ttl": max(execution_timeout_ms + 300000, 900000),
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=f"RunPod profiler failed: {detail}")
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"RunPod profiler unavailable: {exc.reason}")

    if "error" in data:
        raise HTTPException(status_code=502, detail=f"RunPod profiler failed: {data['error']}")
    output = data.get("output")
    if output is None:
        raise HTTPException(status_code=502, detail=f"RunPod profiler returned no output: {data}")
    if isinstance(output, dict) and "error" in output:
        raise HTTPException(status_code=502, detail=f"RunPod profiler failed: {output['error']}")
    return output


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
    remote_profile = _remote_profile(body)
    if remote_profile is not None:
        return _save_run(
            {
                "config": remote_profile["config"],
                "hardware": remote_profile["hardware"],
                "results": remote_profile["results"],
            }
        )

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
