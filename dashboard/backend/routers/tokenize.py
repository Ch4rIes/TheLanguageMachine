"""
Tokenization endpoints:
  GET  /api/tokenize/tokenizers          - list .json files in common dirs
  GET  /api/tokenize/info                - vocab/merge counts for a tokenizer
  POST /api/tokenize/encode-text         - encode text → token ids (in-process)
  POST /api/tokenize/decode-text         - decode ids → text (in-process)
  POST /api/tokenize/train               - spawn BPE training subprocess
  POST /api/tokenize/encode-dataset      - spawn dataset encoding subprocess
  GET  /api/tokenize/tasks               - list running/completed tasks
  GET  /api/tokenize/tasks/{id}/log      - tail log of a task
"""
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/tokenize")

# ── path constants ──────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.parent
DASHBOARD_DIR = BACKEND_DIR.parent
ASSIGNMENT_DIR = DASHBOARD_DIR.parent / "assignment1-basics"
SCRIPTS_DIR = BACKEND_DIR / "scripts"
LOGS_DIR = DASHBOARD_DIR / "data" / "logs" / "tokenize"
CONFIGS_DIR = DASHBOARD_DIR / "configs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── in-memory task store ─────────────────────────────────────────────────────
_tasks: Dict[str, dict] = {}
_tasks_lock = threading.Lock()


def _run_task(task_id: str, cmd: list[str], cwd: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        with _tasks_lock:
            _tasks[task_id]["pid"] = proc.pid
        proc.wait()
        with _tasks_lock:
            _tasks[task_id]["status"] = "completed" if proc.returncode == 0 else "failed"
            _tasks[task_id]["finished_at"] = time.time()
            _tasks[task_id]["returncode"] = proc.returncode


def _spawn(label: str, cmd: list[str], cwd: str) -> dict:
    task_id = secrets.token_hex(5)
    log_path = LOGS_DIR / f"{task_id}.log"
    task = {
        "id": task_id,
        "label": label,
        "status": "running",
        "created_at": time.time(),
        "finished_at": None,
        "log_path": str(log_path),
        "pid": None,
        "returncode": None,
    }
    with _tasks_lock:
        _tasks[task_id] = task
    t = threading.Thread(target=_run_task, args=(task_id, cmd, cwd, log_path), daemon=True)
    t.start()
    return task


# ── pydantic models ──────────────────────────────────────────────────────────
class TrainTokenizerRequest(BaseModel):
    input_path: str
    vocab_size: int = 10000
    output_path: str
    special_tokens: str = "<|endoftext|>"


class EncodeDatasetRequest(BaseModel):
    tokenizer_path: str
    input_path: str
    output_path: str


class EncodeTextRequest(BaseModel):
    tokenizer_path: str
    text: str


class DecodeTextRequest(BaseModel):
    tokenizer_path: str
    token_ids: List[int]


# ── helpers ──────────────────────────────────────────────────────────────────
def _load_tokenizer(path: str):
    import json
    with open(path) as f:
        data = json.load(f)
    vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
    merges = [(bytes(a), bytes(b)) for a, b in data["merges"]]
    return vocab, merges


def _list_json_files(*dirs: Path) -> List[str]:
    paths = []
    for d in dirs:
        if d.exists():
            paths.extend(str(p) for p in sorted(d.glob("**/*.json")))
    return paths


# ── routes ────────────────────────────────────────────────────────────────────
@router.get("/tokenizers")
def list_tokenizers():
    """List all .json tokenizer files in common locations."""
    paths = _list_json_files(
        ASSIGNMENT_DIR / "tokenizers",
        DASHBOARD_DIR / "data" / "tokenizers",
    )
    return [{"path": p, "name": Path(p).name} for p in paths]


@router.get("/configs")
def list_configs():
    """List default YAML configs with parsed values."""
    result = []
    for p in sorted(CONFIGS_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text())
            result.append({"path": str(p), "name": p.stem, "config": data})
        except Exception:
            pass
    return result


@router.get("/info")
def tokenizer_info(tokenizer_path: str):
    try:
        vocab, merges = _load_tokenizer(tokenizer_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    num_special = len(vocab) - 256 - len(merges)
    special = []
    for i in range(256, 256 + num_special):
        if i in vocab:
            try:
                special.append({"id": i, "text": vocab[i].decode("utf-8")})
            except Exception:
                special.append({"id": i, "text": repr(vocab[i])})

    # Sample of vocab tokens (non-byte, non-special)
    samples = []
    for i in range(256 + num_special, min(256 + num_special + 20, len(vocab))):
        if i in vocab:
            try:
                samples.append({"id": i, "text": vocab[i].decode("utf-8", errors="replace")})
            except Exception:
                pass

    return {
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "num_special_tokens": num_special,
        "special_tokens": special,
        "sample_tokens": samples,
    }


@router.post("/encode-text")
def encode_text(body: EncodeTextRequest):
    try:
        # Lazy import to avoid metadata error if not installed
        if str(ASSIGNMENT_DIR) not in sys.path:
            sys.path.insert(0, str(ASSIGNMENT_DIR))
        from language_machine.generate import encode  # noqa: PLC0415
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import error: {e}")

    try:
        vocab, merges = _load_tokenizer(body.tokenizer_path)
        ids = encode(body.text, vocab, merges)
        tokens = [{"id": tid, "bytes": list(vocab.get(tid, b"")), "text": vocab.get(tid, b"").decode("utf-8", errors="replace")} for tid in ids]
        return {"token_ids": ids, "tokens": tokens, "count": len(ids)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/decode-text")
def decode_text(body: DecodeTextRequest):
    try:
        vocab, _ = _load_tokenizer(body.tokenizer_path)
        text_bytes = b"".join(vocab.get(tid, b"") for tid in body.token_ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/train")
def train_tokenizer(body: TrainTokenizerRequest):
    """Spawn BPE training subprocess. Returns task immediately."""
    cmd = [
        sys.executable, "-m", "language_machine.tokenizer_cli",
        "train",
        "-i", body.input_path,
        "-v", str(body.vocab_size),
        "-o", body.output_path,
        "-s", body.special_tokens,
    ]
    task = _spawn(f"Train tokenizer → {Path(body.output_path).name}", cmd, str(ASSIGNMENT_DIR))
    return task


@router.post("/encode-dataset")
def encode_dataset(body: EncodeDatasetRequest):
    """Spawn dataset encoding subprocess. Returns task immediately."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "encode_dataset.py"),
        "--tokenizer", body.tokenizer_path,
        "--input", body.input_path,
        "--output", body.output_path,
    ]
    task = _spawn(f"Encode {Path(body.input_path).name} → {Path(body.output_path).name}", cmd, str(ASSIGNMENT_DIR))
    return task


@router.get("/tasks")
def list_tasks():
    with _tasks_lock:
        return list(_tasks.values())


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/tasks/{task_id}/log")
def task_log(task_id: str, lines: int = 200):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    log_path = Path(task["log_path"])
    if not log_path.exists():
        return {"log": ""}
    text = log_path.read_text(errors="replace")
    tail = "\n".join(text.splitlines()[-lines:])
    return {"log": tail}
