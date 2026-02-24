import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from models import ExperimentRecord

DATA_DIR = Path(__file__).parent.parent / "data"
STORE_FILE = DATA_DIR / "experiments.json"


class ExperimentStore:
    def __init__(self):
        self._lock = threading.Lock()
        STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not STORE_FILE.exists():
            STORE_FILE.write_text("{}")

    def _read(self) -> Dict[str, dict]:
        try:
            return json.loads(STORE_FILE.read_text())
        except Exception:
            return {}

    def _write(self, data: Dict[str, dict]) -> None:
        tmp = STORE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, STORE_FILE)

    def list_all(self) -> List[ExperimentRecord]:
        with self._lock:
            data = self._read()
            return [ExperimentRecord(**v) for v in data.values()]

    def get(self, exp_id: str) -> Optional[ExperimentRecord]:
        with self._lock:
            data = self._read()
            if exp_id not in data:
                return None
            return ExperimentRecord(**data[exp_id])

    def save(self, record: ExperimentRecord) -> None:
        with self._lock:
            data = self._read()
            data[record.id] = record.model_dump()
            self._write(data)

    def delete(self, exp_id: str) -> bool:
        with self._lock:
            data = self._read()
            if exp_id not in data:
                return False
            del data[exp_id]
            self._write(data)
            return True
