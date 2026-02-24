import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

DATA_DIR = Path(__file__).parent.parent / "data"
# Path to the assignment1-basics directory (parent of dashboard)
ASSIGNMENT_DIR = Path(__file__).parent.parent.parent / "assignment1-basics"


class SubprocessManager:
    def __init__(self, store):
        self._store = store
        self._procs: dict[str, subprocess.Popen] = {}

    def launch(self, record) -> subprocess.Popen:
        log_path = DATA_DIR / "logs" / f"{record.id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w")

        cmd = [
            sys.executable,
            "-m",
            "language_machine.training_loop",
            record.config_path,
            "--metrics-file",
            record.metrics_file,
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=str(ASSIGNMENT_DIR),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._procs[record.id] = proc

        record.status = "running"
        record.started_at = time.time()
        record.pid = proc.pid
        self._store.save(record)

        # Start a watcher thread to update status when done
        import threading

        def _watch(exp_id: str, p: subprocess.Popen, log_file) -> None:
            p.wait()
            log_file.close()
            rec = self._store.get(exp_id)
            if rec and rec.status == "running":
                rec.status = "completed" if p.returncode == 0 else "failed"
                rec.finished_at = time.time()
                self._store.save(rec)
            self._procs.pop(exp_id, None)

        t = threading.Thread(target=_watch, args=(record.id, proc, log_fh), daemon=True)
        t.start()

        return proc

    def stop(self, exp_id: str) -> None:
        proc = self._procs.get(exp_id)
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except ProcessLookupError:
            pass

        rec = self._store.get(exp_id)
        if rec:
            rec.status = "stopped"
            rec.finished_at = time.time()
            self._store.save(rec)
        self._procs.pop(exp_id, None)

    def poll(self, exp_id: str) -> str | None:
        """Return current returncode or None if still running."""
        proc = self._procs.get(exp_id)
        if proc is None:
            return None
        return proc.poll()

    def reconcile_on_startup(self) -> None:
        """Mark stale 'running' experiments as failed if their PID is gone."""
        for rec in self._store.list_all():
            if rec.status == "running":
                alive = False
                if rec.pid is not None and HAS_PSUTIL:
                    alive = psutil.pid_exists(rec.pid)
                if not alive:
                    rec.status = "failed"
                    rec.finished_at = time.time()
                    self._store.save(rec)
