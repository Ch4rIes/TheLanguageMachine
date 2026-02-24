import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator, List

from models import MetricPoint


def read_all_metrics(path: str) -> List[MetricPoint]:
    """Read all JSONL lines from a metrics file."""
    p = Path(path)
    if not p.exists():
        return []
    points = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            points.append(MetricPoint(**json.loads(line)))
        except Exception:
            pass
    return points


async def sse_metrics_generator(path: str, last_iter: int = -1) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted strings for new metric lines.
    Polls every 2s, stops after 30s of no new data.
    """
    p = Path(path)
    no_growth_count = 0
    max_no_growth = 15  # 15 * 2s = 30s

    seen_iters = set()

    # Seed with already-seen iterations up to last_iter
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                it = obj.get("iteration", -1)
                if it <= last_iter:
                    seen_iters.add((it, obj.get("train_loss") is not None, obj.get("val_loss") is not None))
            except Exception:
                pass

    prev_size = p.stat().st_size if p.exists() else 0

    while True:
        await asyncio.sleep(2)
        if not p.exists():
            no_growth_count += 1
            if no_growth_count >= max_no_growth:
                break
            continue

        current_size = p.stat().st_size
        if current_size == prev_size:
            no_growth_count += 1
            if no_growth_count >= max_no_growth:
                break
            continue

        no_growth_count = 0
        prev_size = current_size

        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                it = obj.get("iteration", -1)
                key = (it, obj.get("train_loss") is not None, obj.get("val_loss") is not None)
                if key in seen_iters:
                    continue
                if it <= last_iter:
                    seen_iters.add(key)
                    continue
                seen_iters.add(key)
                point = MetricPoint(**obj)
                yield f"data: {point.model_dump_json()}\n\n"
            except Exception:
                pass
