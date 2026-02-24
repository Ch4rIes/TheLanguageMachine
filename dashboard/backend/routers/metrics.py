from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from metrics_reader import read_all_metrics, sse_metrics_generator

router = APIRouter()


def _store(request: Request):
    return request.app.state.store


@router.get("/experiments/{exp_id}/metrics")
def get_metrics(exp_id: str, request: Request):
    rec = _store(request).get(exp_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    points = read_all_metrics(rec.metrics_file)
    return [p.model_dump() for p in points]


@router.get("/experiments/{exp_id}/metrics/stream")
async def stream_metrics(exp_id: str, request: Request, last_iter: int = -1):
    rec = _store(request).get(exp_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")

    async def generator():
        async for event in sse_metrics_generator(rec.metrics_file, last_iter):
            if await request.is_disconnected():
                break
            yield event

    return EventSourceResponse(generator())
