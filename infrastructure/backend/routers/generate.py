from fastapi import APIRouter, HTTPException, Request

from generate_worker import generate_async
from models import GenerateRequest, GenerateResponse

router = APIRouter()


def _store(request: Request):
    return request.app.state.store


@router.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest, request: Request):
    rec = _store(request).get(body.experiment_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Experiment not found")

    tokenizer_path = rec.tokenizer_path
    if not tokenizer_path:
        raise HTTPException(status_code=422, detail="No tokenizer_path set for this experiment")

    try:
        text = await generate_async(
            checkpoint_path=body.checkpoint_path,
            tokenizer_path=tokenizer_path,
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return GenerateResponse(text=text)
