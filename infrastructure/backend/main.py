from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from process_manager import SubprocessManager
from store import ExperimentStore

store = ExperimentStore()
proc_manager = SubprocessManager(store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    proc_manager.reconcile_on_startup()
    yield


app = FastAPI(title="CS336 Experiment Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach shared state so routers can access it
app.state.store = store
app.state.proc_manager = proc_manager

from routers import experiments, generate, metrics, tokenize  # noqa: E402

app.include_router(experiments.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(tokenize.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
