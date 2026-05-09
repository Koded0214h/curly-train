from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from model_store import generate_text, load_bundle, stream_generation


FRONTEND_DIST = ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_bundle()
    yield


app = FastAPI(title="Curly Train API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(default="Ayanokoji-kun,")
    max_new_tokens: int = Field(default=120, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0.0, le=5.0)
    top_k: int = Field(default=50, ge=1, le=2000)


def _sse(data: dict[str, object]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/health")
def health() -> dict[str, object]:
    bundle = load_bundle()
    config = bundle["config"]
    return {
        "ok": True,
        "checkpoint": bundle["checkpoint_path"].name,
        "device": str(bundle["device"]),
        "config": config.__dict__,
    }


@app.get("/api/model")
def model_info() -> dict[str, object]:
    bundle = load_bundle()
    config = bundle["config"]
    return {
        "checkpoint": bundle["checkpoint_path"].name,
        "device": str(bundle["device"]),
        "config": config.__dict__,
    }


@app.post("/api/generate")
def generate(request: GenerateRequest) -> dict[str, object]:
    try:
        return generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/generate/stream")
def generate_stream(
    prompt: str = "Ayanokoji-kun,",
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
):
    def iterator():
        try:
            for payload in stream_generation(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                yield _sse(payload)
        except Exception as exc:
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(iterator(), media_type="text/event-stream")


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
