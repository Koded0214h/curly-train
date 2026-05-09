from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.model_store import generate_text, load_bundle


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="Curly Train API", version="0.1.0")

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


@app.on_event("startup")
def _warm_model() -> None:
    load_bundle()


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


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

