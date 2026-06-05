"""FastAPI web server for the Amazon Returns/Refunds RAG chatbot.

Serves a single-page UI and exposes POST /api/ask wrapping AmazonRAGChatbot.ask.
Chatbot is loaded once at process startup so requests are warm.

Run:
    uvicorn webapp.server:app --reload --port 8000
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is importable so `from src.chat import ...` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chat import AmazonRAGChatbot  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Grounded Policy QA — Amazon Returns",
    description="Hybrid retrieval (BM25 + FAISS) + cross-encoder rerank + Claude, with cited decisions.",
    version="1.0.0",
)

# Mount static assets (single HTML page lives here)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Lazy singleton container
_state = {"bot": None, "load_seconds": None}


@app.on_event("startup")
def _load_bot() -> None:
    t0 = time.time()
    print("[webapp] initializing AmazonRAGChatbot…")
    _state["bot"] = AmazonRAGChatbot(use_reranker=True)
    _state["load_seconds"] = round(time.time() - t0, 2)
    print(f"[webapp] chatbot ready in {_state['load_seconds']}s "
          f"(provider={_state['bot'].provider_info})")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(None, max_length=128)
    k_candidates: int = Field(20, ge=1, le=100)
    k_final: int = Field(5, ge=1, le=20)


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


@app.get("/")
def root() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="index.html missing")
    return FileResponse(str(index))


@app.get("/api/health")
def health() -> dict:
    bot = _state["bot"]
    return {
        "ok": bot is not None,
        "provider": getattr(bot, "provider_info", None) if bot else None,
        "use_llm": getattr(bot, "use_llm", None) if bot else None,
        "load_seconds": _state["load_seconds"],
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> JSONResponse:
    bot = _state["bot"]
    if bot is None:
        raise HTTPException(status_code=503, detail="Chatbot still warming up")
    t0 = time.time()
    try:
        result = bot.ask(
            req.question,
            session_id=req.session_id,
            k_candidates=req.k_candidates,
            k_final=req.k_final,
        )
    except Exception as e:  # surface errors as JSON so the UI can display them
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    result["latency_ms"] = int((time.time() - t0) * 1000)
    return JSONResponse(result)


@app.post("/api/reset")
def reset(req: ResetRequest) -> dict:
    bot = _state["bot"]
    if bot is None:
        raise HTTPException(status_code=503, detail="Chatbot still warming up")
    bot.reset_session(req.session_id)
    return {"ok": True, "session_id": req.session_id, "cleared": True}
