"""
BansuriAI-V2 — FastAPI Application Entry Point

Creates and configures the FastAPI app:
    - CORS middleware for frontend dev server access
    - Lifespan handler that loads the ML model once at startup
    - /health endpoint for connection checks and model status
    - /analyze endpoint (from routes/analyze.py)
    - Static file serving for the production frontend build

Two operating modes:
    DEV MODE:   Frontend runs on Vite (port 5173), Vite proxies API
                calls to this server (port 8000). CORS allows Vite's origin.

    PROD MODE:  Frontend is built (npm run build → frontend/dist/).
                This server serves those static files AND the API,
                so the user runs only one server. CORS is not needed
                because everything is same-origin.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.routes.analyze import router as analyze_router
from app.routes.classify import router as classify_router
from app.schemas.analysis import HealthResponse
from app.services.model_inference import is_model_loaded, load_model
from app.utils.config import CORS_ORIGINS, FRONTEND_DIST_DIR

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup/shutdown) ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, clean up on shutdown."""
    logger.info("=" * 60)
    logger.info("BansuriAI-V2 starting up...")
    logger.info("=" * 60)

    # Load model into memory
    loaded = load_model()
    if loaded:
        logger.info("Model loaded successfully — production mode")
    else:
        logger.warning("Model NOT loaded — running in placeholder mode")
        logger.warning("Predictions will be random until a trained model is provided.")

    # Check for frontend build
    index_html = FRONTEND_DIST_DIR / "index.html"
    if index_html.exists():
        logger.info(f"Frontend build found at {FRONTEND_DIST_DIR}")
        logger.info("Serving frontend at http://localhost:8000")
    else:
        logger.info(f"No frontend build at {FRONTEND_DIST_DIR}")
        logger.info("Run 'cd frontend && npm run build' to enable production serving")
        logger.info("For dev mode, run 'cd frontend && npm run dev' separately")

    logger.info("Startup complete. Ready to accept requests.")

    yield  # App runs here

    # Shutdown
    logger.info("BansuriAI-V2 shutting down...")


# ── Create the FastAPI app ────────────────────────────────────────────
app = FastAPI(
    title="BansuriAI-V2",
    description=(
        "AI-powered bansuri note recognition API. "
        "Upload a .wav recording and receive detected notes, "
        "timestamps, confidence scores, and a summary report."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── CORS middleware ───────────────────────────────────────────────────
# Required for dev mode when the React frontend runs on a different port.
# In production mode (FastAPI serving static files), requests are
# same-origin so CORS headers are not needed but do no harm.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health endpoint ───────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Check if the service is running and the model is loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=is_model_loaded(),
    )


# ── Mount API routes ──────────────────────────────────────────────────
app.include_router(analyze_router, tags=["Analysis"])
app.include_router(classify_router, tags=["Real-time"])


# ── Production: serve frontend static files ───────────────────────────
# This block mounts the built frontend AFTER the API routes, so API
# endpoints take priority. The catch-all at the end serves index.html
# for any path that doesn't match an API route (client-side routing).

_frontend_index = FRONTEND_DIST_DIR / "index.html"

if FRONTEND_DIST_DIR.exists() and _frontend_index.exists():
    # Serve JS, CSS, and other assets from /assets/
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")),
        name="frontend-assets",
    )

    # Serve other root-level static files (favicon, etc.)
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        fav = FRONTEND_DIST_DIR / "favicon.ico"
        if fav.exists():
            return FileResponse(str(fav))
        return JSONResponse(status_code=404, content={})

    # Catch-all: serve index.html for any non-API, non-asset path.
    # This enables client-side routing (if added later).
    @app.get("/{path:path}", include_in_schema=False)
    async def serve_frontend(request: Request, path: str):
        # Don't intercept API docs
        if path in ("docs", "redoc", "openapi.json"):
            return JSONResponse(status_code=404, content={})
        return FileResponse(str(_frontend_index))
