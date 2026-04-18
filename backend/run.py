"""
BansuriAI-V2 — Server Launcher

Run this script to start the backend:
    python run.py

Or use uvicorn directly:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn

from app.utils.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,        # Auto-reload on code changes during development
        log_level="info",
    )
