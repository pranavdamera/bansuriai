"""
BansuriAI-V2 — Central Configuration

Single source of truth for all audio, model, and pipeline constants.
Every module imports from here instead of defining magic numbers locally.

Organized into sections:
    AUDIO   — sample rate, duration limits, silence trimming
    FEATURE — spectrogram parameters (n_mels, n_fft, hop_length)
    MODEL   — architecture settings, class count, checkpoint path
    DECODER — smoothing window, minimum segment duration, confidence floor
    SERVER  — host, port, CORS origins
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path anchors
# ---------------------------------------------------------------------------
# Resolve paths relative to the backend/ directory regardless of where
# the process is started from.
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
PROJECT_ROOT = BACKEND_DIR.parent                             # BansuriAI-V2/

# ---------------------------------------------------------------------------
# AUDIO — raw signal conditioning
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 22050          # Resample all inputs to this rate (Hz)
MAX_DURATION_SEC: float = 30.0    # Reject files longer than this
MIN_DURATION_SEC: float = 0.5     # Reject files shorter than this
TRIM_TOP_DB: int = 20             # librosa.effects.trim threshold (dB)
NORMALIZE: bool = True            # Peak-normalize waveform to [-1, 1]

# ---------------------------------------------------------------------------
# FEATURE — spectrogram extraction
# ---------------------------------------------------------------------------
N_FFT: int = 2048                 # FFT window size (samples)
HOP_LENGTH: int = 512             # Hop between frames (samples)
N_MELS: int = 128                 # Number of Mel filter banks
FMIN: float = 50.0                # Lowest Mel frequency (Hz) — captures bansuri low Sa
FMAX: float = 4000.0              # Highest Mel frequency (Hz) — bansuri upper range
REF_DB: float = 80.0              # Reference for power_to_db normalization

# Derived: frame duration in seconds
FRAME_DURATION_SEC: float = HOP_LENGTH / SAMPLE_RATE  # ~0.023 sec per frame

# ---------------------------------------------------------------------------
# MODEL — architecture and checkpoint
# ---------------------------------------------------------------------------
NUM_CLASSES: int = 7              # Sa, Re, Ga, Ma, Pa, Dha, Ni (no silence class yet)
MODEL_PATH: Path = BACKEND_DIR / "saved_models" / "bansuri_note_model.pt"

# ---------------------------------------------------------------------------
# INFERENCE — sliding window over the spectrogram
# ---------------------------------------------------------------------------
# FIXED_NUM_FRAMES must match training/config.py FIXED_NUM_FRAMES exactly.
# The model was trained on spectrograms of this width. At inference time we
# slide a window of this size across the full spectrogram, producing one
# prediction per window position.
FIXED_NUM_FRAMES: int = 64       # Frames per window (~1.49 sec at hop=512, sr=22050)
INFERENCE_WINDOW_HOP: int = 32   # Step between windows (50% overlap → dense predictions)

# ---------------------------------------------------------------------------
# DECODER — sequence post-processing
# ---------------------------------------------------------------------------
SMOOTHING_WINDOW: int = 5         # Median filter kernel size (frames)
MIN_SEGMENT_DURATION_SEC: float = 0.05   # Discard segments shorter than 50ms
CONFIDENCE_THRESHOLD: float = 0.30       # Discard segments below this confidence

# ---------------------------------------------------------------------------
# SERVER — FastAPI / Uvicorn
# ---------------------------------------------------------------------------
HOST: str = os.getenv("BANSURI_HOST", "0.0.0.0")
PORT: int = int(os.getenv("BANSURI_PORT", "8000"))
CORS_ORIGINS: list[str] = [
    "http://localhost:5173",   # Vite dev server default
    "http://localhost:3000",   # Common React dev port
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# ---------------------------------------------------------------------------
# TEMP — uploaded file handling
# ---------------------------------------------------------------------------
TEMP_UPLOAD_DIR: Path = BACKEND_DIR / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MB max upload size

# ---------------------------------------------------------------------------
# FRONTEND — static file serving for production mode
# ---------------------------------------------------------------------------
# When the frontend is built (npm run build), the output goes here.
# FastAPI serves these files so the whole app runs from a single server.
FRONTEND_DIST_DIR: Path = PROJECT_ROOT / "frontend" / "dist"
