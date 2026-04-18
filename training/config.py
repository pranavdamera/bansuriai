"""
BansuriAI-V2 — Training Configuration

All hyperparameters and constants for the training pipeline, collected in
one file so experiments are reproducible and tunable from a single place.

This file is SELF-CONTAINED — it does NOT import from backend/app/.
The audio/feature constants are intentionally duplicated here so the
training pipeline can run independently of the backend. If you change
a value here, update backend/app/utils/config.py to match before deploying.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # BansuriAI-V2/
DATASET_DIR = PROJECT_ROOT / "dataset"
SPLITS_DIR = DATASET_DIR / "splits"
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "backend" / "saved_models"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_FILENAME = "bansuri_note_model.pt"

LOG_DIR = PROJECT_ROOT / "training" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# AUDIO — MUST MATCH backend/app/utils/config.py
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 22050
TRIM_TOP_DB: int = 20
NORMALIZE_AUDIO: bool = True

# ---------------------------------------------------------------------------
# FEATURES — MUST MATCH backend/app/utils/config.py
# ---------------------------------------------------------------------------
N_FFT: int = 2048
HOP_LENGTH: int = 512
N_MELS: int = 128
FMIN: float = 50.0
FMAX: float = 4000.0
REF_DB: float = 80.0

# Fixed spectrogram width for CNN input.
# 64 frames × 512 hop / 22050 sr ≈ 1.49 seconds of audio.
FIXED_NUM_FRAMES: int = 64

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
NUM_CLASSES: int = 7
SWARA_LABELS: list[str] = ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"]
DROPOUT: float = 0.3

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 50
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
EARLY_STOPPING_PATIENCE: int = 10
SCHEDULER_PATIENCE: int = 5
SCHEDULER_FACTOR: float = 0.5
NUM_WORKERS: int = 0
SEED: int = 42

# ---------------------------------------------------------------------------
# AUGMENTATION
# ---------------------------------------------------------------------------
AUGMENT_ENABLED: bool = True
TIME_SHIFT_MAX_SAMPLES: int = 2000
NOISE_ENABLED: bool = True
NOISE_SNR_DB_MIN: float = 15.0
NOISE_SNR_DB_MAX: float = 30.0
FREQ_MASK_ENABLED: bool = True
FREQ_MASK_MAX_BINS: int = 15
TIME_MASK_ENABLED: bool = True
TIME_MASK_MAX_FRAMES: int = 10
