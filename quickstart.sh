#!/usr/bin/env bash
#
# BansuriAI-V2 — Quick Start
#
# This script sets up the full project, runs a test training on synthetic
# data, and starts the backend + frontend. Run it once to verify everything
# works before recording your own bansuri data.
#
# Usage:
#   chmod +x quickstart.sh
#   ./quickstart.sh
#

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  BansuriAI-V2 — Quick Start"
echo "═══════════════════════════════════════════════════════════"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# ── Step 1: Check Python ──────────────────────────────────────────────
echo "[1/7] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "  ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python $PYVER found"

# ── Step 2: Check Node ────────────────────────────────────────────────
echo "[2/7] Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "  WARNING: node not found. Frontend won't work."
    echo "  Install Node.js 18+ from https://nodejs.org"
    HAS_NODE=0
else
    NODEVER=$(node -v)
    echo "  Node $NODEVER found"
    HAS_NODE=1
fi

# ── Step 3: Install backend dependencies ──────────────────────────────
echo "[3/7] Installing backend dependencies..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
echo "  Backend dependencies installed"
cd "$PROJECT_ROOT"

# ── Step 4: Generate synthetic training data ──────────────────────────
echo "[4/7] Generating synthetic training data..."
python3 training/generate_synthetic_data.py --takes 15 --output dataset
echo "  105 synthetic .wav files created"

# ── Step 5: Validate and split dataset ────────────────────────────────
echo "[5/7] Validating and splitting dataset..."
python3 dataset/validate_dataset.py --csv dataset/labels.csv --root dataset
python3 dataset/split_dataset.py --csv dataset/labels.csv --output dataset/splits
echo "  Dataset validated and split into train/val/test"

# ── Step 6: Train the model ───────────────────────────────────────────
echo "[6/7] Training model on synthetic data (15 epochs)..."
cd training
python3 -c "
import config
config.NUM_EPOCHS = 15
config.EARLY_STOPPING_PATIENCE = 8
from train import train
train()
"
echo "  Model trained and saved to backend/saved_models/"
cd "$PROJECT_ROOT"

# ── Step 7: Install frontend ─────────────────────────────────────────
if [ "$HAS_NODE" = "1" ]; then
    echo "[7/7] Installing frontend dependencies..."
    cd frontend
    npm install --silent
    echo "  Frontend dependencies installed"
    cd "$PROJECT_ROOT"
else
    echo "[7/7] Skipping frontend (no Node.js)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  To start the app:"
echo ""
echo "    # Terminal 1 — Backend"
echo "    cd backend && source venv/bin/activate && python run.py"
echo ""
if [ "$HAS_NODE" = "1" ]; then
echo "    # Terminal 2 — Frontend"
echo "    cd frontend && npm run dev"
echo ""
echo "    Open http://localhost:5173"
fi
echo ""
echo "  To train on YOUR bansuri recordings:"
echo "    1. Read dataset/RECORDING_GUIDE.md"
echo "    2. Record .wav files into dataset/raw/isolated/{Sa,Re,...}/"
echo "    3. Update dataset/labels.csv"
echo "    4. python dataset/validate_dataset.py"
echo "    5. python dataset/split_dataset.py"
echo "    6. cd training && python train.py"
echo "    7. python evaluate.py"
echo ""
