"""
BansuriAI-V2 — Model Inference

Third stage of the pipeline. Loads the trained PyTorch model and runs
note predictions on extracted log-mel spectrograms.

CRITICAL DESIGN PRINCIPLE — PREPROCESSING PARITY:
    The model was trained on spectrograms preprocessed a specific way
    (see training/dataset_loader.py). The inference preprocessing here
    MUST reproduce those exact steps, minus augmentation. Any mismatch
    in normalization, window size, or padding causes the model to see
    data that looks nothing like its training distribution, leading to
    random or degraded predictions even with a perfectly trained model.

    Training pipeline (no augmentation path):
        1. librosa.load(sr=22050, mono=True)
        2. librosa.effects.trim(top_db=20)
        3. Peak normalize waveform to [-1, 1]
        4. melspectrogram(n_fft=2048, hop=512, n_mels=128, ...)
        5. power_to_db(ref=np.max, top_db=80)
        6. Pad or center-crop to FIXED_NUM_FRAMES (64) frames
        7. Per-window [0, 1] normalization: (x - min) / (max - min + 1e-8)
        8. Reshape to (1, 128, 64)

    Steps 1–3 are handled by audio_processor.py (upstream of this module).
    Steps 4–5 are handled by feature_extractor.py (upstream of this module).
    Steps 6–8 are handled HERE inside run_inference().

Inference strategy:
    Short clips (≤ 64 frames): pad to 64 frames → 1 prediction.
    Long clips (> 64 frames): slide a 64-frame window with 50% overlap,
    producing one prediction per window position.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.models.note_model import BansuriNoteModel
from app.utils.config import (
    MODEL_PATH,
    NUM_CLASSES,
    FIXED_NUM_FRAMES,
    INFERENCE_WINDOW_HOP,
)

logger = logging.getLogger(__name__)

# Module-level model reference — set by load_model(), used by run_inference()
_model: BansuriNoteModel | None = None
_model_loaded: bool = False


# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_model(model_path: Path = MODEL_PATH) -> bool:
    """Load the trained model weights from disk.

    Called once at application startup via the lifespan handler in main.py.
    If the checkpoint doesn't exist, initializes with random weights so
    the pipeline can run end-to-end during development.

    Args:
        model_path: Path to the .pt checkpoint file.

    Returns:
        True if real trained weights were loaded successfully.
        False if using random placeholder weights.
    """
    global _model, _model_loaded

    _model = BansuriNoteModel(num_classes=NUM_CLASSES)
    _model.eval()  # Disable dropout and set batchnorm to eval mode

    if model_path.exists():
        try:
            state_dict = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True,
            )
            _model.load_state_dict(state_dict)
            _model_loaded = True

            param_count = sum(p.numel() for p in _model.parameters())
            logger.info(f"Model loaded from {model_path}")
            logger.info(
                f"  Parameters: {param_count:,}  |  "
                f"Input: (B, 1, 128, {FIXED_NUM_FRAMES})  |  "
                f"Window hop: {INFERENCE_WINDOW_HOP}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Using random weights.")
            _model_loaded = False
            return False
    else:
        logger.warning(
            f"No checkpoint found at {model_path}. "
            "Running with random weights (placeholder mode)."
        )
        _model_loaded = False
        return False


def is_model_loaded() -> bool:
    """Check whether a real trained model is loaded (vs placeholder)."""
    return _model_loaded


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def run_inference(log_mel: np.ndarray) -> dict:
    """Run note predictions on a log-mel spectrogram.

    Slides a fixed-size window across the time axis of the spectrogram.
    Each window is padded/cropped to FIXED_NUM_FRAMES, normalized to
    [0, 1], and fed through the CNN to produce one class prediction.

    Args:
        log_mel: Spectrogram array of shape (n_mels, total_frames).
                 Values are in dB (output of feature_extractor.py).

    Returns:
        Dictionary with keys:
            - "frame_labels"       : np.ndarray (W,) — predicted class per window
            - "frame_confidences"  : np.ndarray (W,) — max softmax prob per window
            - "frame_probabilities": np.ndarray (W, num_classes) — full softmax
            - "window_hop"         : int — hop used between windows
            - "window_centers"     : np.ndarray (W,) — center frame of each window
    """
    if _model is None:
        raise RuntimeError("Model not initialized. Call load_model() first.")

    n_mels, total_frames = log_mel.shape

    logger.debug(
        f"Inference: spectrogram ({n_mels}, {total_frames}), "
        f"window={FIXED_NUM_FRAMES}, hop={INFERENCE_WINDOW_HOP}"
    )

    # ── Step 6: Extract and pad/crop windows ──────────────────────────
    windows, window_centers = _extract_windows(
        log_mel, FIXED_NUM_FRAMES, INFERENCE_WINDOW_HOP
    )

    # ── Step 7: Normalize each window to [0, 1] ──────────────────────
    # Matches training/dataset_loader.py exactly:
    #     (log_mel - min) / (max - min + 1e-8)
    normalized = _normalize_windows(windows)

    # ── Step 8: Stack into batch tensor (W, 1, n_mels, 64) ───────────
    batch = np.stack(normalized, axis=0)[:, np.newaxis, :, :]
    tensor = torch.from_numpy(batch).float()

    logger.debug(f"Batch tensor: {tensor.shape}, {len(windows)} windows")

    # ── Forward pass ──────────────────────────────────────────────────
    with torch.no_grad():
        logits = _model(tensor)                            # (W, num_classes)
        probabilities = F.softmax(logits, dim=-1)
        confidences, labels = torch.max(probabilities, dim=-1)

    logger.debug(
        f"Predictions: {len(labels)} windows, "
        f"mean confidence {confidences.mean():.3f}"
    )

    return {
        "frame_labels": labels.numpy(),
        "frame_confidences": confidences.numpy(),
        "frame_probabilities": probabilities.numpy(),
        "window_hop": INFERENCE_WINDOW_HOP,
        "window_centers": np.array(window_centers),
    }


# ═══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _extract_windows(
    spectrogram: np.ndarray,
    window_size: int,
    hop: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Slice the spectrogram into overlapping fixed-size windows.

    Short spectrograms (< window_size) are edge-padded to window_size,
    producing a single window — matching training/dataset_loader.py
    _fix_length() behavior exactly.

    Long spectrograms get a sliding window with 50% overlap. A final
    right-aligned window is added if the last regular window doesn't
    reach the end, so no audio at the tail is missed.

    Args:
        spectrogram: (n_mels, total_frames) array.
        window_size: Frames per window (== FIXED_NUM_FRAMES == 64).
        hop: Frames between consecutive window starts.

    Returns:
        (windows, centers) where:
            windows: list of (n_mels, window_size) arrays
            centers: list of center frame indices (for timestamp math)
    """
    n_mels, total_frames = spectrogram.shape
    windows = []
    centers = []

    if total_frames <= window_size:
        # ── Short clip: pad to window_size, one prediction ────────────
        padded = _pad_or_crop(spectrogram, window_size)
        windows.append(padded)
        centers.append(total_frames // 2)

    else:
        # ── Sliding window ────────────────────────────────────────────
        for start in range(0, total_frames - window_size + 1, hop):
            chunk = spectrogram[:, start : start + window_size]
            windows.append(chunk)
            centers.append(start + window_size // 2)

        # Right-aligned tail window if we missed the end
        last_covered = (len(windows) - 1) * hop + window_size
        if last_covered < total_frames:
            tail = spectrogram[:, total_frames - window_size : total_frames]
            windows.append(tail)
            centers.append(total_frames - window_size // 2)

    return windows, centers


def _pad_or_crop(spectrogram: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad or center-crop to exactly target_frames.

    MUST match training/dataset_loader.py _fix_length():
        - Longer than target: center-crop
        - Shorter than target: edge-pad (split evenly left/right)

    Args:
        spectrogram: (n_mels, current_frames) array.
        target_frames: Desired number of time frames.

    Returns:
        (n_mels, target_frames) array.
    """
    _, current_frames = spectrogram.shape

    if current_frames == target_frames:
        return spectrogram

    elif current_frames > target_frames:
        # Center-crop
        start = (current_frames - target_frames) // 2
        return spectrogram[:, start : start + target_frames]

    else:
        # Edge-pad (replicate boundary values)
        pad_left = (target_frames - current_frames) // 2
        pad_right = target_frames - current_frames - pad_left
        return np.pad(
            spectrogram,
            ((0, 0), (pad_left, pad_right)),
            mode="edge",
        )


def _normalize_windows(windows: list[np.ndarray]) -> list[np.ndarray]:
    """Normalize each window independently to [0, 1].

    MUST match training/dataset_loader.py:
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

    Each window uses its own min/max — not global stats. This makes the
    model invariant to absolute volume differences between recordings.

    Args:
        windows: List of (n_mels, window_size) arrays (dB values).

    Returns:
        List of (n_mels, window_size) arrays in [0, 1] range.
    """
    normalized = []
    for w in windows:
        w_min = w.min()
        w_max = w.max()
        w_norm = (w - w_min) / (w_max - w_min + 1e-8)
        normalized.append(w_norm.astype(np.float32))
    return normalized
