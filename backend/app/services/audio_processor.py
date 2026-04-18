"""
BansuriAI-V2 — Audio Processor

First stage of the pipeline. Takes a raw .wav file path and returns a clean,
standardized numpy waveform ready for feature extraction.

Processing steps:
    1. Load the wav file at the target sample rate (resampling if needed)
    2. Convert to mono if stereo
    3. Validate duration is within acceptable bounds
    4. Trim leading and trailing silence
    5. Peak-normalize amplitude to [-1, 1]

Analogous to the "signal conditioning" stage in an ECG pipeline:
raw electrical signal → filtered, normalized, artifact-free signal.
"""

import numpy as np
import librosa
import soundfile as sf

from app.utils.config import (
    SAMPLE_RATE,
    MAX_DURATION_SEC,
    MIN_DURATION_SEC,
    TRIM_TOP_DB,
    NORMALIZE,
)


class AudioProcessingError(Exception):
    """Raised when audio loading or validation fails."""
    pass


def load_and_preprocess(file_path: str) -> tuple[np.ndarray, int]:
    """Load a wav file and return a clean, normalized waveform.

    Args:
        file_path: Path to the .wav file on disk.

    Returns:
        Tuple of (waveform, sample_rate) where waveform is a 1-D numpy
        float32 array and sample_rate is an integer (always SAMPLE_RATE).

    Raises:
        AudioProcessingError: If the file can't be loaded, is too short,
            too long, or contains no signal after trimming.
    """

    # ── Step 1: Load and resample ──────────────────────────────────────
    try:
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio file: {e}")

    # ── Step 2: Validate duration ──────────────────────────────────────
    duration = len(waveform) / sr

    if duration < MIN_DURATION_SEC:
        raise AudioProcessingError(
            f"Audio too short ({duration:.2f}s). "
            f"Minimum is {MIN_DURATION_SEC}s."
        )

    if duration > MAX_DURATION_SEC:
        raise AudioProcessingError(
            f"Audio too long ({duration:.2f}s). "
            f"Maximum is {MAX_DURATION_SEC}s."
        )

    # ── Step 3: Trim silence ───────────────────────────────────────────
    waveform_trimmed, _ = librosa.effects.trim(
        waveform, top_db=TRIM_TOP_DB
    )

    # Guard against fully-silent files
    if len(waveform_trimmed) < int(sr * MIN_DURATION_SEC):
        raise AudioProcessingError(
            "Audio contains only silence after trimming."
        )

    waveform = waveform_trimmed

    # ── Step 4: Peak normalize ─────────────────────────────────────────
    if NORMALIZE:
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak

    return waveform.astype(np.float32), sr


def get_audio_info(file_path: str) -> dict:
    """Return basic metadata about a wav file without full processing.

    Useful for quick validation before committing to the full pipeline.

    Args:
        file_path: Path to the .wav file on disk.

    Returns:
        Dict with keys: duration_sec, sample_rate, channels, frames.
    """
    try:
        info = sf.info(file_path)
        return {
            "duration_sec": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
        }
    except Exception as e:
        raise AudioProcessingError(f"Cannot read audio info: {e}")
