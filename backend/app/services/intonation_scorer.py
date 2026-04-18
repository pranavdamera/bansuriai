"""
BansuriAI-V2 — Intonation Scorer

Given a waveform and the model's predicted swara, detects the actual
fundamental frequency using pYIN and computes how far the note is from
ideal intonation in cents.

Output:
    intonation  — "sharp", "flat", or "in_tune"
    cents_off   — deviation from the reference pitch (positive = sharp)
    feedback    — human-readable embouchure suggestion
"""

import numpy as np
import librosa

from app.utils.config import SAMPLE_RATE, HOP_LENGTH, FMIN, FMAX

# Reference frequencies for an E-key bansuri (Sa ≈ E4 = 330 Hz).
# These must match the frequencies used when training — see
# training/generate_synthetic_data.py for the authoritative values.
# Update this table when retraining on a different key.
SWARA_REFERENCE_HZ: dict[str, float] = {
    "Sa":  329.63,
    "Re":  369.99,
    "Ga":  415.30,
    "Ma":  440.00,
    "Pa":  493.88,
    "Dha": 554.37,
    "Ni":  622.25,
}

# Thresholds in cents. Below IN_TUNE_CENTS is considered in tune.
IN_TUNE_CENTS = 10

# pYIN confidence minimum for a frame to be considered voiced
PYIN_CONFIDENCE_MIN = 0.5


def score_intonation(
    waveform: np.ndarray,
    sr: int,
    predicted_note: str,
) -> dict:
    """Compute intonation deviation for the predicted note.

    Args:
        waveform: Preprocessed float32 waveform at `sr` Hz.
        sr: Sample rate.
        predicted_note: Swara name predicted by the classifier (e.g. "Pa").

    Returns:
        Dict with keys:
            intonation  — "sharp", "flat", or "in_tune"
            cents_off   — int, signed cents deviation from reference
            feedback    — str, embouchure suggestion
    """
    reference_hz = SWARA_REFERENCE_HZ.get(predicted_note)

    # If we don't have a reference (unknown note), return neutral
    if reference_hz is None:
        return {
            "intonation": "in_tune",
            "cents_off": 0,
            "feedback": f"Playing {predicted_note}.",
        }

    # Run pYIN to get per-frame fundamental frequency estimates
    f0, voiced_flag, voiced_probs = librosa.pyin(
        waveform,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    # Filter to high-confidence voiced frames
    confident_voiced = voiced_probs > PYIN_CONFIDENCE_MIN
    voiced_f0 = f0[confident_voiced & voiced_flag]

    if len(voiced_f0) == 0:
        return {
            "intonation": "in_tune",
            "cents_off": 0,
            "feedback": "Could not detect clear pitch — try playing more sustained notes.",
        }

    # Use median to be robust against octave errors and transients
    detected_hz = float(np.median(voiced_f0))

    # Normalize across octaves: shift detected pitch to the same octave
    # as the reference so we measure intonation not octave placement.
    ratio = detected_hz / reference_hz
    # Bring ratio into [0.5, 2.0) — within one octave of reference
    while ratio < 0.5:
        ratio *= 2.0
    while ratio >= 2.0:
        ratio /= 2.0

    # Cents deviation from reference
    cents = round(1200.0 * np.log2(ratio))

    # Classify
    if cents > IN_TUNE_CENTS:
        intonation = "sharp"
    elif cents < -IN_TUNE_CENTS:
        intonation = "flat"
    else:
        intonation = "in_tune"

    feedback = _build_feedback(predicted_note, intonation, cents)

    return {
        "intonation": intonation,
        "cents_off": cents,
        "feedback": feedback,
    }


def _build_feedback(note: str, intonation: str, cents: int) -> str:
    """Generate an embouchure suggestion based on intonation direction and magnitude."""
    abs_cents = abs(cents)

    if intonation == "in_tune":
        return f"Good intonation on {note} — holding center well."

    severity = "Slightly" if abs_cents < 25 else "Noticeably" if abs_cents < 40 else "Very"

    if intonation == "sharp":
        return (
            f"{severity} sharp on {note} (+{abs_cents}¢) — "
            "relax the embouchure or reduce blow angle slightly."
        )
    else:
        return (
            f"{severity} flat on {note} (−{abs_cents}¢) — "
            "firm the embouchure or increase air speed."
        )
