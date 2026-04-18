"""
BansuriAI-V2 — Sequence Decoder

Fourth stage of the pipeline. Converts raw frame-level model predictions
into a clean, musically meaningful note sequence.

The model produces one prediction per time window. These raw predictions
are noisy — a sustained "Sa" note might flicker to "Re" for a single frame
due to overtones or transition effects. The decoder cleans this up.

Processing steps:
    1. Median filter — smooth out single-frame glitches
    2. Segment grouping — merge consecutive identical labels into segments
    3. Confidence averaging — compute mean confidence per segment
    4. Short segment removal — discard segments below minimum duration
    5. Low confidence removal — discard segments below confidence threshold
    6. Label mapping — convert class indices to swara names

Analogous to the "QRS complex detection + beat classification" stage in an
ECG pipeline: raw classifier outputs → cleaned, discrete event sequence.
"""

import numpy as np
from scipy.ndimage import median_filter

from app.utils.config import (
    HOP_LENGTH,
    SAMPLE_RATE,
    SMOOTHING_WINDOW,
    MIN_SEGMENT_DURATION_SEC,
    CONFIDENCE_THRESHOLD,
    INFERENCE_WINDOW_HOP,
)
from app.utils.note_mapper import index_to_swara


def decode_sequence(
    frame_labels: np.ndarray,
    frame_confidences: np.ndarray,
    window_hop_frames: int = INFERENCE_WINDOW_HOP,
) -> list[dict]:
    """Decode raw frame predictions into a clean note segment list.

    Args:
        frame_labels: 1-D array of predicted class indices, one per window.
        frame_confidences: 1-D array of confidence values (0–1), one per window.
        window_hop_frames: Number of spectrogram frames between window centers.
            Used to convert window indices to time in seconds.

    Returns:
        List of segment dicts, each with keys:
            - "note"       : str — swara label
            - "start"      : float — start time in seconds
            - "end"        : float — end time in seconds
            - "confidence" : float — mean confidence for the segment
    """

    if len(frame_labels) == 0:
        return []

    # ── Step 1: Median filter to smooth single-frame glitches ─────────
    # median_filter requires float input for label arrays
    smoothed_labels = median_filter(
        frame_labels.astype(np.float32),
        size=SMOOTHING_WINDOW,
    ).astype(int)

    # ── Step 2: Group consecutive identical labels into segments ───────
    raw_segments = _group_consecutive(smoothed_labels, frame_confidences)

    # ── Step 3: Convert frame indices to timestamps ───────────────────
    # Each window center corresponds to a time position.
    # Time = window_index × window_hop_frames × hop_length / sample_rate
    frame_to_sec = window_hop_frames * HOP_LENGTH / SAMPLE_RATE

    segments = []
    for seg in raw_segments:
        start_sec = round(seg["start_idx"] * frame_to_sec, 3)
        end_sec = round((seg["end_idx"] + 1) * frame_to_sec, 3)
        duration = end_sec - start_sec
        avg_confidence = round(float(seg["avg_confidence"]), 3)

        # ── Step 4: Filter by minimum duration ────────────────────────
        if duration < MIN_SEGMENT_DURATION_SEC:
            continue

        # ── Step 5: Filter by confidence threshold ────────────────────
        if avg_confidence < CONFIDENCE_THRESHOLD:
            continue

        # ── Step 6: Map class index to swara label ────────────────────
        try:
            note_name = index_to_swara(seg["label"])
        except IndexError:
            continue  # Skip unknown labels silently

        segments.append({
            "note": note_name,
            "start": start_sec,
            "end": end_sec,
            "confidence": avg_confidence,
        })

    return segments


def _group_consecutive(
    labels: np.ndarray,
    confidences: np.ndarray,
) -> list[dict]:
    """Group consecutive identical labels into segments.

    Args:
        labels: 1-D int array of class labels.
        confidences: 1-D float array of per-frame confidences.

    Returns:
        List of dicts with keys: label, start_idx, end_idx, avg_confidence.
    """
    segments = []
    current_label = labels[0]
    start_idx = 0
    conf_accumulator = [confidences[0]]

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            # Close the current segment
            segments.append({
                "label": int(current_label),
                "start_idx": start_idx,
                "end_idx": i - 1,
                "avg_confidence": float(np.mean(conf_accumulator)),
            })
            # Start a new segment
            current_label = labels[i]
            start_idx = i
            conf_accumulator = [confidences[i]]
        else:
            conf_accumulator.append(confidences[i])

    # Close the final segment
    segments.append({
        "label": int(current_label),
        "start_idx": start_idx,
        "end_idx": len(labels) - 1,
        "avg_confidence": float(np.mean(conf_accumulator)),
    })

    return segments
