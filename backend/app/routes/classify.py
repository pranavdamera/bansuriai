"""
BansuriAI-V2 — Classify Route

POST /classify — real-time single-note prediction.

Designed for streaming clients (e.g. the web app's Live Mode) that send
small PCM audio chunks from a live microphone. Returns a single swara
prediction with intonation scoring, suitable for sub-100ms feedback loops.

Request: JSON body with base64-encoded Float32 PCM + sample rate.
Response: predicted_note, confidence, intonation, cents_off, feedback.

Pipeline:
    decode base64 → float32 array
    → trim silence + normalize
    → log-mel spectrogram
    → sliding-window CNN inference
    → pick highest-confidence prediction
    → pYIN intonation scoring
    → ClassifyResponse
"""

import base64
import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas.analysis import ClassifyRequest, ClassifyResponse
from app.services.audio_processor import AudioProcessingError
from app.services.feature_extractor import extract_features
from app.services.model_inference import run_inference
from app.services.intonation_scorer import score_intonation
from app.utils.config import (
    SAMPLE_RATE,
    MIN_DURATION_SEC,
    TRIM_TOP_DB,
    NORMALIZE,
)
from app.utils.note_mapper import index_to_swara

logger = logging.getLogger(__name__)

router = APIRouter()

# Minimum PCM samples required (MIN_DURATION_SEC at SAMPLE_RATE)
MIN_SAMPLES = int(MIN_DURATION_SEC * SAMPLE_RATE)


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Real-time single-note classification",
    description=(
        "Send a base64-encoded Float32 PCM audio chunk. Returns the most likely "
        "swara with confidence score and intonation feedback. "
        "Designed for live microphone streaming at low latency."
    ),
)
async def classify_audio(request: ClassifyRequest) -> ClassifyResponse:
    """Decode PCM, run inference, score intonation, return single-note result."""

    # ── Decode base64 PCM ─────────────────────────────────────────────
    try:
        pcm_bytes = base64.b64decode(request.audio_b64)
        waveform = np.frombuffer(pcm_bytes, dtype=np.float32).copy()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio_b64: could not decode PCM data. {e}",
        )

    if len(waveform) < MIN_SAMPLES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Audio too short ({len(waveform)} samples). "
                f"Send at least {MIN_SAMPLES} samples "
                f"({MIN_DURATION_SEC:.1f}s at {SAMPLE_RATE} Hz)."
            ),
        )

    # ── Resample if needed ────────────────────────────────────────────
    if request.sample_rate != SAMPLE_RATE:
        import librosa
        waveform = librosa.resample(
            waveform, orig_sr=request.sample_rate, target_sr=SAMPLE_RATE
        )

    # ── Trim silence ──────────────────────────────────────────────────
    import librosa
    waveform_trimmed, _ = librosa.effects.trim(waveform, top_db=TRIM_TOP_DB)
    if len(waveform_trimmed) >= MIN_SAMPLES:
        waveform = waveform_trimmed

    # ── Peak normalize ────────────────────────────────────────────────
    if NORMALIZE:
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
    waveform = waveform.astype(np.float32)

    # ── Feature extraction ────────────────────────────────────────────
    try:
        features = extract_features(waveform, SAMPLE_RATE)
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Feature extraction failed.")

    # ── Model inference ───────────────────────────────────────────────
    try:
        predictions = run_inference(features["log_mel"])
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Model inference failed.")

    # ── Pick top prediction ───────────────────────────────────────────
    # frame_labels: array of class indices, one per inference window
    # frame_confidences: array of softmax max values per window
    frame_labels = predictions["frame_labels"]
    frame_confidences = predictions["frame_confidences"]
    frame_probs = predictions["frame_probabilities"]  # shape: (windows, num_classes)

    if len(frame_labels) == 0:
        raise HTTPException(status_code=422, detail="No predictions produced — audio too short.")

    # Aggregate: sum per-class probabilities across all windows, pick winner
    class_scores = np.sum(frame_probs, axis=0)
    top_class = int(np.argmax(class_scores))
    # Confidence = mean of windows where this class was the top prediction
    matching = frame_confidences[frame_labels == top_class]
    confidence = float(np.mean(matching)) if len(matching) > 0 else float(np.max(frame_confidences))
    confidence = round(min(confidence, 1.0), 3)

    predicted_note = index_to_swara(top_class)

    logger.info(f"Classify: {predicted_note} conf={confidence:.2f}")

    # ── Intonation scoring ────────────────────────────────────────────
    try:
        intonation_result = score_intonation(waveform, SAMPLE_RATE, predicted_note)
    except Exception as e:
        logger.warning(f"Intonation scoring failed: {e}")
        intonation_result = {
            "intonation": "in_tune",
            "cents_off": 0,
            "feedback": f"Playing {predicted_note}.",
        }

    return ClassifyResponse(
        predicted_note=predicted_note,
        confidence=confidence,
        intonation=intonation_result["intonation"],
        cents_off=intonation_result["cents_off"],
        feedback=intonation_result["feedback"],
    )
