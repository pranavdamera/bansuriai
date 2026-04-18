"""
BansuriAI-V2 — Analyze Route

Defines the POST /analyze endpoint that orchestrates the full note
recognition pipeline:

    upload → save temp file → audio_processor → feature_extractor →
    model_inference → sequence_decoder → report_generator → JSON response

This is the single entry point for the frontend. It accepts a .wav file
via multipart form upload and returns a structured AnalysisResponse.

Error handling strategy:
    - AudioProcessingError → 422 (bad input, user can fix it)
    - File type validation → 400 (wrong file format)
    - Any other exception  → 500 (server-side bug)
    - Temp file is ALWAYS cleaned up, even on failure
"""

import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.analysis import AnalysisResponse
from app.services.audio_processor import AudioProcessingError, load_and_preprocess
from app.services.feature_extractor import extract_features
from app.services.model_inference import run_inference
from app.services.sequence_decoder import decode_sequence
from app.services.report_generator import generate_report
from app.utils.config import TEMP_UPLOAD_DIR, MAX_UPLOAD_BYTES

logger = logging.getLogger(__name__)

router = APIRouter()

# Allowed MIME types for uploaded audio
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
}

# Also check file extension as a fallback (some browsers send wrong MIME)
ALLOWED_EXTENSIONS = {".wav"}


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze a bansuri audio recording",
    description=(
        "Upload a .wav file of bansuri playing. Returns detected notes, "
        "timestamps, confidence scores, and a summary report."
    ),
)
async def analyze_audio(
    file: UploadFile = File(..., description="A .wav audio file to analyze"),
) -> AnalysisResponse:
    """Full analysis pipeline: upload → preprocess → features → model → decode → report."""

    # ── Validate file type ────────────────────────────────────────────
    _validate_upload(file)

    # ── Save to temp file ─────────────────────────────────────────────
    # We need a file path on disk because librosa.load() reads from path.
    temp_path = None
    try:
        temp_path = await _save_temp_file(file)

        # ── Stage 1: Audio preprocessing ──────────────────────────────
        logger.info(f"Processing uploaded file: {file.filename}")
        waveform, sr = load_and_preprocess(temp_path)
        logger.info(f"Audio loaded: {len(waveform)} samples, {sr} Hz")

        # ── Stage 2: Feature extraction ───────────────────────────────
        features = extract_features(waveform, sr)
        log_mel = features["log_mel"]
        logger.info(f"Features extracted: spectrogram shape {log_mel.shape}")

        # ── Stage 3: Model inference ──────────────────────────────────
        predictions = run_inference(log_mel)
        logger.info(
            f"Inference complete: {len(predictions['frame_labels'])} windows"
        )

        # ── Stage 4: Sequence decoding ────────────────────────────────
        segments = decode_sequence(
            frame_labels=predictions["frame_labels"],
            frame_confidences=predictions["frame_confidences"],
            window_hop_frames=predictions["window_hop"],
        )
        logger.info(f"Decoded {len(segments)} note segments")

        # ── Stage 5: Report generation ────────────────────────────────
        report = generate_report(
            segments=segments,
            frame_confidences=predictions["frame_confidences"],
        )
        logger.info(f"Report generated: {report.summary_report}")

        return report

    except AudioProcessingError as e:
        logger.warning(f"Audio processing failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during analysis. Please try again.",
        )

    finally:
        # Always clean up the temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.debug(f"Cleaned up temp file: {temp_path}")


def _validate_upload(file: UploadFile) -> None:
    """Validate that the uploaded file is a .wav audio file.

    Checks both MIME type and file extension. Raises HTTPException
    with a 400 status if validation fails.
    """
    # Check MIME type
    content_type = (file.content_type or "").lower()
    ext = os.path.splitext(file.filename or "")[1].lower()

    if content_type not in ALLOWED_CONTENT_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type '{content_type}' ({ext}). "
                "Please upload a .wav audio file."
            ),
        )


async def _save_temp_file(file: UploadFile) -> str:
    """Read the upload into a temporary file and return its path.

    The caller is responsible for deleting this file when done.
    Raises HTTPException 413 if the file exceeds MAX_UPLOAD_BYTES.
    """
    suffix = os.path.splitext(file.filename or ".wav")[1]
    fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=str(TEMP_UPLOAD_DIR))

    try:
        contents = await file.read()

        # ── File size guard ───────────────────────────────────────────
        if len(contents) > MAX_UPLOAD_BYTES:
            os.close(fd)
            os.remove(temp_path)
            max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
            actual_mb = len(contents) / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({actual_mb:.1f} MB). "
                    f"Maximum allowed is {max_mb:.0f} MB."
                ),
            )

        with os.fdopen(fd, "wb") as f:
            f.write(contents)
    except HTTPException:
        raise  # Re-raise our own 413
    except Exception:
        # If writing fails, close the fd and clean up
        os.close(fd)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return temp_path
