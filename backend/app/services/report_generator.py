"""
BansuriAI-V2 — Report Generator

Final backend stage. Takes decoded note segments and assembles the complete
API response matching the AnalysisResponse schema.

Responsibilities:
    - Extract the unique ordered list of detected notes
    - Compute overall confidence (weighted by segment duration)
    - Estimate signal quality from frame-level confidence distribution
    - Generate a human-readable summary sentence
    - Produce feedback observations (strongest notes, weakest notes, patterns)
"""

import numpy as np

from app.schemas.analysis import AnalysisResponse, NoteSegment


def generate_report(
    segments: list[dict],
    frame_confidences: np.ndarray,
) -> AnalysisResponse:
    """Build the complete analysis response from decoded segments.

    Args:
        segments: List of segment dicts from the sequence decoder.
            Each has keys: note, start, end, confidence.
        frame_confidences: Raw per-window confidence array from inference.
            Used to compute signal quality score.

    Returns:
        AnalysisResponse Pydantic model ready for JSON serialization.
    """

    # ── Handle empty results ──────────────────────────────────────────
    if not segments:
        return AnalysisResponse(
            detected_notes=[],
            decoded_sequence=[],
            overall_confidence=0.0,
            signal_quality_score=0.0,
            summary_report="No notes detected in the audio clip.",
            feedback=["The audio may be too quiet, too short, or not contain bansuri playing."],
        )

    # ── Build decoded sequence (Pydantic models) ──────────────────────
    decoded_sequence = [
        NoteSegment(
            note=seg["note"],
            start=seg["start"],
            end=seg["end"],
            confidence=seg["confidence"],
        )
        for seg in segments
    ]

    # ── Unique detected notes (preserving first-occurrence order) ─────
    seen = set()
    detected_notes = []
    for seg in segments:
        if seg["note"] not in seen:
            seen.add(seg["note"])
            detected_notes.append(seg["note"])

    # ── Overall confidence (duration-weighted mean) ───────────────────
    durations = [seg["end"] - seg["start"] for seg in segments]
    confidences = [seg["confidence"] for seg in segments]
    total_duration = sum(durations)

    if total_duration > 0:
        overall_confidence = round(
            sum(d * c for d, c in zip(durations, confidences)) / total_duration,
            3,
        )
    else:
        overall_confidence = 0.0

    # ── Signal quality score ──────────────────────────────────────────
    # Fraction of inference windows where the model was reasonably confident
    high_conf_threshold = 0.6
    if len(frame_confidences) > 0:
        signal_quality_score = round(
            float(np.mean(frame_confidences >= high_conf_threshold)),
            3,
        )
    else:
        signal_quality_score = 0.0

    # ── Summary report ────────────────────────────────────────────────
    summary_report = _build_summary(detected_notes, segments)

    # ── Feedback observations ─────────────────────────────────────────
    feedback = _build_feedback(segments)

    return AnalysisResponse(
        detected_notes=detected_notes,
        decoded_sequence=decoded_sequence,
        overall_confidence=overall_confidence,
        signal_quality_score=signal_quality_score,
        summary_report=summary_report,
        feedback=feedback,
    )


def _build_summary(detected_notes: list[str], segments: list[dict]) -> str:
    """Generate a one-line human-readable summary.

    Examples:
        "Detected an ascending sequence: Sa, Re, Ga, Ma."
        "Detected 3 notes over 4.2 seconds: Sa, Pa, Sa."
    """
    note_str = ", ".join(detected_notes)
    n_segments = len(segments)

    if n_segments == 0:
        return "No notes detected."

    total_duration = segments[-1]["end"] - segments[0]["start"]

    # Check if the sequence is ascending/descending in the standard swara order
    pattern = _detect_pattern(detected_notes)

    if pattern:
        return f"Detected {pattern}: {note_str}."
    else:
        return (
            f"Detected {len(detected_notes)} unique note(s) across "
            f"{n_segments} segment(s) over {total_duration:.1f} seconds: {note_str}."
        )


def _detect_pattern(notes: list[str]) -> str | None:
    """Check if the note list follows a recognizable ascending/descending pattern."""
    swara_order = ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"]
    swara_rank = {name: i for i, name in enumerate(swara_order)}

    if len(notes) < 2:
        return None

    ranks = [swara_rank.get(n) for n in notes]
    if any(r is None for r in ranks):
        return None

    # Check strictly ascending
    if all(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1)):
        return "an ascending sequence (aaroha)"

    # Check strictly descending
    if all(ranks[i] > ranks[i + 1] for i in range(len(ranks) - 1)):
        return "a descending sequence (avroha)"

    return None


def _build_feedback(segments: list[dict]) -> list[str]:
    """Generate a list of observation strings about the detected notes."""
    feedback = []

    if not segments:
        return feedback

    confidences = [(seg["note"], seg["confidence"]) for seg in segments]

    # Identify highest and lowest confidence segments
    best = max(confidences, key=lambda x: x[1])
    worst = min(confidences, key=lambda x: x[1])

    feedback.append(
        f"Highest confidence on {best[0]} ({best[1]:.0%})."
    )

    if best[0] != worst[0]:
        feedback.append(
            f"Lowest confidence on {worst[0]} ({worst[1]:.0%})."
        )

    # Overall assessment
    avg_conf = sum(c for _, c in confidences) / len(confidences)
    if avg_conf >= 0.85:
        feedback.append("Overall prediction confidence is strong.")
    elif avg_conf >= 0.65:
        feedback.append("Moderate overall confidence — some notes may be ambiguous.")
    else:
        feedback.append("Low overall confidence — audio quality or note clarity may be limited.")

    return feedback
