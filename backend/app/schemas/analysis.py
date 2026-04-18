"""
BansuriAI-V2 — Pydantic Schemas

Defines the data contracts for the /analyze API endpoint.

These models serve two purposes:
    1. Validate and serialize the response so the frontend always gets
       a predictable JSON shape.
    2. Document the API contract — any developer can read this file
       to understand exactly what the endpoint returns.

Matches the "Expected API Response" section of the spec.
"""

from pydantic import BaseModel, Field


class NoteSegment(BaseModel):
    """A single detected note with its time boundaries and confidence.

    Represents one continuous segment where the model predicts the same note.
    Produced by the sequence decoder after smoothing raw frame predictions.
    """
    note: str = Field(
        ...,
        description="Swara label (e.g. 'Sa', 'Re', 'Ga')",
        examples=["Sa"],
    )
    start: float = Field(
        ...,
        ge=0.0,
        description="Segment start time in seconds",
        examples=[0.0],
    )
    end: float = Field(
        ...,
        gt=0.0,
        description="Segment end time in seconds",
        examples=[0.52],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average prediction confidence for this segment (0–1)",
        examples=[0.94],
    )


class AnalysisResponse(BaseModel):
    """Complete analysis result returned by POST /analyze.

    Contains the full note recognition output: the ordered list of unique
    detected notes, the decoded time-segmented sequence, aggregate scores,
    a human-readable summary, and feedback observations.
    """
    detected_notes: list[str] = Field(
        ...,
        description="Ordered list of unique swaras detected in the clip",
        examples=[["Sa", "Re", "Ga", "Ma"]],
    )
    decoded_sequence: list[NoteSegment] = Field(
        ...,
        description="Time-segmented note sequence with confidence scores",
    )
    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted mean confidence across all segments",
        examples=[0.87],
    )
    signal_quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of frames with high-confidence predictions",
        examples=[0.90],
    )
    summary_report: str = Field(
        ...,
        description="Human-readable one-line summary of what was played",
        examples=["Detected an ascending sequence: Sa, Re, Ga, Ma."],
    )
    feedback: list[str] = Field(
        ...,
        description="List of observation strings (strengths, weaknesses)",
        examples=[
            [
                "Highest confidence on Sa and Re.",
                "Slightly lower certainty on later notes.",
            ]
        ],
    )


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str = Field(
        default="healthy",
        description="Service health status",
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready",
    )


class ClassifyRequest(BaseModel):
    """Request body for POST /classify.

    Accepts a base64-encoded chunk of raw Float32 PCM audio at the
    specified sample rate. Designed for real-time streaming clients that
    send small overlapping windows from a live microphone.
    """
    audio_b64: str = Field(
        ...,
        description="Base64-encoded raw Float32 PCM audio bytes",
    )
    sample_rate: int = Field(
        default=22050,
        ge=8000,
        le=48000,
        description="Sample rate of the PCM data in Hz",
    )


class ClassifyResponse(BaseModel):
    """Response from POST /classify.

    Single-note real-time prediction with intonation scoring.
    Matches the API contract documented in the project README.
    """
    predicted_note: str = Field(
        ...,
        description="Swara label with highest model confidence (e.g. 'Pa')",
        examples=["Pa"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence for the predicted note (0–1)",
        examples=[0.91],
    )
    intonation: str = Field(
        ...,
        description="Intonation assessment: 'sharp', 'flat', or 'in_tune'",
        examples=["sharp"],
    )
    cents_off: int = Field(
        ...,
        description="Signed cents deviation from reference pitch (positive = sharp)",
        examples=[12],
    )
    feedback: str = Field(
        ...,
        description="Human-readable embouchure suggestion",
        examples=["Slightly sharp on Pa (+12¢) — relax the embouchure or reduce blow angle slightly."],
    )
