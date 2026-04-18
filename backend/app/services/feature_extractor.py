"""
BansuriAI-V2 — Feature Extractor

Second stage of the pipeline. Takes a preprocessed waveform and produces
model-ready features: a log-mel spectrogram matrix.

Primary output:
    Log-mel spectrogram — shape (n_mels, time_frames)
    This is a 2-D "image" of the audio where:
        - Y axis = Mel-scaled frequency bins (128 bins, 50–4000 Hz)
        - X axis = time frames (one frame every hop_length samples)
        - Values = log-scaled power (dB)

Optional output:
    Pitch contour — useful for debugging and visualization, not fed to model.

Why log-mel over raw STFT or MFCCs:
    - Mel scale matches human (and musical) pitch perception
    - Log compression handles the huge dynamic range of audio
    - 2-D format is natural input for CNNs
    - More information-rich than MFCCs (which discard fine spectral detail)
    - Bansuri has strong harmonics that are well-captured in Mel bands
"""

import numpy as np
import librosa

from app.utils.config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    REF_DB,
    FRAME_DURATION_SEC,
)


def extract_features(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> dict:
    """Extract log-mel spectrogram and frame timestamps from a waveform.

    Args:
        waveform: 1-D numpy float32 array (preprocessed, normalized).
        sr: Sample rate of the waveform.

    Returns:
        Dictionary with keys:
            - "log_mel"      : np.ndarray shape (n_mels, T) — the spectrogram
            - "frame_times"  : np.ndarray shape (T,) — center time of each frame (sec)
            - "num_frames"   : int — total number of time frames
            - "duration_sec" : float — total audio duration
    """

    # ── Compute Mel spectrogram (power) ────────────────────────────────
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,   # Power spectrogram (squared magnitude)
    )

    # ── Convert to log scale (dB) ─────────────────────────────────────
    log_mel = librosa.power_to_db(
        mel_spec,
        ref=np.max,       # Normalize relative to the peak
        top_db=REF_DB,    # Clip floor at -80 dB below peak
    )

    # ── Compute frame center timestamps ────────────────────────────────
    num_frames = log_mel.shape[1]
    frame_times = librosa.frames_to_time(
        frames=np.arange(num_frames),
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    return {
        "log_mel": log_mel.astype(np.float32),
        "frame_times": frame_times.astype(np.float32),
        "num_frames": num_frames,
        "duration_sec": len(waveform) / sr,
    }


def extract_pitch_contour(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> dict:
    """Extract a pitch contour using pYIN (optional, for debugging).

    This is NOT used by the model. It provides a reference pitch track
    that can be displayed in the frontend or used to sanity-check model
    predictions during development.

    Args:
        waveform: 1-D numpy float32 array.
        sr: Sample rate.

    Returns:
        Dictionary with keys:
            - "f0"          : np.ndarray — fundamental frequency per frame (Hz), NaN where unvoiced
            - "voiced_flag" : np.ndarray — boolean mask, True where pitch is detected
            - "frame_times" : np.ndarray — timestamp for each frame (sec)
    """
    f0, voiced_flag, _ = librosa.pyin(
        waveform,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    num_frames = len(f0)
    frame_times = librosa.frames_to_time(
        frames=np.arange(num_frames),
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    return {
        "f0": f0.astype(np.float32),
        "voiced_flag": voiced_flag,
        "frame_times": frame_times.astype(np.float32),
    }
