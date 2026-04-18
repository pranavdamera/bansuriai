"""
BansuriAI-V2 — Data Augmentations

Waveform augmentations (before spectrogram): time_shift, add_noise
Spectrogram augmentations (after spectrogram): freq_mask, time_mask

Each augmentation is applied with 50% probability during training only.
"""

import numpy as np
from config import (
    TIME_SHIFT_MAX_SAMPLES, NOISE_SNR_DB_MIN, NOISE_SNR_DB_MAX,
    FREQ_MASK_MAX_BINS, TIME_MASK_MAX_FRAMES,
)


# ═══════════════════════════════════════════════════════════════════════
# WAVEFORM AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════════════

def time_shift(waveform: np.ndarray, max_shift: int = TIME_SHIFT_MAX_SAMPLES) -> np.ndarray:
    """Randomly circular-shift the waveform forward or backward in time."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(waveform, shift)


def add_noise(
    waveform: np.ndarray,
    snr_db_min: float = NOISE_SNR_DB_MIN,
    snr_db_max: float = NOISE_SNR_DB_MAX,
) -> np.ndarray:
    """Add Gaussian noise at a random signal-to-noise ratio."""
    snr_db = np.random.uniform(snr_db_min, snr_db_max)
    signal_power = np.mean(waveform ** 2)
    if signal_power == 0:
        return waveform
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return waveform + noise


# ═══════════════════════════════════════════════════════════════════════
# SPECTROGRAM AUGMENTATIONS (SpecAugment-style)
# ═══════════════════════════════════════════════════════════════════════

def freq_mask(spectrogram: np.ndarray, max_bins: int = FREQ_MASK_MAX_BINS) -> np.ndarray:
    """Zero out a random contiguous band of frequency bins."""
    spec = spectrogram.copy()
    n_mels, _ = spec.shape
    mask_width = np.random.randint(1, max_bins + 1)
    start = np.random.randint(0, n_mels - mask_width + 1)
    spec[start : start + mask_width, :] = spec.min()
    return spec


def time_mask(spectrogram: np.ndarray, max_frames: int = TIME_MASK_MAX_FRAMES) -> np.ndarray:
    """Zero out a random contiguous span of time frames."""
    spec = spectrogram.copy()
    _, n_frames = spec.shape
    mask_width = np.random.randint(1, min(max_frames + 1, n_frames))
    start = np.random.randint(0, n_frames - mask_width + 1)
    spec[:, start : start + mask_width] = spec.min()
    return spec


# ═══════════════════════════════════════════════════════════════════════
# COMBINED PIPELINES (each augmentation applied with 50% probability)
# ═══════════════════════════════════════════════════════════════════════

def augment_waveform(waveform: np.ndarray, noise_enabled: bool = True) -> np.ndarray:
    if np.random.random() < 0.5:
        waveform = time_shift(waveform)
    if noise_enabled and np.random.random() < 0.5:
        waveform = add_noise(waveform)
    return waveform


def augment_spectrogram(
    spectrogram: np.ndarray,
    freq_mask_enabled: bool = True,
    time_mask_enabled: bool = True,
) -> np.ndarray:
    if freq_mask_enabled and np.random.random() < 0.5:
        spectrogram = freq_mask(spectrogram)
    if time_mask_enabled and np.random.random() < 0.5:
        spectrogram = time_mask(spectrogram)
    return spectrogram
