"""
BansuriAI-V2 — Dataset Loader

PyTorch Dataset that reads a split CSV, loads .wav files, extracts
log-mel spectrograms, and returns (spectrogram_tensor, label_index) pairs.

Data flow per sample:
    CSV row → librosa.load → [augment waveform] → trim silence → normalize →
    mel spectrogram → log scale → pad/truncate to 64 frames →
    [augment spectrogram] → normalize to [0,1] → tensor (1, 128, 64)
"""

import csv
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    SAMPLE_RATE, TRIM_TOP_DB, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX,
    REF_DB, FIXED_NUM_FRAMES, SWARA_LABELS, DATASET_DIR, BATCH_SIZE,
    NUM_WORKERS, AUGMENT_ENABLED, NOISE_ENABLED, FREQ_MASK_ENABLED,
    TIME_MASK_ENABLED,
)
from augmentations import augment_waveform, augment_spectrogram

SWARA_TO_INDEX = {label: idx for idx, label in enumerate(SWARA_LABELS)}


class BansuriDataset(Dataset):
    """PyTorch Dataset for isolated bansuri note recordings.

    Args:
        csv_path: Path to a split CSV (train.csv, val.csv, or test.csv).
        dataset_root: Root directory of the dataset (parent of raw/).
        augment: Whether to apply data augmentation (True for train only).
    """

    def __init__(self, csv_path, dataset_root=DATASET_DIR, augment=False):
        self.dataset_root = Path(dataset_root)
        self.augment = augment
        self.samples = []

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("category", "").strip() != "isolated":
                    continue
                label = row.get("label", "").strip()
                if label not in SWARA_TO_INDEX:
                    continue
                full_path = self.dataset_root / row["file_path"].strip()
                if not full_path.exists():
                    print(f"  WARNING: Skipping missing file: {full_path}")
                    continue
                self.samples.append({
                    "file_path": str(full_path),
                    "label": label,
                    "label_index": SWARA_TO_INDEX[label],
                })
        print(f"  Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sr = librosa.load(sample["file_path"], sr=SAMPLE_RATE, mono=True)

        # Waveform augmentation (training only)
        if self.augment and AUGMENT_ENABLED:
            waveform = augment_waveform(waveform, noise_enabled=NOISE_ENABLED)

        # Trim silence + normalize
        waveform, _ = librosa.effects.trim(waveform, top_db=TRIM_TOP_DB)
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak

        # Log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=REF_DB)

        # Pad or truncate to fixed length
        log_mel = _fix_length(log_mel, FIXED_NUM_FRAMES)

        # Spectrogram augmentation (training only)
        if self.augment and AUGMENT_ENABLED:
            log_mel = augment_spectrogram(
                log_mel, freq_mask_enabled=FREQ_MASK_ENABLED,
                time_mask_enabled=TIME_MASK_ENABLED,
            )

        # Normalize to [0, 1]
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

        # To tensor: (1, n_mels, T)
        tensor = torch.from_numpy(log_mel).float().unsqueeze(0)
        return tensor, sample["label_index"]


def _fix_length(spectrogram, target_frames):
    """Pad (edge) or truncate (center-crop) to fixed time length."""
    _, current = spectrogram.shape
    if current == target_frames:
        return spectrogram
    elif current > target_frames:
        start = (current - target_frames) // 2
        return spectrogram[:, start : start + target_frames]
    else:
        pad_left = (target_frames - current) // 2
        pad_right = target_frames - current - pad_left
        return np.pad(spectrogram, ((0, 0), (pad_left, pad_right)), mode="edge")


def create_train_loader(csv_path, batch_size=BATCH_SIZE):
    return DataLoader(BansuriDataset(csv_path, augment=True),
                      batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)


def create_val_loader(csv_path, batch_size=BATCH_SIZE):
    return DataLoader(BansuriDataset(csv_path, augment=False),
                      batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)


def create_test_loader(csv_path, batch_size=BATCH_SIZE):
    return DataLoader(BansuriDataset(csv_path, augment=False),
                      batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
