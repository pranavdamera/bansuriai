"""
BansuriAI-V2 — Note Recognition Model (Architecture Only)

Defines a compact CNN that classifies bansuri notes from log-mel spectrogram
frames. This file contains ONLY the nn.Module definition — no training logic.

Architecture overview:
    Input:  (batch, 1, n_mels, time_frames)   — single-channel spectrogram
    Conv blocks: 3× [Conv2d → BatchNorm → ReLU → MaxPool]
    Adaptive pool to fixed spatial size
    Flatten → Dropout → Linear → per-frame class logits

Why CNN over CRNN for V2 scaffold:
    A plain CNN is simpler to debug and train on small datasets. Once the
    full pipeline is validated, swapping in a CRNN (add a GRU after the
    conv blocks) is a one-file change that doesn't affect any other module.

The model outputs raw logits. Softmax is applied during inference in
model_inference.py, NOT inside forward(), because CrossEntropyLoss in
PyTorch expects raw logits during training.
"""

import torch
import torch.nn as nn

from app.utils.config import NUM_CLASSES, N_MELS


class BansuriNoteModel(nn.Module):
    """Compact CNN for frame-level bansuri note classification.

    Args:
        num_classes: Number of output classes (default from config).
        n_mels: Number of Mel frequency bins in the input spectrogram.
        dropout: Dropout probability before the final classifier.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        n_mels: int = N_MELS,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        # ----- Convolutional feature extractor -----
        # Each block: Conv2d → BatchNorm2d → ReLU → MaxPool2d
        # Channels: 1 → 32 → 64 → 128
        # MaxPool(2,2) halves both freq and time dimensions each block.

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (B, 32, n_mels, T)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # (B, 32, n_mels/2, T/2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (B, 64, n_mels/2, T/2)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # (B, 64, n_mels/4, T/4)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 128, n_mels/4, T/4)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # (B, 128, n_mels/8, T/8)
        )

        # ----- Adaptive pooling -----
        # Collapse the frequency axis to a fixed size (4) while keeping
        # the time axis at 1 — this lets us produce one prediction per
        # "chunk" of time frames passed in.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 1))  # (B, 128, 4, 1)

        # ----- Classifier head -----
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # (B, 128 * 4 * 1) = (B, 512)
            nn.Dropout(p=dropout),
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),          # (B, num_classes)  — raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames).

        Returns:
            Raw logits of shape (batch, num_classes). Apply softmax externally
            for probabilities.
        """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
