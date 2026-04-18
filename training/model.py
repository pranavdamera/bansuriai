"""
BansuriAI-V2 — Note Recognition Model (Training Copy)

IMPORTANT: Same architecture as backend/app/models/note_model.py.
If you change this, update the backend copy to match.

Input:  (batch, 1, 128, 64)  — single-channel log-mel spectrogram
Output: (batch, 7) raw logits — one score per swara class
"""

import torch
import torch.nn as nn

from config import NUM_CLASSES, N_MELS, DROPOUT


class BansuriNoteModel(nn.Module):
    """Compact CNN for single-note bansuri classification."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        n_mels: int = N_MELS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: (B,1,128,T) → (B,32,64,T/2)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 2: → (B,64,32,T/4)
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 3: → (B,128,16,T/8)
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 1))  # → (B,128,4,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # (B, 512)
            nn.Dropout(p=dropout),
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),         # (B, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = BansuriNoteModel()
    print(f"Parameters: {count_parameters(model):,}")
    dummy = torch.randn(4, 1, N_MELS, 64)
    out = model(dummy)
    print(f"Input {dummy.shape} → Output {out.shape}")
    assert out.shape == (4, NUM_CLASSES)
    print("✓ Forward pass OK")
