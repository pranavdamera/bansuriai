"""
BansuriAI-V2 — Training Script

Usage:  cd BansuriAI-V2/training && python train.py
"""

import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    TRAIN_CSV, VAL_CSV, CHECKPOINT_DIR, BEST_MODEL_FILENAME, LOG_DIR,
    NUM_CLASSES, SWARA_LABELS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR, SEED,
)
from model import BansuriNoteModel, count_parameters
from dataset_loader import create_train_loader, create_val_loader


def train():
    """Run the full training pipeline."""
    print("=" * 65)
    print("  BansuriAI-V2 — Model Training")
    print("=" * 65)

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────
    print(f"\n  Loading training data from: {TRAIN_CSV}")
    train_loader = create_train_loader(TRAIN_CSV, batch_size=BATCH_SIZE)
    print(f"  Loading validation data from: {VAL_CSV}")
    val_loader = create_val_loader(VAL_CSV, batch_size=BATCH_SIZE)

    if len(train_loader.dataset) == 0:
        print("\n  ERROR: Training set is empty.")
        print("  → Run generate_synthetic_data.py, validate_dataset.py, split_dataset.py first.")
        sys.exit(1)

    print(f"\n  Training samples  : {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Batch size        : {BATCH_SIZE}")

    # ── Initialize model, loss, optimizer, scheduler ──────────────────
    model = BansuriNoteModel(num_classes=NUM_CLASSES).to(device)
    print(f"  Model parameters  : {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, verbose=True,
    )

    # ── Metric logging ────────────────────────────────────────────────
    log_path = LOG_DIR / "training_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_sec"])

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = CHECKPOINT_DIR / BEST_MODEL_FILENAME

    print(f"  Checkpoint path   : {best_model_path}")
    print(f"  Max epochs        : {NUM_EPOCHS}")
    print()
    print("─" * 65)
    print(f"  {'Epoch':>5}  │  {'Train Loss':>10}  {'Train Acc':>9}  │"
          f"  {'Val Loss':>10}  {'Val Acc':>9}  │  {'LR':>10}")
    print("─" * 65)

    # ══════════════════════════════════════════════════════════════════
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        marker = " ★" if val_loss < best_val_loss else ""
        print(
            f"  {epoch:>5}  │  {train_loss:>10.4f}  {train_acc:>8.1%}  │"
            f"  {val_loss:>10.4f}  {val_acc:>8.1%}  │  {lr:>10.2e}{marker}"
        )

        log_writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
                             f"{val_loss:.6f}", f"{val_acc:.4f}", f"{lr:.2e}", f"{elapsed:.1f}"])
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    log_file.close()
    print("─" * 65)
    print(f"\n  Training complete!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Model saved   : {best_model_path}")
    print(f"  Log           : {log_path}\n")


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        logits = model(specs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * specs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += specs.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
            logits = model(specs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * specs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += specs.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


if __name__ == "__main__":
    train()
