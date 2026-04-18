"""
BansuriAI-V2 — Model Evaluation

Usage:  cd BansuriAI-V2/training && python evaluate.py
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from config import (
    TEST_CSV, CHECKPOINT_DIR, BEST_MODEL_FILENAME,
    NUM_CLASSES, SWARA_LABELS, BATCH_SIZE,
)
from model import BansuriNoteModel
from dataset_loader import create_test_loader


def evaluate(model_path, csv_path, batch_size=BATCH_SIZE):
    model_path, csv_path = Path(model_path), Path(csv_path)

    print("=" * 65)
    print("  BansuriAI-V2 — Model Evaluation")
    print("=" * 65)
    print(f"  Model : {model_path}")
    print(f"  Data  : {csv_path}")

    if not model_path.exists():
        print(f"\n  ERROR: No checkpoint at {model_path}. Train first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BansuriNoteModel(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_loader = create_test_loader(csv_path, batch_size=batch_size)
    if len(test_loader.dataset) == 0:
        print("\n  ERROR: Test set empty."); sys.exit(1)

    all_preds, all_labels, all_confs = [], [], []
    with torch.no_grad():
        for specs, labels in test_loader:
            probs = F.softmax(model(specs.to(device)), dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_confs.extend(confs.cpu().tolist())

    # Confusion matrix
    n = NUM_CLASSES
    cm = [[0]*n for _ in range(n)]
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    print("\n  Confusion Matrix (rows=true, cols=predicted):\n")
    cw = 6
    print("  " + " "*cw + "│" + "".join(f" {l:>{cw-1}}" for l in SWARA_LABELS))
    print("  " + "─"*cw + "┼" + "─"*(cw*n))
    for i, rl in enumerate(SWARA_LABELS):
        row = f"  {rl:>{cw}}│"
        for j in range(n):
            row += f"  [{cm[i][j]:>{cw-3}}]" if i == j else f" {cm[i][j]:>{cw-1}}"
        print(row)

    # Per-class metrics
    print(f"\n  {'Swara':>6}  │  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  │  {'N':>4}")
    print(f"  {'─'*6}  │  {'─'*6}  {'─'*6}  {'─'*6}  │  {'─'*4}")
    for i in range(n):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i]) - tp
        p = tp/(tp+fp) if tp+fp > 0 else 0
        r = tp/(tp+fn) if tp+fn > 0 else 0
        f1 = 2*p*r/(p+r) if p+r > 0 else 0
        print(f"  {SWARA_LABELS[i]:>6}  │  {p:>6.3f}  {r:>6.3f}  {f1:>6.3f}  │  {sum(cm[i]):>4}")

    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    print(f"\n  Overall accuracy : {correct/len(all_labels):.1%} ({correct}/{len(all_labels)})")
    print(f"  Average confidence: {sum(all_confs)/len(all_confs):.3f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(CHECKPOINT_DIR / BEST_MODEL_FILENAME))
    parser.add_argument("--csv", default=str(TEST_CSV))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    evaluate(args.model, args.csv, args.batch_size)


if __name__ == "__main__":
    main()
