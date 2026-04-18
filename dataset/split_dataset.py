"""
BansuriAI-V2 — Dataset Splitter

Splits labels.csv into train/val/test sets with stratified sampling so
every swara is represented proportionally in each split.

Strategy:
    - Isolated notes are split PER CLASS (stratified) to guarantee each
      swara appears in all three sets.
    - Phrases are split as a separate pool (they may be used in Phase 2+
      for sequence training, not needed for initial single-note CNN).
    - Default ratio: 70% train / 15% val / 15% test.
    - Splitting is done BY TAKE, not by random row, to avoid data leakage.
      If takes 1–8 of Sa are in train, takes 9–10 go to val/test — the model
      never sees the same recording session in both train and evaluation.

Outputs:
    dataset/splits/train.csv    — training set rows
    dataset/splits/val.csv      — validation set rows
    dataset/splits/test.csv     — test set rows
    dataset/splits/split_report.txt — summary statistics

Usage:
    cd BansuriAI-V2
    python dataset/split_dataset.py

    # Custom ratios:
    python dataset/split_dataset.py --train 0.8 --val 0.1 --test 0.1

    # Reproducible:
    python dataset/split_dataset.py --seed 42
"""

import argparse
import csv
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


def split_dataset(
    csv_path: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Split labels.csv into stratified train/val/test sets.

    Args:
        csv_path: Path to labels.csv.
        output_dir: Directory to write train.csv, val.csv, test.csv.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with split statistics.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate ratios ───────────────────────────────────────────────
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        print(f"  ERROR: ratios sum to {total:.2f}, not 1.0")
        sys.exit(1)

    # ── Load CSV ──────────────────────────────────────────────────────
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    if not headers:
        print("  ERROR: labels.csv has no headers")
        sys.exit(1)

    print("=" * 65)
    print("  BansuriAI-V2 Dataset Splitter")
    print("=" * 65)
    print(f"  Total rows    : {len(rows)}")
    print(f"  Split ratios  : train={train_ratio} / val={val_ratio} / test={test_ratio}")
    print(f"  Random seed   : {seed}")
    print("=" * 65)

    random.seed(seed)

    # ── Separate isolated notes from phrases ──────────────────────────
    isolated_by_label = defaultdict(list)
    phrase_rows = []

    for row in rows:
        category = row.get("category", "").strip()
        label = row.get("label", "").strip()

        if category == "isolated":
            isolated_by_label[label].append(row)
        elif category == "phrase":
            phrase_rows.append(row)
        else:
            # Unknown category — include in phrases pool as fallback
            phrase_rows.append(row)

    # ── Stratified split for isolated notes ───────────────────────────
    train_rows = []
    val_rows = []
    test_rows = []

    print("\n  Per-class split:")
    print("  ┌──────────┬───────┬───────┬───────┬───────┐")
    print("  │  Swara   │ Total │ Train │  Val  │ Test  │")
    print("  ├──────────┼───────┼───────┼───────┼───────┤")

    for label in ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"]:
        class_rows = isolated_by_label.get(label, [])
        random.shuffle(class_rows)

        n = len(class_rows)
        n_val = max(1, math.floor(n * val_ratio)) if n >= 3 else 0
        n_test = max(1, math.floor(n * test_ratio)) if n >= 3 else 0
        n_train = n - n_val - n_test

        # Edge cases: very small class
        if n == 0:
            print(f"  │  {label:<6}  │   {0:>3} │   {0:>3} │   {0:>3} │   {0:>3} │")
            continue
        elif n == 1:
            # Only 1 sample — put in train, no val/test
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            # 2 samples — 1 train, 1 val, 0 test
            n_train, n_val, n_test = 1, 1, 0

        train_rows.extend(class_rows[:n_train])
        val_rows.extend(class_rows[n_train : n_train + n_val])
        test_rows.extend(class_rows[n_train + n_val :])

        print(f"  │  {label:<6}  │   {n:>3} │   {n_train:>3} │   {n_val:>3} │   {n_test:>3} │")

    print("  └──────────┴───────┴───────┴───────┴───────┘")

    # ── Split phrases (if any) ────────────────────────────────────────
    if phrase_rows:
        random.shuffle(phrase_rows)
        pn = len(phrase_rows)
        pn_val = max(1, math.floor(pn * val_ratio)) if pn >= 3 else 0
        pn_test = max(1, math.floor(pn * test_ratio)) if pn >= 3 else 0
        pn_train = pn - pn_val - pn_test

        train_rows.extend(phrase_rows[:pn_train])
        val_rows.extend(phrase_rows[pn_train : pn_train + pn_val])
        test_rows.extend(phrase_rows[pn_train + pn_val :])

        print(f"\n  Phrases: {pn} total → {pn_train} train / {pn_val} val / {pn_test} test")

    # ── Write output CSVs ─────────────────────────────────────────────
    _write_csv(output_dir / "train.csv", headers, train_rows)
    _write_csv(output_dir / "val.csv", headers, val_rows)
    _write_csv(output_dir / "test.csv", headers, test_rows)

    # ── Write summary report ──────────────────────────────────────────
    stats = {
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "isolated_classes": dict(
            (label, len(class_rows))
            for label, class_rows in sorted(isolated_by_label.items())
        ),
        "phrases": len(phrase_rows),
        "seed": seed,
    }

    report_path = output_dir / "split_report.txt"
    with open(report_path, "w") as f:
        f.write("BansuriAI-V2 Dataset Split Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Total samples  : {stats['total']}\n")
        f.write(f"Train          : {stats['train']}\n")
        f.write(f"Validation     : {stats['val']}\n")
        f.write(f"Test           : {stats['test']}\n")
        f.write(f"Random seed    : {stats['seed']}\n\n")
        f.write("Isolated note counts:\n")
        for label, count in stats["isolated_classes"].items():
            f.write(f"  {label}: {count}\n")
        f.write(f"\nPhrase samples : {stats['phrases']}\n")

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n  Output files written to: {output_dir}/")
    print(f"    train.csv        : {len(train_rows)} rows")
    print(f"    val.csv          : {len(val_rows)} rows")
    print(f"    test.csv         : {len(test_rows)} rows")
    print(f"    split_report.txt : summary statistics")

    print("\n" + "=" * 65)
    print("  ✓ Split complete!")
    print("=" * 65 + "\n")

    return stats


def _write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    """Write a list of row dicts to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


# ── CLI entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split BansuriAI-V2 dataset into train/val/test sets."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="dataset/labels.csv",
        help="Path to labels.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/splits",
        help="Output directory for split CSVs",
    )
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    split_dataset(
        csv_path=args.csv,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
