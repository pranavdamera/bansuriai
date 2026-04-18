"""
BansuriAI-V2 — Dataset Validator

Validates the dataset before training by checking every row in labels.csv
against the actual audio files on disk.

Checks performed:
    1. CSV schema      — required columns present, no empty cells
    2. File existence   — every listed .wav file actually exists
    3. Audio integrity  — each file loads without error via soundfile
    4. Duration bounds  — each clip falls within min/max limits
    5. Label validity   — every label is a recognized swara or phrase format
    6. Folder agreement — file's parent folder matches its label (isolated only)
    7. Duplicate check  — no duplicate file paths in the CSV
    8. Balance report   — per-class sample counts to spot severe imbalance

Usage:
    cd BansuriAI-V2
    python dataset/validate_dataset.py

    # With custom paths:
    python dataset/validate_dataset.py --csv dataset/labels.csv --root dataset
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf


# ── Constants ─────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "file_path",
    "label",
    "category",
    "duration_sec",
    "sample_rate",
    "channels",
    "take",
    "flute_key",
    "recording_date",
    "notes",
]

VALID_SWARAS = {"Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"}
VALID_CATEGORIES = {"isolated", "phrase"}

# Duration limits (seconds) — generous bounds for validation
MIN_DURATION = 0.3    # Anything shorter is probably a mistake
MAX_DURATION = 30.0   # Matches backend config MAX_DURATION_SEC


def validate_dataset(csv_path: str, dataset_root: str) -> bool:
    """Run all validation checks and print a report.

    Args:
        csv_path: Path to labels.csv.
        dataset_root: Root directory of the dataset (parent of raw/).

    Returns:
        True if all checks pass, False if any errors were found.
    """
    csv_path = Path(csv_path)
    dataset_root = Path(dataset_root)

    errors: list[str] = []
    warnings: list[str] = []

    print("=" * 65)
    print("  BansuriAI-V2 Dataset Validator")
    print("=" * 65)
    print(f"  CSV file    : {csv_path}")
    print(f"  Dataset root: {dataset_root}")
    print("=" * 65)

    # ── Check 0: CSV file exists ──────────────────────────────────────
    if not csv_path.exists():
        print(f"\n  FATAL: labels.csv not found at {csv_path}")
        return False

    # ── Load CSV ──────────────────────────────────────────────────────
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    print(f"\n  Loaded {len(rows)} rows with columns: {headers}\n")

    # ── Check 1: Schema — required columns ────────────────────────────
    print("  [1/8] Schema check...")
    missing_cols = set(REQUIRED_COLUMNS) - set(headers)
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(sorted(missing_cols))}")
    else:
        print("        OK — all required columns present")

    # ── Check 2–7: Per-row validation ─────────────────────────────────
    print("  [2/8] File existence check...")
    print("  [3/8] Audio integrity check...")
    print("  [4/8] Duration bounds check...")
    print("  [5/8] Label validity check...")
    print("  [6/8] Folder-label agreement check...")
    print("  [7/8] Duplicate path check...")

    seen_paths = set()
    label_counts = Counter()
    category_counts = Counter()
    files_checked = 0
    files_ok = 0

    for row_idx, row in enumerate(rows, start=2):  # Row 2 = first data row
        file_path = row.get("file_path", "").strip()
        label = row.get("label", "").strip()
        category = row.get("category", "").strip()
        duration_str = row.get("duration_sec", "").strip()

        row_prefix = f"Row {row_idx} ({file_path})"

        # ── Empty required fields ─────────────────────────────────────
        if not file_path:
            errors.append(f"{row_prefix}: file_path is empty")
            continue
        if not label:
            errors.append(f"{row_prefix}: label is empty")
        if not category:
            errors.append(f"{row_prefix}: category is empty")

        # ── Duplicate detection ───────────────────────────────────────
        if file_path in seen_paths:
            errors.append(f"{row_prefix}: DUPLICATE file_path")
        seen_paths.add(file_path)

        # ── Category check ────────────────────────────────────────────
        if category and category not in VALID_CATEGORIES:
            errors.append(
                f"{row_prefix}: unknown category '{category}'. "
                f"Expected: {', '.join(sorted(VALID_CATEGORIES))}"
            )

        # ── Label check ───────────────────────────────────────────────
        if category == "isolated":
            if label not in VALID_SWARAS:
                errors.append(
                    f"{row_prefix}: invalid isolated label '{label}'. "
                    f"Expected one of: {', '.join(sorted(VALID_SWARAS))}"
                )
            label_counts[label] += 1

        elif category == "phrase":
            # Phrase labels are hyphen-separated swaras: "Sa-Re-Ga"
            phrase_notes = label.split("-")
            invalid_notes = [n for n in phrase_notes if n not in VALID_SWARAS]
            if invalid_notes:
                errors.append(
                    f"{row_prefix}: phrase contains invalid swara(s): "
                    f"{', '.join(invalid_notes)}"
                )
            label_counts[f"phrase:{label}"] += 1

        category_counts[category] += 1

        # ── File existence ────────────────────────────────────────────
        full_path = dataset_root / file_path
        if not full_path.exists():
            errors.append(f"{row_prefix}: FILE NOT FOUND at {full_path}")
            continue

        files_checked += 1

        # ── Audio integrity + duration ────────────────────────────────
        try:
            info = sf.info(str(full_path))
            actual_duration = info.duration
            actual_sr = info.samplerate
            actual_channels = info.channels

            # Duration bounds
            if actual_duration < MIN_DURATION:
                errors.append(
                    f"{row_prefix}: too short ({actual_duration:.2f}s < {MIN_DURATION}s)"
                )
            elif actual_duration > MAX_DURATION:
                errors.append(
                    f"{row_prefix}: too long ({actual_duration:.2f}s > {MAX_DURATION}s)"
                )

            # Cross-check CSV metadata against actual file
            if duration_str:
                csv_dur = float(duration_str)
                if abs(csv_dur - actual_duration) > 0.5:
                    warnings.append(
                        f"{row_prefix}: CSV duration ({csv_dur:.1f}s) differs from "
                        f"actual ({actual_duration:.2f}s) by more than 0.5s"
                    )

            files_ok += 1

        except Exception as e:
            errors.append(f"{row_prefix}: CANNOT READ AUDIO — {e}")

        # ── Folder-label agreement (isolated only) ────────────────────
        if category == "isolated":
            parent_folder = Path(file_path).parent.name
            if parent_folder != label:
                errors.append(
                    f"{row_prefix}: folder '{parent_folder}' does not match "
                    f"label '{label}'"
                )

    # ── Check 8: Balance report ───────────────────────────────────────
    print("  [8/8] Class balance report...\n")

    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  ISOLATED NOTE DISTRIBUTION                     │")
    print("  ├──────────┬──────────┬───────────────────────────┤")
    print("  │  Swara   │  Count   │  Bar                      │")
    print("  ├──────────┼──────────┼───────────────────────────┤")

    isolated_labels = {k: v for k, v in label_counts.items() if not k.startswith("phrase:")}
    max_count = max(isolated_labels.values()) if isolated_labels else 1

    for swara in ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"]:
        count = isolated_labels.get(swara, 0)
        bar_len = int((count / max_count) * 20) if max_count > 0 else 0
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  │  {swara:<6}  │  {count:<6}  │  {bar}  │")

        if count == 0:
            warnings.append(f"No samples for swara '{swara}' — model cannot learn this note")
        elif count < 5:
            warnings.append(f"Only {count} sample(s) for '{swara}' — aim for at least 10–15")

    print("  └──────────┴──────────┴───────────────────────────┘")

    phrase_labels = {k: v for k, v in label_counts.items() if k.startswith("phrase:")}
    if phrase_labels:
        print(f"\n  Phrases: {sum(phrase_labels.values())} total across "
              f"{len(phrase_labels)} unique sequence(s)")

    print(f"\n  Categories: {dict(category_counts)}")

    # ── Final report ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  Files checked : {files_checked}")
    print(f"  Files OK      : {files_ok}")
    print(f"  Errors        : {len(errors)}")
    print(f"  Warnings      : {len(warnings)}")
    print("=" * 65)

    if errors:
        print("\n  ERRORS (must fix before training):")
        for i, err in enumerate(errors, 1):
            print(f"    {i}. {err}")

    if warnings:
        print("\n  WARNINGS (should fix, but won't block):")
        for i, warn in enumerate(warnings, 1):
            print(f"    {i}. {warn}")

    if not errors and not warnings:
        print("\n  ✓ Dataset is clean and ready for training!")
    elif not errors:
        print("\n  ✓ No errors — dataset is usable, but review warnings above.")
    else:
        print("\n  ✗ Fix errors above before proceeding to training.")

    print()
    return len(errors) == 0


# ── CLI entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate BansuriAI-V2 dataset metadata and audio files."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="dataset/labels.csv",
        help="Path to labels.csv (default: dataset/labels.csv)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="dataset",
        help="Dataset root directory containing raw/ (default: dataset)",
    )
    args = parser.parse_args()

    ok = validate_dataset(args.csv, args.root)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
