"""
BansuriAI-V2 — Synthetic Data Generator

Creates sine-wave .wav files for each swara at approximate E-base bansuri
frequencies. NOT musically accurate — meant for pipeline validation only.

Usage:  cd BansuriAI-V2 && python training/generate_synthetic_data.py
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import soundfile as sf

SWARA_FREQUENCIES = {
    "Sa": 330.0, "Re": 370.0, "Ga": 415.0, "Ma": 440.0,
    "Pa": 495.0, "Dha": 554.0, "Ni": 622.0,
}
SAMPLE_RATE = 44100


def generate_synthetic_dataset(output_dir, takes_per_note=15, seed=42):
    output_dir = Path(output_dir)
    np.random.seed(seed)

    print("=" * 60)
    print("  BansuriAI-V2 — Synthetic Data Generator")
    print(f"  Output: {output_dir}  |  Takes/note: {takes_per_note}")
    print("=" * 60)

    csv_rows = []
    for swara, base_freq in SWARA_FREQUENCIES.items():
        swara_dir = output_dir / "raw" / "isolated" / swara
        swara_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  {swara} ({base_freq:.0f} Hz)...")

        for take in range(1, takes_per_note + 1):
            freq = base_freq * (1 + np.random.uniform(-0.03, 0.03))
            duration = np.random.uniform(1.5, 3.0)
            amplitude = np.random.uniform(0.3, 0.9)

            waveform = _generate_note(freq, duration, amplitude)
            silence = np.zeros(int(SAMPLE_RATE * 0.3))
            waveform = np.concatenate([silence, waveform, silence])

            filename = f"{swara}_take{take:02d}.wav"
            sf.write(str(swara_dir / filename), waveform, SAMPLE_RATE)

            csv_rows.append({
                "file_path": f"raw/isolated/{swara}/{filename}",
                "label": swara, "category": "isolated",
                "duration_sec": f"{len(waveform)/SAMPLE_RATE:.2f}",
                "sample_rate": str(SAMPLE_RATE), "channels": "1",
                "take": str(take), "flute_key": "E",
                "recording_date": "2025-01-01",
                "notes": f"synthetic {swara} at {freq:.1f}Hz",
            })

    csv_path = output_dir / "labels.csv"
    headers = ["file_path", "label", "category", "duration_sec", "sample_rate",
               "channels", "take", "flute_key", "recording_date", "notes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n  labels.csv: {len(csv_rows)} rows")
    print(f"  Next: validate → split → train\n")


def _generate_note(frequency, duration, amplitude):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    wave = np.sin(2*np.pi*frequency*t) + 0.5*np.sin(4*np.pi*frequency*t) + 0.25*np.sin(6*np.pi*frequency*t)
    peak = np.max(np.abs(wave))
    if peak > 0: wave /= peak
    # Envelope: attack 50ms, release 100ms
    env = np.ones(len(t), dtype=np.float32)
    att = int(0.05 * SAMPLE_RATE)
    rel = int(0.10 * SAMPLE_RATE)
    if att < len(t): env[:att] = np.linspace(0, 1, att)
    if rel < len(t): env[-rel:] = np.linspace(1, 0, rel)
    return wave * env * amplitude


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset")
    parser.add_argument("--takes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_synthetic_dataset(args.output, args.takes, args.seed)
