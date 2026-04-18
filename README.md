# BansuriAI — Real-Time Bansuri Flute Tutor

> A real-time pitch classification and feedback system for bansuri flute practice, built with a TensorFlow CNN, MFCC-based audio features, and a FastAPI + React Native mobile client.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi) ![React Native](https://img.shields.io/badge/React_Native-mobile-blue?logo=react) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

BansuriAI is a real-time music tutor for the bansuri — a North Indian bamboo flute. It listens to live audio input, extracts pitch features, runs on-device classification, and delivers instant feedback on intonation accuracy.

Built because standard tuner apps don't understand the ornaments, microtones, and tonal character of bansuri playing. BansuriAI classifies not just "sharp" or "flat" but identifies which note a phrase is being played on and whether the tonal shaping is consistent with the expected timbre of that note on a bansuri.

**87% pitch classification accuracy across 1,200+ labeled audio samples. Sub-100ms end-to-end latency.**

---

## Architecture

```
Live Audio (React Native mic)
         │
         ▼
┌──────────────────────┐
│  Audio Buffer (PCM)  │  512-sample frames at 22,050 Hz
└────────┬─────────────┘
         │  HTTP stream
         ▼
┌──────────────────────┐
│  FastAPI Backend     │
│  ├── Librosa         │  MFCC extraction (40 coefficients)
│  ├── CNN Inference   │  13-class pitch classification
│  └── Feedback Engine │  Error type + severity scoring
└────────┬─────────────┘
         │  JSON response
         ▼
┌──────────────────────┐
│  React Native UI     │  Real-time pitch overlay,
│                      │  session accuracy summary
└──────────────────────┘
```

---

## Model

- **Architecture:** 2D CNN over MFCC spectrograms
- **Input:** 40 MFCC coefficients × 128 time frames
- **Output:** 13-class softmax (Sa, Re, Ga, Ma, Pa, Dha, Ni × octave)
- **Training data:** 1,200+ labeled bansuri audio clips (self-recorded + augmented)
- **Accuracy:** 87% on held-out test set

**Data augmentation applied:** pitch shift ±2 cents, time stretch 0.9–1.1x, additive noise (SNR 20–40dB)

---

## Results

| Metric          | Value  |
|-----------------|--------|
| Test Accuracy   | 87%    |
| Inference (avg) | ~85ms  |
| Session Length  | 8+ min avg continuous input |
| Notes Classified| 13 (full octave Sa–Ni) |

---

## Quickstart

### Backend

```bash
git clone https://github.com/pranavdamera/bansuriai
cd bansuriai/backend
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload --port 8000
```

### Mobile App (Expo)

```bash
cd bansuriai/mobile
npm install
npx expo start
```

Point the app at your local backend IP in `config.js`.

---

## API

### `POST /classify`

```json
{
  "audio_b64": "<base64-encoded PCM chunk>",
  "sample_rate": 22050
}
```

**Response:**

```json
{
  "predicted_note": "Pa",
  "confidence": 0.91,
  "intonation": "sharp",
  "cents_off": 12,
  "feedback": "Slightly sharp on Pa — loosen embouchure."
}
```

---

## Project Structure

```
bansuriai/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI routes
│   │   ├── classifier.py    # CNN inference
│   │   └── feedback.py      # Error scoring logic
│   ├── model/
│   │   ├── train.py
│   │   └── cnn_model.h5
│   └── requirements.txt
├── mobile/
│   ├── App.js
│   ├── screens/
│   └── config.js
└── README.md
```

---

## Why Bansuri?

Most pitch detection tools work for Western equal temperament. Bansuri is inherently microtonal — notes like Komal Re and Teevra Ma exist at intervals that standard tuners misread. BansuriAI is trained on actual bansuri recordings and understands the instrument's natural tonal space.

---

## License

MIT
