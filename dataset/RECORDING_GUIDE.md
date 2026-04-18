# BansuriAI-V2 — Recording & Labeling Guide

## How to collect training data for your dad's flute

This guide walks you through recording bansuri notes so the AI model can
learn to recognize them. You do not need a professional studio. A quiet room,
a phone, and a free audio app are enough to start.

---

## What you are building

The model needs to hear many examples of each swara played on the bansuri,
so it can learn what makes a "Sa" different from a "Re". Think of it like
teaching a child — you point at things and repeat the name many times, in
slightly different ways, until the pattern clicks.

---

## Equipment you need

**Minimum (perfectly fine to start):**
- A smartphone with a voice recorder app
- A quiet room
- Your dad's bansuri

**Better (if available):**
- A USB condenser microphone (e.g. Blue Yeti, AT2020)
- Audacity (free, https://www.audacityteam.org) on a laptop
- A pop filter is optional — bansuri doesn't have plosives like voice

**Recording settings:**
- Format: WAV (not MP3 — lossy compression destroys the subtle harmonics)
- Sample rate: 44100 Hz (standard, will be resampled to 22050 in the pipeline)
- Channels: Mono preferred, stereo is fine (pipeline converts to mono)
- Bit depth: 16-bit or 24-bit

Most phone recorder apps can export WAV. On Android, "ASR Voice Recorder"
works. On iPhone, the built-in Voice Memos app exports M4A — use
"Hokusai Audio Editor" or "ShurePlus MOTIV" for WAV export.

---

## Session 1: Isolated Notes (most important — do this first)

These are the core training samples. One note per file, sustained clearly.

### What to record

For each of the 7 swaras, record **10–15 separate takes**:

| Swara | Fingering note | Takes needed |
|-------|---------------|--------------|
| Sa    | Tonic / all holes closed | 10–15 |
| Re    | Second degree | 10–15 |
| Ga    | Third degree | 10–15 |
| Ma    | Fourth degree (tivra or shuddh) | 10–15 |
| Pa    | Fifth degree | 10–15 |
| Dha   | Sixth degree | 10–15 |
| Ni    | Seventh degree | 10–15 |

**Total: 70–105 isolated note files**

### How to record each take

1. Start recording
2. Wait 0.5 seconds of silence (gives the trimmer a clean edge)
3. Play the note clearly and steadily for **1.5 to 3 seconds**
4. Wait 0.5 seconds of silence
5. Stop recording
6. Save immediately with the naming convention below

### Vary across takes (this is critical)

The model must learn that "Sa" sounds like "Sa" regardless of:
- **Dynamics** — play some takes softly (pp), some at normal volume (mf),
  some strongly (f). Do NOT play all takes at the same volume.
- **Breath intensity** — some takes with full breath support, some lighter.
- **Mic distance** — sit a bit closer for some takes, a bit further for others.
  Try 6 inches, 12 inches, 18 inches.
- **Attack** — some takes with a clean tongue attack, some with a soft air onset.
- **Sustain length** — vary between 1.5 and 3 seconds.

If every take sounds exactly the same, the model learns to recognize only
that one specific way of playing — and then fails on any real music.

### What NOT to do

- Do not use meend (slides) or gamak (oscillations) in isolated note recordings.
  Save those for phrase recordings.
- Do not record in a room with a loud fan, TV, or street noise.
- Do not play multiple notes in one isolated-note file.
- Do not whisper or hum along while playing.

---

## Session 2: Repeated Takes at Different Times

If possible, record across **2–3 different days or sessions**:
- Session A: 5 takes per note (morning, quiet room)
- Session B: 5 takes per note (evening, same room)
- Session C: 5 takes per note (different room or different day)

This teaches the model to handle room acoustics variation, which is the
biggest real-world variable after the flute itself.

---

## Session 3: Simple Phrases (optional but valuable for Phase 2)

These are short melodic patterns with 2–5 notes played in sequence.
They are NOT used for the initial single-note CNN training, but will be
valuable later for testing the full pipeline and for sequence model training.

### Suggested phrases

| Phrase name         | Notes              | Musical context              |
|--------------------|--------------------|------------------------------|
| aaroha_lower       | Sa Re Ga Ma Pa     | Ascending first half         |
| avroha_lower       | Pa Ma Ga Re Sa     | Descending first half        |
| aaroha_upper       | Pa Dha Ni Sa'      | Ascending second half        |
| avroha_upper       | Sa' Ni Dha Pa      | Descending second half       |
| sa_pa_sa           | Sa Pa Sa           | Tonic to fifth and back      |
| sa_ma_pa           | Sa Ma Pa           | Common melodic phrase         |
| full_aaroha        | Sa Re Ga Ma Pa Dha Ni | Full ascending scale      |
| full_avroha        | Ni Dha Pa Ma Ga Re Sa | Full descending scale     |

Record **2–3 takes** of each phrase. Play each note clearly and hold it
for about 0.5–1 second before moving to the next. Do not rush — clean
transitions help the decoder learn note boundaries.

### Naming phrases

Save as: `dataset/raw/phrases/aaroha_lower_take01.wav`

---

## File Naming Convention

### Isolated notes

```
dataset/raw/isolated/{Swara}/{Swara}_take{NN}.wav
```

Examples:
```
dataset/raw/isolated/Sa/Sa_take01.wav
dataset/raw/isolated/Sa/Sa_take02.wav
dataset/raw/isolated/Sa/Sa_take03.wav
dataset/raw/isolated/Re/Re_take01.wav
dataset/raw/isolated/Re/Re_take02.wav
...
dataset/raw/isolated/Ni/Ni_take15.wav
```

### Phrases

```
dataset/raw/phrases/{phrase_name}_take{NN}.wav
```

Examples:
```
dataset/raw/phrases/aaroha_lower_take01.wav
dataset/raw/phrases/avroha_lower_take01.wav
dataset/raw/phrases/sa_pa_sa_take01.wav
```

### Rules
- Use the exact swara capitalization: Sa, Re, Ga, Ma, Pa, Dha, Ni
- Take numbers are zero-padded to 2 digits: take01, take02, ... take15
- No spaces in filenames
- WAV format only

---

## Filling Out labels.csv

After recording, add one row per file to `dataset/labels.csv`.

### Column reference

| Column         | Type    | Required | Description |
|---------------|---------|----------|-------------|
| file_path     | string  | yes      | Relative path from dataset/, e.g. `raw/isolated/Sa/Sa_take01.wav` |
| label         | string  | yes      | Swara name (`Sa`) for isolated, or hyphenated sequence (`Sa-Re-Ga-Ma-Pa`) for phrases |
| category      | string  | yes      | `isolated` or `phrase` |
| duration_sec  | float   | yes      | Approximate duration in seconds (measured or estimated) |
| sample_rate   | int     | yes      | Recording sample rate (usually `44100`) |
| channels      | int     | yes      | Number of channels (`1` for mono, `2` for stereo) |
| take          | int     | yes      | Take number (1, 2, 3, ...) |
| flute_key     | string  | yes      | Key of the bansuri (e.g. `E`, `G`, `A`, `C`) |
| recording_date| string  | yes      | Date of recording session (YYYY-MM-DD) |
| notes         | string  | no       | Free-text description of the take |

### Example rows

```csv
file_path,label,category,duration_sec,sample_rate,channels,take,flute_key,recording_date,notes
raw/isolated/Sa/Sa_take01.wav,Sa,isolated,2.1,44100,1,1,E,2025-06-15,clean sustained Sa
raw/isolated/Sa/Sa_take02.wav,Sa,isolated,1.8,44100,1,2,E,2025-06-15,slightly breathy
raw/isolated/Re/Re_take01.wav,Re,isolated,2.0,44100,1,1,E,2025-06-15,clean sustained Re
raw/phrases/aaroha_lower_take01.wav,Sa-Re-Ga-Ma-Pa,phrase,5.2,44100,1,1,E,2025-06-17,ascending scale
```

### Important: flute_key

The bansuri's key determines the absolute pitch of each swara. An E-base
bansuri's "Sa" is at ~330 Hz, while a G-base bansuri's "Sa" is at ~392 Hz.
Recording the flute key lets us handle multi-flute datasets in the future.

For your dad's flute, check which key it is in (often printed near the
blowing hole, or ask him). Common keys: E, G, A, C, D.

If you don't know, write `unknown` — the model will still work, it just
won't generalize to a different flute as easily.

---

## Workflow: Record → Name → Label → Validate → Split

### Step-by-step

```
1. Record all isolated notes (Session 1 + Session 2)
        ↓
2. Record phrases if desired (Session 3)
        ↓
3. Transfer .wav files from phone/recorder to computer
        ↓
4. Place files in the correct folders:
        dataset/raw/isolated/Sa/Sa_take01.wav
        dataset/raw/isolated/Sa/Sa_take02.wav
        ...
        ↓
5. Open dataset/labels.csv and add one row per file
        ↓
6. Run the validator:
        cd BansuriAI-V2
        python dataset/validate_dataset.py
        ↓
   Fix any errors it reports, then run again until clean
        ↓
7. Run the splitter:
        python dataset/split_dataset.py
        ↓
   Outputs: dataset/splits/train.csv, val.csv, test.csv
        ↓
8. Ready for feature extraction and training (Phase 2)
```

---

## How much data is enough?

| Dataset size per note | Expected result |
|-----------------------|----------------|
| 3–5 takes            | Minimum. Model may learn but will be fragile. |
| 10–15 takes          | Good starting point. Expect ~70–80% accuracy. |
| 20–30 takes          | Strong. Expect ~85–90% accuracy with augmentation. |
| 50+ takes            | Excellent. Diminishing returns above this. |

With 10 takes per note × 7 notes = 70 files, plus data augmentation
(time shift, noise injection, pitch shift applied during training),
the effective training set is much larger than the raw file count.

**Start with 10 per note. You can always record more later.**

---

## Tips for recording your dad's bansuri

- **Let him warm up first.** The first few minutes of playing have unstable
  pitch. Wait until he's settled into the flute.
- **Record him, don't ask him to "perform."** Natural playing with relaxed
  breath support gives more realistic training data than stiff, self-conscious
  recordings.
- **Note which flute he uses.** If he has multiple bansuris in different keys,
  label which flute was used. Mixing flutes without labeling confuses the model.
- **Record in his usual practice space.** The model should learn to recognize
  notes in the same acoustic environment where it will be tested.
- **Short sessions are fine.** 15–20 minutes per session is plenty. Recording
  fatigue makes later takes worse, not better.
