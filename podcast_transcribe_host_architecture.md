# Podcast Transcription & Host Identification Pipeline

This document describes the architecture and processing flow of `podcast_transcribe_host.py`.

---

## 🧠 System Flow Diagram

```mermaid
flowchart TD

    A[Start main()] --> B[parse_args()]

    B --> C[Validate input dir & HF token]

    C --> D[Load preferred terms + replacement map]

    D --> E[get_device()]
    E --> F[Initialize WhisperModel]

    F --> G[Load pyannote diarization Pipeline]
    G --> H[Initialize SpeakerRecognition model]

    H --> I[Load known speaker profiles]

    I --> J[Scan input_dir for audio files]

    J --> K{Any audio files?}
    K -->|No| Z[Exit with error]
    K -->|Yes| L[Loop: process_file(audio)]

    %% --- PROCESS FILE ---
    L --> M[transcribe_audio()]
    M --> N[diarize_audio()]

    N --> O[assign_speakers_to_segments()]

    O --> P[load_audio_mono_16k()]
    P --> Q[load_host_profile()]

    Q --> R[choose_host_speaker()]

    R --> S[match_known_speakers()]

    S --> T{Known host override?}
    T -->|Yes| U[Set host_speaker]
    T -->|No| V[Keep detected host]

    U --> W
    V --> W

    W[rename_speakers()] --> X[coalesce_segments()]

    X --> Y[collect_review_rows()]

    %% --- OUTPUTS ---
    Y --> O1[write_text_transcript (all speakers)]
    Y --> O2[write_text_transcript (host only)]
    Y --> O3[write_review_csv()]
    Y --> O4[write_json_output()]

    O4 --> O5{Update host profile?}
    O5 -->|Yes| O6[save_host_profile()]
    O5 -->|No| O7[Skip]

    O6 --> P1
    O7 --> P1

    P1[build_episode_summary_row()] --> P2[Append to summary list]

    P2 --> L

    %% --- FINAL ---
    L --> Q1[write_episode_summary_csv()]
    Q1 --> R1[End]
```

---

## 🔍 Architectural Layers

### 1. Input & Configuration Layer

Handles CLI arguments and environment setup.

**Key responsibilities:**
- Parse runtime parameters (`parse_args`)
- Validate input directory and Hugging Face token
- Load:
  - Preferred vocabulary terms
  - Replacement mappings
  - Known speaker configurations
  - Host voice profile (if available)

---

### 2. Core Processing Pipeline (Per Audio File)

#### A. Transcription
- **Function:** `transcribe_audio`
- **Model:** Faster-Whisper

**Outputs:**
- Time-aligned segments
- Word-level timestamps
- Confidence metadata

---

#### B. Diarization
- **Function:** `diarize_audio`
- **Model:** Pyannote

**Followed by:**
- `assign_speakers_to_segments`

**Purpose:**
- Identify speaker turns
- Align speakers to transcript segments and words

---

#### C. Speaker Intelligence

This is the most advanced part of the system.

**Key functions:**
- `choose_host_speaker`
- `match_known_speakers`
- `rename_speakers`

**Capabilities:**
- Speaker embedding generation (SpeechBrain)
- Host identification via:
  - Reference audio
  - Persistent profile
  - Cosine similarity
- Fallback:
  - Dominant speaker assumption
- Known speaker matching
- Final label normalization (HOST, SPEAKER_01, etc.)

---

### 3. Text Normalization Layer

**Function:** `coalesce_segments`

**Features:**
- Merge adjacent segments from same speaker
- Apply glossary replacements
- Normalize transcription output
- Track replacement events for QA

---

### 4. Quality & Review System

**Function:** `collect_review_rows`

Generates structured QA signals:

**Examples:**
- Host not detected
- Low-confidence host match
- Ambiguous speaker identification
- Low host speech coverage
- Glossary corrections applied

---

### 5. Output Layer

For each episode:

- **Speaker transcript (TXT)**
- **Host-only transcript (TXT)**
- **Structured transcript (JSON)**
- **Review report (CSV)**

Batch-level:

- `_episode_review_summary.csv`

---

## ⚡ Notable Design Features

### 🔹 Adaptive Host Identification
- Learns host voice over time
- Updates embedding profile across episodes

---

### 🔹 Multi-Stage Speaker Resolution
1. Diarization (raw speaker IDs)  
2. Embedding similarity matching  
3. Known speaker override  
4. Final label normalization  

---

### 🔹 Feedback Loop
- Host profile persists and improves accuracy over time

---

### 🔹 Built-in QA System
- Automatically flags:
  - Weak matches
  - Ambiguities
  - Data quality issues

---

## 📌 Summary

This pipeline goes beyond transcription:

> It is a **speaker-aware, self-improving podcast intelligence system** that combines ASR, diarization, speaker recognition, and QA into a unified workflow.
