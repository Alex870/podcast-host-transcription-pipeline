# Podcast Transcription & Host Identification Pipeline

This document describes the architecture and processing flow of `podcast_transcribe_host.py`.

---

## 🧠 System Flow Diagram

```mermaid
flowchart LR

    %% --- SETUP ---
    A[Start] --> B[parse_args]
    B --> C[Validate inputs]
    C --> D[Load configs]
    D --> E[get_device]
    E --> F[Init Whisper]
    F --> G[Init Diarization]
    G --> H[Init Speaker Model]
    H --> I[Load Known Speakers]
    I --> J[Scan Audio Files]

    J --> K{Files Found?}
    K -->|No| Z[Exit]
    K -->|Yes| L[Process File Loop]

    %% --- PER FILE ---
    subgraph Processing
        L --> M[Transcribe]
        M --> N[Diarize]
        N --> O[Assign Speakers]
        O --> P[Load Audio 16k]
        P --> Q[Load Host Profile]
        Q --> R[Detect Host]
        R --> S[Match Known Speakers]
        S --> T[Rename Speakers]
        T --> U[Normalize Text]
        U --> V[Collect QA Signals]
    end

    %% --- OUTPUTS ---
    subgraph Outputs
        V --> O1[Transcript All]
        V --> O2[Transcript Host Only]
        V --> O3[Review CSV]
        V --> O4[JSON Output]
    end

    O4 --> W{Update Profile?}
    W -->|Yes| X[Save Profile]
    W -->|No| Y[Skip]

    X --> AA[Build Episode Summary]
    Y --> AA

    AA --> L

    %% --- FINAL ---
    L --> AB[Write Summary CSV]
    AB --> AC[End]
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
