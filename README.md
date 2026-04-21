# Podcast Host Transcription Pipeline

This project batch-processes podcast audio files into speaker-labeled transcripts, host-only extracts, JSON metadata, and review CSVs. The output is suitable for ingesting into a RAG pipeline for storing in a vector database.  It combines speech-to-text, speaker diarization, speaker-embedding matching, and terminology normalization so episodes can be transcribed in a way that is more useful for editorial review and downstream content workflows.

The repository is designed for shows where identifying the host matters. In addition to generic speaker diarization, it can:

- label the host from a one-time reference clip
- maintain a persistent host voice profile across episodes
- identify recurring named speakers from a reference-sample directory
- create a host-only transcript for faster review
- flag episodes and transcript segments that likely need manual verification

## What The Project Does

For each supported audio file in an input folder, the pipeline:

1. Transcribes the episode with `faster-whisper`
2. Runs speaker diarization with `pyannote.audio`
3. Builds speaker embeddings with `speechbrain`
4. Tries to identify the host from:
   - a selected host reference clip
   - a saved `host_profile.json`
   - a known speaker marked as the host
   - or, optionally, the dominant speaker as a bootstrap fallback
5. Renames diarized speakers to `HOST`, known names, or `SPEAKER_01`, `SPEAKER_02`, and so on
6. Applies preferred-term biasing and post-transcription replacement cleanup
7. Writes transcript, JSON, and review outputs back to disk

## Repository Contents

- `podcast_transcribe_host.py`: main Python pipeline
- `Convert MP3 to TXT diarized.ps1`: Windows PowerShell launcher with persisted source-folder setup and first-run host-sample onboarding
- `podcast_transcribe_config.example.json`: example runtime configuration file
- `podcast_transcribe_requirements.txt`: Python package list
- `preferred_terms.txt`: optional glossary for domain-specific spellings
- `preferred_replacements.json`: optional post-processing replacements for common mistranscriptions
- `speaker_reference_samples/speakers.json`: sample configuration for recurring known speakers
- `podcast_transcribe_README.md`: legacy project notes and feature details

## Technical Details

The pipeline is centered around three model-driven stages:

- Transcription: `faster-whisper` performs speech-to-text with word timestamps enabled.
- Diarization: `pyannote/speaker-diarization-community-1` assigns speaker turns across the episode.
- Speaker matching: `speechbrain/spkrec-ecapa-voxceleb` generates embeddings used to match diarized speakers against the host profile or known speaker references.

Important implementation details:

- Audio is normalized to mono 16 kHz before speaker embedding extraction.
- Host matching uses cosine similarity against a host reference embedding or saved host profile.
- Known speakers are matched one-to-one against diarized speakers when similarity clears the configured threshold.
- The host profile can be updated over time from matched host speech to improve stability across episodes.
- A review-priority score is generated per episode so the riskiest outputs can be checked first.

Supported audio formats:

- `.mp3`
- `.wav`
- `.m4a`
- `.flac`
- `.ogg`

Generated outputs per audio file:

- `*_speaker_transcript.txt`
- `*_host_only.txt`
- `*_review.csv`
- `*_speaker_transcript.json`

Generated output per batch:

- `_episode_review_summary.csv`

## Requirements

This project currently assumes a Windows workflow because the included launcher is a PowerShell script that opens Windows folder and file dialogs.

You will need:

- Python installed and available on `PATH`
- Conda if you want to use the launcher exactly as written
- A Hugging Face account and access token
- Access approval for `pyannote/speaker-diarization-community-1` on Hugging Face
- Enough local compute for Whisper, pyannote, and PyTorch-based audio processing

Python dependencies:

- `faster-whisper`
- `pyannote.audio`
- `speechbrain`
- `torchaudio`

## First-Time Setup

### 1. Clone the repository

```powershell
git clone https://github.com/Alex870/podcast-host-transcription-pipeline.git
cd podcast-host-transcription-pipeline
```

### 2. Create a Python environment

The PowerShell launcher currently runs `conda activate whisper`, so the path of least resistance is to create an environment with that name.

Example:

```powershell
conda create -n whisper python=3.11 -y
conda activate whisper
pip install -r podcast_transcribe_requirements.txt
```

If you prefer a different environment name or a plain virtual environment, that is fine, but you will need to update `Convert MP3 to TXT diarized.ps1` so it does not assume `conda activate whisper`.

### 3. Get a Hugging Face token

The diarization pipeline will not run without a valid Hugging Face token.

First:

- Sign in to Hugging Face
- Request and accept access to `pyannote/speaker-diarization-community-1`
- Create an access token

Then provide the token in one of these ways:

- Set `HF_TOKEN` in your shell environment
- or place it in `podcast_transcribe_config.json`

Example for the current shell:

```powershell
$env:HF_TOKEN = "your_token_here"
```

### 4. Create your runtime config file

Copy the example file to a working config:

```powershell
Copy-Item .\podcast_transcribe_config.example.json .\podcast_transcribe_config.json
```

Recommended first-pass config:

```json
{
  "default_source_dir": "D:/Speech_to_text/audio",
  "hf_token": "",
  "known_speakers_dir": "speaker_reference_samples",
  "preferred_terms_file": "preferred_terms.txt",
  "replacement_map_json": "preferred_replacements.json",
  "host_profile_json": "host_profile.json",
  "model": "large-v3",
  "language": "en",
  "compute_type": "auto",
  "beam_size": 5,
  "batch_size": 8,
  "assume_dominant_speaker_is_host": true,
  "host_threshold": 0.45
}
```

Configuration notes:

- `default_source_dir`: starting folder shown in the launcher dialog
- `hf_token`: optional fallback if `HF_TOKEN` is not already set in the environment
- `known_speakers_dir`: folder containing `speakers.json` and reference clips
- `preferred_terms_file`: glossary terms to bias transcription
- `replacement_map_json`: preferred replacements for cleanup after transcription
- `host_profile_json`: persistent host voice profile created over time
- `model`: Whisper model name
- `language`: language code passed to Whisper
- `compute_type`: Whisper compute setting such as `auto`, `float16`, or `int8`
- `beam_size`: decode beam size
- `batch_size`: transcription batch size
- `assume_dominant_speaker_is_host`: fallback host bootstrap if no better match exists
- `host_threshold`: speaker similarity threshold for host and known-speaker matching

### 5. Optional: set up known speaker samples

If you want stable speaker naming across episodes, add clean reference clips to `speaker_reference_samples` and edit `speaker_reference_samples/speakers.json`.

Example:

```json
{
  "speakers": [
    {
      "name": "HOST",
      "is_host": true,
      "files": ["host_sample.wav"]
    },
    {
      "name": "Guest_A",
      "files": ["guest_a_sample.wav"]
    }
  ]
}
```

Best practices:

- use short, clean clips with only one speaker
- avoid overlap, music beds, and heavy background noise
- provide more than one clip per recurring speaker when possible

## Running The Project

### Option 1: Use the PowerShell launcher

This is the easiest way to run the project on Windows.

```powershell
conda activate whisper
.\Convert MP3 to TXT diarized.ps1
```

The launcher will:

- load `podcast_transcribe_config.json` if it already exists
- ask whether you want to change the saved `default_source_dir` when one is already configured
- automatically open the folder picker and save the result into `podcast_transcribe_config.json` when no default source folder has been set yet
- only prompt for a clean host reference clip on first run, when `podcast_transcribe_config.json` does not already exist
- create or update `speaker_reference_samples/speakers.json` with the selected host sample during that first-run setup
- fall back to the dominant-speaker approach if you cancel the host-sample prompt
- pass the configured options into `podcast_transcribe_host.py`

### Option 2: Run the Python script directly

Example:

```powershell
python .\podcast_transcribe_host.py `
  --input-dir "D:\Speech_to_text\audio" `
  --output-dir "D:\Speech_to_text\audio" `
  --model large-v3 `
  --language en `
  --compute-type auto `
  --beam-size 5 `
  --batch-size 8 `
  --preferred-terms-file .\preferred_terms.txt `
  --replacement-map-json .\preferred_replacements.json `
  --host-profile-json .\host_profile.json `
  --known-speakers-dir .\speaker_reference_samples `
  --assume-dominant-speaker-is-host `
  --host-threshold 0.45 `
  --hf-token $env:HF_TOKEN
```

## Basic Workflow

For the best initial results:

1. Put one or more episodes in a source folder
2. Create `podcast_transcribe_config.json`
3. Set a valid Hugging Face token
4. On first launcher run, choose whether to provide a clean host sample
5. Run the launcher
6. Review `*_host_only.txt`, `*_review.csv`, and `_episode_review_summary.csv`

## Troubleshooting

Common first-run issues:

- Missing Hugging Face token:
  The pipeline will stop before diarization if no token is available.
- Token access errors:
  You may have a valid token but still need to accept access terms for `pyannote/speaker-diarization-community-1`.
- Launcher environment mismatch:
  If `conda activate whisper` fails, either create that environment name or update the launcher script.
- No audio files found:
  The selected input folder must contain supported audio formats directly inside it.
- Weak host labeling:
  Provide a cleaner host sample, use named reference clips, and keep `host_profile.json` between runs.
- First-run host setup was skipped:
  If you cancel the host-sample picker, the pipeline will continue by falling back to the dominant-speaker host bootstrap logic.

## Notes

- `host_profile.json` is generated during use and should usually be kept if you want host matching to improve over time.
- The review CSVs are intentionally conservative and may flag segments that are acceptable in practice.
- The repository currently focuses on local batch processing rather than a packaged application or service deployment.
