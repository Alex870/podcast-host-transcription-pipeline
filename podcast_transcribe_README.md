# Podcast Host Transcription Pipeline

This revision adds:

- transcription with `faster-whisper`
- speaker diarization with `pyannote.audio`
- host detection using either:
  - a reference host voice clip, or
  - a persisted host embedding profile, or
  - dominant-speaker bootstrapping
- full speaker-labeled transcripts
- host-only transcripts
- review CSVs for risky host matches and glossary corrections
- preferred terminology biasing and post-correction
- optional named speaker identification from a reference-sample directory

## Files

- `podcast_transcribe_host.py`: core Python pipeline
- `Convert MP3 to TXT diarized.ps1`: folder-based PowerShell launcher
- `preferred_terms.txt`: terms to bias during decoding
- `preferred_replacements.json`: likely mistranscriptions to normalize after decoding
- `host_profile.json`: auto-created persistent host voice profile
- `speaker_reference_samples/speakers.json`: template config for named speaker reference clips
- `podcast_transcribe_config.json`: launcher defaults for folders, model settings, and matching thresholds

## Setup

Install dependencies:

```powershell
pip install -r podcast_transcribe_requirements.txt
```

Set a Hugging Face token in the shell before running. The token must have access to `pyannote/speaker-diarization-community-1`, and you must accept that model's user conditions on Hugging Face first.

```powershell
$env:HF_TOKEN = "your_token_here"
```

## Config File

The PowerShell launcher now reads `podcast_transcribe_config.json` from the same directory as the scripts.

You can use it to store default paths and runtime settings instead of editing the script directly.

Example:

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

Parameters:

- `default_source_dir`: default starting folder for the source-audio selection dialog
- `hf_token`: Hugging Face token used for pyannote diarization model access
- `known_speakers_dir`: directory containing `speakers.json` and named voice samples
- `preferred_terms_file`: glossary file used to bias transcription
- `replacement_map_json`: post-correction map for common mistranscriptions
- `host_profile_json`: persistent learned host voice profile
- `model`: `faster-whisper` model name
- `language`: language code for transcription
- `compute_type`: `faster-whisper` compute type such as `auto`, `float16`, `int8`, etc.
- `beam_size`: beam size for decoding
- `batch_size`: transcription batch size
- `assume_dominant_speaker_is_host`: whether to fall back to the dominant speaker when no host match is found
- `host_threshold`: similarity threshold used for speaker matching

Path handling:

- Absolute paths are used as-is
- Relative paths are resolved relative to the script directory
- If `default_source_dir` exists, the folder picker opens there first

Token handling:

- The PowerShell launcher first looks for `HF_TOKEN` in the environment
- If `HF_TOKEN` is not set, it uses `hf_token` from `podcast_transcribe_config.json`
- If no token is available, the launcher stops and shows a message before running Python
- If the token is invalid or lacks access to `pyannote/speaker-diarization-community-1`, the Python helper raises a clearer authentication error and the launcher shows a follow-up warning

## Best results

- Provide a short clean host voice sample with minimal guest overlap.
- Keep `host_profile.json` between runs so host labeling stays more stable from episode to episode.
- Add show-specific terms to `preferred_terms.txt`.
- Add likely mistakes to `preferred_replacements.json`.

## Adding a Host Voice Sample

There are now two supported ways to help the system recognize the host:

1. One-off host sample in the launcher dialog:
   - Run `Convert MP3 to TXT diarized.ps1`
   - After you choose the source folder, select a clean `.mp3`, `.wav`, or similar clip of the host speaking alone
   - The script will compare diarized speakers in each episode against that sample

2. Persistent named reference sample directory:
   - Put one or more clean clips for the host in `speaker_reference_samples`
   - Edit `speaker_reference_samples/speakers.json`
   - Mark the host entry with `"is_host": true`

Example:

```json
{
  "speakers": [
    {
      "name": "HOST",
      "is_host": true,
      "files": ["host_sample.wav", "host_sample_2.wav"]
    }
  ]
}
```

Using the reference directory is the better long-term option because it can label recurring speakers by name, not just the host.

## Identifying Other Known Speakers

Yes, this is now supported in a feasible way.

Use the `speaker_reference_samples` directory with a `speakers.json` file and one or more sample clips per person:

```json
{
  "speakers": [
    {
      "name": "HOST",
      "is_host": true,
      "files": ["host_sample.wav"]
    },
    {
      "name": "Alice",
      "files": ["alice_clip_1.wav", "alice_clip_2.wav"]
    },
    {
      "name": "Bob",
      "files": ["bob_clip.wav"]
    }
  ]
}
```

Notes:

- The file paths are relative to `speaker_reference_samples/`
- Each clip should be mostly one speaker with minimal overlap or background audio
- Multiple clips per person are supported; the code averages their embeddings
- Known speakers are matched one-to-one against diarized speakers when similarity clears the configured threshold
- If a known speaker is marked as host, that label overrides weaker host guessing

## Recommended Workflow

For the most reliable setup:

- Use `speaker_reference_samples/speakers.json` for recurring speakers
- Mark the host with `"is_host": true`
- Keep `host_profile.json` enabled as an additional evolving host fingerprint
- Use the one-off host sample dialog only as a quick bootstrap or when you do not yet have the reference directory set up

This is better than relying only on “dominant speaker = host”, which can drift on interview-heavy or guest-heavy episodes.

## Outputs per audio file

- `*_speaker_transcript.txt`
- `*_host_only.txt`
- `*_review.csv`
- `*_speaker_transcript.json`
- `_episode_review_summary.csv`

## Practical note

Host-only filtering is only as good as diarization plus host matching. A clean reference sample is much more reliable than dominant-speaker guessing.

The review CSV is meant for fast spot checks. It flags:

- weak or ambiguous host matches
- low host coverage episodes
- transcript segments where glossary aliases were normalized

The folder summary CSV gives one row per episode so you can sort the batch by:

- low host similarity
- low host speech share
- ambiguous host matches
- episodes with the most review flags

It now also includes `review_priority_score` and `review_priority_reason`, and writes the rows in descending priority order so the riskiest episodes appear first. The score is driven by missing/weak host detection, low host speech share, ambiguous speaker matches, and accumulated review flags, while the reason column gives a plain-English summary of the biggest drivers.
