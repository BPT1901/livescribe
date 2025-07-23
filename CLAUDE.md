# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a live transcription application that records audio from a Focusrite audio interface and provides real-time speech-to-text transcription using OpenAI's Whisper model. The application includes speaker diarization, punctuation enhancement, and outputs transcripts to text files.

## Key Commands

**Installation and Setup:**
```bash
pip install -r requirements.txt
```

**Run the application:**
```bash
python main.py
```

**Stop recording:**
- Press `Ctrl+C` to stop recording and save transcript

## Architecture Overview

The application is built as a single-file Python script (`main.py`) with the following key components:

1. **Audio Processing Pipeline:**
   - Records audio in configurable chunks (default 45 seconds) from Focusrite device
   - Uses PyAudio for real-time audio capture
   - Processes audio in overlapping chunks to avoid word cutoff

2. **Transcription System:**
   - Whisper large-v3 model for speech-to-text
   - DeepMultilingualPunctuation for punctuation restoration
   - Configurable temperature and beam size for quality control

3. **Speaker Diarization:**
   - Optional pyannote.audio integration for speaker identification
   - Automatic speaker name extraction from introductions
   - Falls back gracefully when HuggingFace token unavailable

4. **Output Management:**
   - Real-time console output with spinner animation
   - Automatic overlap detection and removal between chunks
   - UTF-8 encoded text file output

## Configuration

Key parameters in `main.py`:
- `CHUNK_DURATION`: Audio processing chunk size (45 seconds)
- `OVERLAP_DURATION`: Overlap between chunks (0 seconds)
- `MODEL_NAME`: Whisper model version ("large-v3")
- `SAMPLE_RATE`: Audio sample rate (16000 Hz)

## Dependencies

Core dependencies:
- `openai-whisper`: Speech-to-text transcription
- `pyaudio`: Real-time audio capture
- `torch`: Machine learning framework
- `deepmultilingualpunctuation`: Text punctuation enhancement
- `pyannote.audio`: Speaker diarization (optional)

## Audio Device Detection

The application automatically detects Focusrite audio interfaces by searching device names for "Focusrite" keyword. Falls back to system default if not found.

## Speaker Diarization Setup

Two modes supported:
1. **Offline**: Place `pyannote_diarization_config.yaml` in project root
2. **Online**: Set `HF_TOKEN` environment variable or modify token in code

Application continues without speaker diarization if neither is available.