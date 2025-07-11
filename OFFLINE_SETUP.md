# Offline Setup Guide

This guide explains how to set up the Live Transcriber to run completely offline after initial downloads.

## Overview

After initial setup, all components can run offline:
- **Whisper**: Models cached to `~/.cache/whisper/`
- **Punctuation Model**: Cached to `~/.cache/huggingface/hub/`
- **Speaker Diarization**: Uses local config file and models

## Initial Setup (Requires Internet)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Whisper Models
Run the application once to download and cache Whisper models:
```bash
python main.py
```
The first run will download the selected model (e.g., `large-v3` ~3GB) to `~/.cache/whisper/`.

### 3. Download Punctuation Model
The punctuation model downloads automatically on first use and caches to `~/.cache/huggingface/hub/`.

### 4. Setup Speaker Diarization (Optional)

#### Option A: Online Setup (Recommended)
1. Create HuggingFace account and get access token
2. Accept user conditions for required models
3. Set token: `export HF_TOKEN=your_token_here`
4. Run once to download models to cache

#### Option B: Manual Offline Setup
1. Create `models/` directory
2. Download required models:
   - `wespeaker-voxceleb-resnet34-LM/pytorch_model.bin`
   - `segmentation-3.0/pytorch_model.bin`
3. Rename models:
   - `pyannote_model_wespeaker-voxceleb-resnet34-LM.bin`
   - `pyannote_model_segmentation-3.0.bin`
4. Create `pyannote_diarization_config.yaml`:
```yaml
version: 3.1.0
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    embedding: models/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin
    segmentation: models/pyannote_model_segmentation-3.0.bin
```

## Offline Operation

### Environment Variables for Offline Mode
```bash
# Force offline mode for transformers
export HF_HUB_OFFLINE=1

# Optional: Set cache directories
export TRANSFORMERS_CACHE=~/.cache/huggingface/hub
export TIKTOKEN_CACHE_DIR=~/.cache/tiktoken
```

### Running Offline
```bash
python main.py
```

The application will:
1. Load Whisper from local cache
2. Load punctuation model from local cache
3. Use local speaker diarization config (if available)
4. Run without internet connectivity

## Verification

### Test Offline Mode
1. Disconnect from internet
2. Run: `python main.py`
3. Verify all models load successfully
4. Check for any "downloading" messages (should be none)

### Cache Locations
- **Whisper**: `~/.cache/whisper/`
- **Punctuation**: `~/.cache/huggingface/hub/`
- **Speaker Diarization**: `pyannote_diarization_config.yaml` (local)

## Troubleshooting

### Common Issues

1. **Missing Whisper Model Cache**
   - Re-run online once to download models
   - Check `~/.cache/whisper/` for model files

2. **Punctuation Model Not Found**
   - Set `HF_HUB_OFFLINE=1` environment variable
   - Ensure `~/.cache/huggingface/hub/` exists with model files

3. **Speaker Diarization Fails**
   - Check `pyannote_diarization_config.yaml` exists
   - Verify model files in `models/` directory
   - Ensure correct file naming with "pyannote" prefix

### Air-Gapped Systems
For completely disconnected systems:
1. Run setup on internet-connected machine
2. Copy entire cache directories:
   - `~/.cache/whisper/`
   - `~/.cache/huggingface/`
   - `pyannote_diarization_config.yaml`
   - `models/` directory
3. Transfer to offline machine
4. Set environment variables for offline mode

## File Structure
```
livescribe/
├── main.py
├── requirements.txt
├── OFFLINE_SETUP.md
├── pyannote_diarization_config.yaml  # Optional
├── models/                           # Optional
│   ├── pyannote_model_wespeaker-voxceleb-resnet34-LM.bin
│   └── pyannote_model_segmentation-3.0.bin
└── ~/.cache/                        # System cache
    ├── whisper/
    ├── huggingface/hub/
    └── tiktoken/
```