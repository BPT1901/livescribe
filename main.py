import os
import whisper
import pyaudio
import wave
import threading
import time
from datetime import datetime
import torch
from deepmultilingualpunctuation import PunctuationModel
import re

# ===== CONFIGURATION =====
CHUNK_DURATION = 45              # seconds
OVERLAP_DURATION = 0            # seconds
SAMPLE_RATE = 16000             # Hz
CHANNELS = 1
FORMAT = pyaudio.paInt16
MODEL_NAME = "large-v3"

# HuggingFace token for speaker diarization
HF_TOKEN = ""  # Replace with your actual token

# Prompt for output filename
TRANSCRIPT_FILE = input("Enter the filename for your transcription (e.g meeting_notes.txt): ").strip()
if TRANSCRIPT_FILE == "":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TRANSCRIPT_FILE = f"transcription_{timestamp}.txt"

TEMP_AUDIO_FILE = "temp_chunk.wav"
FOCUSRITE_KEYWORD = "Focusrite"  # Keyword to identify your device

# ===== GLOBALS =====
audio_buffer = bytearray()
stop_flag = False
current_speaker = None
speaker_names = {}  # Maps speaker IDs to names
diarization_pipeline = None


def detect_focusrite_device(p):
    """
    Search all audio devices for one matching the Focusrite keyword.
    Returns the device index if found.
    """
    print("[INFO] Searching for Focusrite device...")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if FOCUSRITE_KEYWORD.lower() in info["name"].lower() and info["maxInputChannels"] > 0:
            print(f"[INFO] Found Focusrite device: {info['name']} (Index {i})")
            return i
    print("[WARN] Focusrite device not found. Using default input.")
    return None


def record_audio_loop(stream, chunk_size):
    global audio_buffer, stop_flag
    print("[INFO] Recording started. Press Ctrl+C to stop.")
    while not stop_flag:
        data = stream.read(chunk_size, exception_on_overflow=False)
        audio_buffer.extend(data)


def spinner(stop_event):
    while not stop_event.is_set():
        for ch in "|/-\\":
            print(f"\rTranscribing... {ch}", end="", flush=True)
            if stop_event.wait(0.1):
                break


def save_chunk(filename, data, sample_rate=SAMPLE_RATE):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(data)


def clean_punctuation(text, punctuation_model):
    """
    Clean up punctuation in transcribed text using deepmultilingualpunctuation.
    
    Args:
        text (str): The transcribed text to clean
        punctuation_model: Loaded PunctuationModel instance
    
    Returns:
        str: Text with improved punctuation
    """
    if not text or not text.strip():
        return text
    
    try:
        # Remove existing punctuation except for apostrophes and hyphens
        cleaned_text = ''.join(char for char in text if char.isalnum() or char.isspace() or char in "''-")
        
        # Apply punctuation model
        result = punctuation_model.restore_punctuation(cleaned_text)
        
        # Ensure first letter is capitalized
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
            
        return result
    except Exception as e:
        print(f"[WARN] Punctuation cleanup failed: {e}")
        return text


def load_diarization_pipeline():
    """
    Load the pyannote.audio speaker diarization pipeline.
    Tries offline config first, then falls back to online download.
    """
    global diarization_pipeline
    
    try:
        from pyannote.audio import Pipeline
        
        # First try to load from local offline config
        offline_config_path = "pyannote_diarization_config.yaml"
        if os.path.exists(offline_config_path):
            print("[INFO] Loading speaker diarization pipeline from local config...")
            diarization_pipeline = Pipeline.from_pretrained(offline_config_path)
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            diarization_pipeline.to(device)
            
            print("[INFO] Speaker diarization pipeline loaded successfully (offline mode)")
            return diarization_pipeline
        
        # Fallback to online download
        hf_token = os.environ.get('HF_TOKEN') or HF_TOKEN
        if not hf_token or hf_token == "your_token_here":
            print("[INFO] No offline config found and HuggingFace token not set.")
            print("[INFO] Speaker diarization disabled.")
            print("[INFO] For offline setup: create pyannote_diarization_config.yaml")
            print("[INFO] For online setup: export HF_TOKEN=your_token_here")
            return None
        
        print("[INFO] Loading speaker diarization pipeline from HuggingFace...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diarization_pipeline.to(device)
        
        print("[INFO] Speaker diarization pipeline loaded successfully (online mode)")
        return diarization_pipeline
        
    except Exception as e:
        print(f"[WARN] Failed to load diarization pipeline: {e}")
        print("[INFO] Speaker diarization disabled")
        return None


def get_speaker_for_chunk(audio_file, start_time, end_time):
    """
    Get the dominant speaker for a specific audio chunk.
    
    Args:
        audio_file (str): Path to the audio file
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
    
    Returns:
        str: Speaker ID or None if diarization unavailable
    """
    global diarization_pipeline
    
    if not diarization_pipeline:
        return None
    
    try:
        # Run diarization on the chunk
        diarization = diarization_pipeline(audio_file)
        
        # Find the dominant speaker in the time range
        speaker_time = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Check if this segment overlaps with our chunk
            overlap_start = max(turn.start, start_time)
            overlap_end = min(turn.end, end_time)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                speaker_time[speaker] = speaker_time.get(speaker, 0) + overlap_duration
        
        # Return the speaker with the most time in this chunk
        if speaker_time:
            return max(speaker_time, key=speaker_time.get)
        
        return None
        
    except Exception as e:
        print(f"[WARN] Speaker diarization failed: {e}")
        return None


def extract_speaker_name(text):
    """
    Extract speaker name from text when they introduce themselves.
    
    Args:
        text (str): The transcribed text
    
    Returns:
        str: Extracted name or None
    """
    if not text:
        return None
    
    # Common introduction patterns
    patterns = [
        r"(?:my name is|i'm|i am|this is)\s+([a-zA-Z]+)",
        r"([a-zA-Z]+)\s+speaking",
        r"([a-zA-Z]+)\s+here",
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            name = match.group(1).capitalize()
            # Filter out common words that aren't names
            if name not in ['The', 'This', 'That', 'Here', 'There', 'Now', 'Then']:
                return name
    
    return None


def format_speaker_text(text, speaker_id):
    """
    Format text with speaker label if speaker changed.
    
    Args:
        text (str): The transcribed text
        speaker_id (str): Current speaker ID
    
    Returns:
        str: Formatted text with speaker label if needed
    """
    global current_speaker, speaker_names
    
    if not speaker_id:
        return text
    
    # Check if speaker changed
    speaker_changed = current_speaker != speaker_id
    current_speaker = speaker_id
    
    # Try to extract name from text
    extracted_name = extract_speaker_name(text)
    if extracted_name:
        speaker_names[speaker_id] = extracted_name
    
    # Format output
    if speaker_changed:
        speaker_label = speaker_names.get(speaker_id, f"Speaker {speaker_id}")
        return f"[{speaker_label}]: {text}"
    
    return text


def main():
    global stop_flag, audio_buffer

    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Torch device detected: {device}")

    print(f"[INFO] Loading Whisper model '{MODEL_NAME}'...")
    model = whisper.load_model(MODEL_NAME, device=device)
    
    print("[INFO] Loading punctuation model...")
    punctuation_model = PunctuationModel()
    
    # Load speaker diarization pipeline
    load_diarization_pipeline()

    # Set up audio stream
    p = pyaudio.PyAudio()
    focusrite_index = detect_focusrite_device(p)

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=focusrite_index,
        frames_per_buffer=int(SAMPLE_RATE * CHUNK_DURATION)
    )

    # Start audio recording thread
    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
    record_thread = threading.Thread(target=record_audio_loop, args=(stream, chunk_size))
    record_thread.start()

    # Processing loop
    try:
        chunk_stride = int(SAMPLE_RATE * (CHUNK_DURATION - OVERLAP_DURATION))
        transcript = []
        chunk_start_time = 0

        while not stop_flag:
            if len(audio_buffer) >= chunk_size:
                current_chunk = audio_buffer[:chunk_size]
                save_chunk(TEMP_AUDIO_FILE, current_chunk)

                # Start spinner
                stop_event = threading.Event()
                t = threading.Thread(target=spinner, args=(stop_event,))
                t.start()

                # Run transcription
                result = model.transcribe(
                    TEMP_AUDIO_FILE,
                    language='en',
                    temperature=0,
                    beam_size=5
                )

                # Stop spinner
                stop_event.set()
                t.join()
                print("\rTranscribing... done!        ")

                text = result['text'].strip()

                if text:
                    # Clean up punctuation using deepmultilingualpunctuation
                    text = clean_punctuation(text, punctuation_model)
                    
                    # Get speaker information for this chunk
                    speaker_id = get_speaker_for_chunk(TEMP_AUDIO_FILE, 0, CHUNK_DURATION)
                    
                    # Format text with speaker label if needed
                    text = format_speaker_text(text, speaker_id)

                if transcript:
                    last_text = transcript[-1].strip().lower()
                    # Remove speaker labels for overlap detection
                    last_text_clean = re.sub(r'^\[.*?\]:\s*', '', last_text)
                    text_clean = re.sub(r'^\[.*?\]:\s*', '', text.lower())
                
                    if last_text_clean:
                        overlap_len = min(30, len(last_text_clean))
                        overlap = last_text_clean[-overlap_len:]

                        if overlap and text_clean.startswith(overlap):
                            # Preserve speaker label if present
                            speaker_match = re.match(r'(\[.*?\]:\s*)', text)
                            if speaker_match:
                                text = speaker_match.group(1) + text_clean[len(overlap):].lstrip()
                            else:
                                text = text_clean[len(overlap):].lstrip()
                
                print(text)
                transcript.append(text)

                # Slide buffer forward
                audio_buffer = audio_buffer[chunk_stride:]
                chunk_start_time += (CHUNK_DURATION - OVERLAP_DURATION)
            else:
                time.sleep(0.25)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        stop_flag = True

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        if os.path.exists(TEMP_AUDIO_FILE):
            os.remove(TEMP_AUDIO_FILE)

        with open(TRANSCRIPT_FILE, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(transcript))
        print(f"[INFO] Transcription saved to {TRANSCRIPT_FILE}")


if __name__ == '__main__':
    main()
