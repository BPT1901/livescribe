import os
import warnings

import whisper
import pyaudio
import wave
import threading
import time
from datetime import datetime
import torch
import re
import sys
import select

# ===== CONFIGURATION =====
CHUNK_DURATION = 45              # seconds
OVERLAP_DURATION = 2             # seconds
SAMPLE_RATE = 16000              # Hz
CHANNELS = 1
FORMAT = pyaudio.paInt16
BYTES_PER_SAMPLE = pyaudio.get_sample_size(FORMAT)  # 2 for paInt16
MODEL_NAME = "large-v3"

# Default output filename (will be set in main function)
TRANSCRIPT_FILE = ""

TEMP_AUDIO_FILE = "temp_chunk.wav"
FOCUSRITE_KEYWORD = "Focusrite"  # Keyword to identify your device

# ===== GLOBALS =====
audio_buffer = bytearray()
stop_flag = False


def check_quit_input():
    """Check for 'q' input in a non-blocking way."""
    global stop_flag
    while not stop_flag:
        try:
            if sys.platform == "win32":
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    if key == 'q':
                        stop_flag = True
                        break
            else:
                # Unix-like systems
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    if key == 'q':
                        stop_flag = True
                        break
        except:
            pass
        time.sleep(0.1)


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
    print("[INFO] Recording started. Press 'q' to stop.")
    while not stop_flag:
        data = stream.read(1024, exception_on_overflow=False)
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


def remove_overlapping_text(previous_text, new_text, min_overlap_words=4):
    """
    Remove overlapping words from the start of new_text that appear at the end of previous_text.
    Uses word-level matching with punctuation normalization so minor differences like
    Oxford commas or capitalisation don't prevent detection.

    Args:
        previous_text (str): The previously transcribed text
        new_text (str): The new text to check for overlaps
        min_overlap_words (int): Minimum word count to qualify as an overlap

    Returns:
        str: new_text with the overlapping prefix removed
    """
    if not previous_text or not new_text:
        return new_text

    def to_words(text):
        """Lowercase, strip punctuation, return word list."""
        return re.sub(r'[^\w\s]', '', text.lower()).split()

    prev_words = to_words(previous_text)
    new_words  = to_words(new_text)

    if not prev_words or not new_words:
        return new_text

    # Find the longest N where the last N words of prev equal the first N words of new
    best_overlap = 0
    max_check = min(len(prev_words), len(new_words), 50)

    for n in range(max_check, min_overlap_words - 1, -1):
        if prev_words[-n:] == new_words[:n]:
            best_overlap = n
            break

    if best_overlap == 0:
        return new_text

    # Remove exactly best_overlap content words from the start of new_text,
    # preserving original punctuation and capitalisation in the remainder.
    tokens = new_text.strip().split()
    content_seen = 0
    skip_until = len(tokens)  # default: skip everything (full overlap)

    for i, token in enumerate(tokens):
        if re.sub(r'[^\w]', '', token):   # token contains at least one word character
            content_seen += 1
        if content_seen >= best_overlap:
            skip_until = i + 1
            break

    remaining = ' '.join(tokens[skip_until:]).strip()

    # Avoid returning empty — keep original text if the whole chunk was an overlap
    if not remaining:
        return new_text

    return remaining[0].upper() + remaining[1:]


def main():
    global stop_flag, audio_buffer, TRANSCRIPT_FILE

    # Suppress FP16 warnings from Whisper
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

    # Prompt for output filename
    try:
        filename_input = input("Enter the filename for your transcription (e.g. meeting_notes): ").strip()
        if filename_input == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            TRANSCRIPT_FILE = f"transcription_{timestamp}.txt"
        else:
            TRANSCRIPT_FILE = filename_input if filename_input.endswith('.txt') else f"{filename_input}.txt"
    except EOFError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        TRANSCRIPT_FILE = f"transcription_{timestamp}.txt"
        print(f"[INFO] Using default filename: {TRANSCRIPT_FILE}")

    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Torch device detected: {device}")

    print(f"[INFO] Loading Whisper model '{MODEL_NAME}'...")
    model = whisper.load_model(MODEL_NAME, device=device)

    # Set up audio stream
    p = pyaudio.PyAudio()
    focusrite_index = detect_focusrite_device(p)

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=focusrite_index,
        frames_per_buffer=1024
    )

    # Start audio recording thread
    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION * BYTES_PER_SAMPLE)
    record_thread = threading.Thread(target=record_audio_loop, args=(stream, chunk_size))
    record_thread.start()

    # Start quit input monitoring thread
    quit_thread = threading.Thread(target=check_quit_input)
    quit_thread.daemon = True
    quit_thread.start()

    # Processing loop
    try:
        chunk_stride = int(SAMPLE_RATE * (CHUNK_DURATION - OVERLAP_DURATION) * BYTES_PER_SAMPLE)
        transcript = []

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

                # Filter known Whisper hallucinations on silence/noise
                if text.lower().startswith('transcription by'):
                    text = ''

                if transcript and text:
                    text = remove_overlapping_text(transcript[-1], text)

                if text:
                    print(text)
                    transcript.append(text)

                # Slide buffer forward
                audio_buffer = audio_buffer[chunk_stride:]
            else:
                time.sleep(0.25)

        # If we get here, stop_flag was set by 'q' press
        print("\n[INFO] 'q' pressed, stopping...")

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        stop_flag = True

    finally:
        # Wait for recording thread to finish
        record_thread.join(timeout=2.0)

        # Ensure cleanup happens
        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except:
            pass

        if os.path.exists(TEMP_AUDIO_FILE):
            try:
                os.remove(TEMP_AUDIO_FILE)
            except:
                pass

        # Save transcript
        try:
            with open(TRANSCRIPT_FILE, 'w', encoding='utf-8') as f:
                f.write("Transcription by ClearwaveTX\n\n")
                f.write("\n\n".join(transcript))
            print(f"[INFO] Transcription saved to {TRANSCRIPT_FILE}")
        except Exception as e:
            print(f"[WARN] Could not save transcript: {e}")

        print("[INFO] Cleanup complete.")


if __name__ == '__main__':
    main()
