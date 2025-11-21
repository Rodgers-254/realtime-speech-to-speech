import os
import sys
import time
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import keyboard
import ollama
import colorama
from colorama import Fore, Style
from threading import Thread, Event
from queue import Queue

# --- FIX: Ensure DLLs are found ---
try:
    scripts_folder = os.path.join(sys.prefix, "Scripts")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(scripts_folder)
except Exception:
    pass

from transformers import pipeline
from TTS.api import TTS

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OLLAMA_MODEL = "qwen2.5:7b"                                #set model here
STT_MODEL_ID = "distil-whisper/distil-medium.en"
SAMPLE_RATE = 16000 
TTS_SAMPLE_RATE = 24000 
VOICE_CLONE_PATH = "Voices/female2.wav"                               #set your clone voice here
CONVO_FOLDER = "conversations"
os.makedirs(CONVO_FOLDER, exist_ok=True) 

# --- SETUP ---
colorama.init(autoreset=True)
audio_queue = Queue()
playback_finished = Event()
interrupt_event = Event()

def print_status(text):
    print(f"{Fore.CYAN}[SYSTEM] {text}{Style.RESET_ALL}")

def print_user(text):
    print(f"\n{Fore.GREEN}[YOU]: {text}{Style.RESET_ALL}")

def print_ai(text):
    sys.stdout.write(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")
    sys.stdout.flush()

# --- 1. LOAD MODELS ---
print_status(f"Loading models on {DEVICE}...")

try:
    stt_pipeline = pipeline("automatic-speech-recognition", model=STT_MODEL_ID, torch_dtype=torch.float16, device=DEVICE)
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
    
    if os.path.exists(VOICE_CLONE_PATH):
        print_status(f"✅ Voice Clone found: {VOICE_CLONE_PATH}")
    else:
        print_status(f"⚠️ Voice Clone file '{VOICE_CLONE_PATH}' NOT FOUND. Using default voice.")
        VOICE_CLONE_PATH = None

    ollama.chat(model=OLLAMA_MODEL, messages=[{'role':'user','content':'hi'}])
    print_status(">>> READY - PRESS SPACE TO TALK <<<")

except Exception as e:
    print(f"{Fore.RED}CRITICAL LOAD ERROR: {e}")
    exit()

# --- 2. WORKER THREADS ---

def audio_player_worker():
    """Runs in background to play audio as it arrives in the queue."""
    while True:
        chunk_path = audio_queue.get()
        if chunk_path is None: # Sentinel to stop
            break
        
        # If interrupted, clear queue and skip
        if interrupt_event.is_set():
            audio_queue.task_done()
            continue

        try:
            data, _ = sf.read(chunk_path, dtype='float32')
            sd.play(data, TTS_SAMPLE_RATE)
            sd.wait()
        except Exception:
            pass
        finally:
            audio_queue.task_done()

# --- 3. HELPER FUNCTIONS ---

def record_audio():
    """Records audio while SPACE is held."""
    print_status("Hold [SPACE] to speak...")
    audio_data = []
    
    keyboard.wait('space')
    print(f"{Fore.RED}🔴 Recording... (Release SPACE to stop){Style.RESET_ALL}")
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while keyboard.is_pressed('space'):
            chunk, overflow = stream.read(1024)
            audio_data.append(chunk)
    
    print_status("Processing...")
    return np.concatenate(audio_data, axis=0)

# --- 4. MAIN LOOP ---

def main():
    # Start the background player
    player_thread = Thread(target=audio_player_worker, daemon=True)
    player_thread.start()

    history = []
    
    print(f"\n{Fore.MAGENTA}=== INSTRUCTIONS ==={Style.RESET_ALL}")
    print("1. Hold SPACE to talk.")
    print("2. Release SPACE to send.")
    print("3. Press SPACE while AI is talking to INTERRUPT.")
    print("4. Press ESC to quit.\n")

    while True:
        # --- 0. CLEANUP STATE ---
        interrupt_event.clear()
        sd.stop() 
        
        # FIXED: Robust release check
        # Wait up to 1 second for user to release space, then force reset
        start_wait = time.time()
        while keyboard.is_pressed('space'):
            time.sleep(0.05)
            if time.time() - start_wait > 1.0:
                break # Force break if stuck

        # Flush keyboard events to prevent "ghost" presses
        try:
            while keyboard.read_event(suppress=True).event_type == keyboard.KEY_DOWN:
                pass
        except:
            pass

        if keyboard.is_pressed('esc'):
            print("Exiting...")
            break
            
        # --- 1. RECORD ---
        user_audio = record_audio()
        
        if len(user_audio) < SAMPLE_RATE * 0.5:
            print_status("Audio too short, ignored.")
            continue

        timestamp = int(time.time())
        user_filename = os.path.join(CONVO_FOLDER, f"user_{timestamp}.wav")
        sf.write(user_filename, user_audio, SAMPLE_RATE)

        # --- 2. TRANSCRIBE ---
        try:
            transcription = stt_pipeline(user_filename)["text"].strip()
            print_user(transcription)
        except Exception as e:
            print(f"STT Error: {e}")
            continue

        if not transcription:
            continue

        # --- 3. STREAMING RESPONSE ---
        print(f"\n{Fore.YELLOW}[AI]: ", end="")
        
        messages = []
        for u, a in history[-2:]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": transcription})

        stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
        
        sentence_buffer = ""
        full_response = ""
        chunk_count = 0

        for chunk in stream:
            # Check Interruption
            if keyboard.is_pressed('space'):
                interrupt_event.set()
                sd.stop() # Kill current audio
                # Clear pending audio
                while not audio_queue.empty():
                    try: audio_queue.get_nowait()
                    except: pass
                print(f"\n{Fore.RED}[INTERRUPTED]{Style.RESET_ALL}")
                break

            token = chunk['message']['content']
            print_ai(token)
            
            sentence_buffer += token
            full_response += token

            if any(p in token for p in [".", "!", "?", "\n"]):
                clean_sentence = sentence_buffer.strip()
                if len(clean_sentence) > 2:
                    try:
                        chunk_count += 1
                        ai_filename = os.path.join(CONVO_FOLDER, f"ai_{timestamp}_part{chunk_count}.wav")
                        
                        # Generate to file
                        tts_model.tts_to_file(
                            text=clean_sentence, 
                            speaker_wav=VOICE_CLONE_PATH, 
                            language="en", 
                            file_path=ai_filename
                        )
                        
                        # PUSH TO QUEUE (Background player handles it)
                        # This allows the GPU to immediately start generating the NEXT sentence
                        # while the player is still reading this one.
                        audio_queue.put(ai_filename)
                            
                    except Exception as e:
                        pass 
                
                sentence_buffer = ""

        # Wait for audio queue to finish playing before allowing next recording
        # (Unless we interrupted, then we skip waiting)
        if not interrupt_event.is_set():
            audio_queue.join() 

        history.append((transcription, full_response))
        print("\n")

if __name__ == "__main__":
    main()