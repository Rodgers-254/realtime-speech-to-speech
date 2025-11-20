import os
import sys
import time
import json
import torch
import numpy as np
import gradio as gr
import threading
import queue
import ollama 
from transformers import pipeline
from TTS.api import TTS
from datetime import datetime

# --- 1. Configuration & Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Accelerating on: {DEVICE.upper()}")

# Configuration
OLLAMA_MODEL = "qwen2.5:7b" 
STT_MODEL_ID = "distil-whisper/distil-medium.en" 
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
VOICE_CLONE_PATH = "Voices" # Default folder for voices
CONV_LOG_FOLDER = "conversations"

# Ensure folders exist
os.makedirs(VOICE_CLONE_PATH, exist_ok=True)
os.makedirs(CONV_LOG_FOLDER, exist_ok=True)

# Fix for Windows DLL issues
try:
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.join(sys.prefix, "Scripts"))
except:
    pass

# --- 2. Load Models (Global) ---
try:
    print("1/3 Loading Whisper (Ear)...")
    # Use Flash Attention via accelerate if available
    stt_pipeline = pipeline(
        "automatic-speech-recognition", 
        model=STT_MODEL_ID, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32, 
        device=DEVICE
    )

    print(f"2/3 Checking Ollama ({OLLAMA_MODEL})...")
    ollama.chat(model=OLLAMA_MODEL, messages=[{'role':'user','content':'ping'}])

    print("3/3 Loading XTTS (Voice)...")
    tts_model = TTS(XTTS_MODEL_NAME).to(DEVICE)
    
    print(">>> SYSTEM READY - LIGHTSPEED MODE <<<")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")
    print("Make sure Ollama is running and models are installed!")
    sys.exit(1)

# --- 3. Helper: Save Conversation ---
def save_conversation_log(history):
    """Saves the conversation history to a JSON file."""
    if not history:
        return
    
    # Simple filename based on date
    today = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(CONV_LOG_FOLDER, f"chat_log_{today}.json")
    
    # Load existing if exists to append, or create new
    data = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            data = []
            
    # Append new turns (avoid duplicates if possible, or just overwrite with full history)
    # Here we overwrite with the full current session history for simplicity
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

# --- 4. Threaded Text Producer ---
def llm_worker(messages, sentence_queue, stop_event):
    try:
        stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
        buffer = ""
        for chunk in stream:
            if stop_event.is_set(): break
            text_chunk = chunk['message']['content']
            buffer += text_chunk
            
            if any(p in text_chunk for p in [".", "!", "?", "\n", "。"]):
                clean_sentence = buffer.strip()
                if len(clean_sentence) > 2:
                    sentence_queue.put(clean_sentence)
                buffer = ""
        
        if buffer.strip():
            sentence_queue.put(buffer.strip())
            
    except Exception as e:
        print(f"LLM Error: {e}")
    finally:
        sentence_queue.put(None)

# --- 5. Main Pipeline ---
def run_fast_pipeline(user_audio_path, voice_clone_file, language, history_state):
    if user_audio_path is None:
        return None, history_state, "No audio detected."

    # A. Transcribe
    try:
        transcription = stt_pipeline(user_audio_path)
        user_text = transcription["text"].strip()
        print(f"🗣️ User: {user_text}")
        if not user_text:
            return None, history_state, "Audio was silent."
    except Exception as e:
        return None, history_state, f"Whisper Error: {e}"

    # B. Prepare Context
    messages = []
    for u, a in history_state[-2:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "system", "content": "You are a helpful voice assistant. Answer in one or two short, conversational sentences."})
    messages.append({"role": "user", "content": user_text})

    # C. Start LLM Thread
    sentence_queue = queue.Queue()
    stop_event = threading.Event()
    t = threading.Thread(target=llm_worker, args=(messages, sentence_queue, stop_event))
    t.start()

    # D. Determine Voice File
    # Priority: Uploaded File -> Default in Voices folder -> Fallback
    speaker_wav = voice_clone_file
    if not speaker_wav:
        # Check for any .wav file in Voices folder
        files = [f for f in os.listdir(VOICE_CLONE_PATH) if f.endswith('.wav')]
        if files:
            speaker_wav = os.path.join(VOICE_CLONE_PATH, files[0])
            print(f"Using default voice: {speaker_wav}")
        else:
            print("Warning: No voice found. XTTS may fail or use internal default.")

    # E. TTS Loop
    full_response = ""
    try:
        while True:
            try:
                sentence = sentence_queue.get(timeout=15)
            except queue.Empty:
                break
            if sentence is None: break
            
            print(f"🤖 AI: {sentence}")
            full_response += sentence + " "
            
            # Generate Audio
            if speaker_wav:
                wav = tts_model.tts(text=sentence, speaker_wav=speaker_wav, language=language, speed=1.1)
                yield (24000, np.array(wav, dtype=np.float32)), history_state, "Streaming..."
            else:
                # Fallback if no speaker wav provided (might fail depending on model version)
                yield None, history_state, "Error: Please upload a voice sample or add one to Voices/"

    except Exception as e:
        print(f"Streaming Error: {e}")
        yield None, history_state, f"Error: {e}"
    
    finally:
        stop_event.set()
        t.join()

    # F. Update History & Save Log
    history_state.append((user_text, full_response))
    save_conversation_log(history_state)
    yield None, history_state, "Ready."

# --- 6. Gradio UI ---
with gr.Blocks(title="Jarvis Pro", theme=gr.themes.Ocean()) as demo:
    gr.Markdown(f"## ⚡ Jarvis: Real-Time AI")
    
    state = gr.State(value=[])
    
    with gr.Row():
        with gr.Column():
            voice_in = gr.Audio(sources=["microphone"], type="filepath", label="Speak")
            
            with gr.Accordion("⚙️ Options", open=True):
                lang_dropdown = gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt"], value="en", label="Language")
                # Allow upload, but we also handle the default folder in backend
                ref_audio = gr.Audio(label="Custom Voice (Optional - defaults to 'Voices' folder)", type="filepath")

        with gr.Column():
            audio_out = gr.Audio(label="Response", streaming=True, autoplay=True)
            chatbot = gr.Chatbot(label="Transcript")
            status = gr.Textbox(label="Status", interactive=False)

    voice_in.stop_recording(
        fn=run_fast_pipeline,
        inputs=[voice_in, ref_audio, lang_dropdown, state],
        outputs=[audio_out, chatbot, status]
    )

if __name__ == "__main__":
    demo.launch()