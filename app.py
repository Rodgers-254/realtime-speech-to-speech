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
VOICE_CLONE_PATH = "Voices" 
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
    
    print(">>> SYSTEM READY <<<")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")
    sys.exit(1)

# --- 3. Helpers ---
def get_voice_list():
    """Scans the Voices folder for .wav files"""
    voices = [f for f in os.listdir(VOICE_CLONE_PATH) if f.lower().endswith('.wav') or f.lower().endswith('.mp3')]
    if not voices:
        return ["No voices found in 'Voices' folder"]
    return sorted(voices)

def save_conversation_log(history):
    if not history: return
    today = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(CONV_LOG_FOLDER, f"chat_log_{today}.json")
    
    # Load existing or create new
    existing_data = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except: pass
    
    # We only append the LAST turn to avoid duplicating the whole history every time
    last_turn = history[-1]
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "user": last_turn[0],
        "ai": last_turn[1]
    }
    existing_data.append(entry)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

def float_to_int16(audio_float):
    """
    CRITICAL FIX: Converts float32 audio to int16.
    This prevents the 'h11' protocol error and the Gradio UserWarning.
    """
    audio_float = np.clip(audio_float, -1, 1)
    return (audio_float * 32767).astype(np.int16)

# --- 4. Logic ---
def llm_worker(messages, sentence_queue, stop_event):
    """Generates text in background"""
    try:
        stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
        buffer = ""
        for chunk in stream:
            if stop_event.is_set(): break
            text_chunk = chunk['message']['content']
            buffer += text_chunk
            
            # Split by sentence endings
            if any(p in text_chunk for p in [".", "!", "?", "\n"]):
                clean_sentence = buffer.strip()
                if len(clean_sentence) > 1:
                    sentence_queue.put(clean_sentence)
                buffer = ""
        
        if buffer.strip():
            sentence_queue.put(buffer.strip())
            
    except Exception as e:
        print(f"LLM Error: {e}")
    finally:
        sentence_queue.put(None) # End signal

def run_pipeline(mic_audio, upload_audio, voice_dropdown, voice_upload, language, history_state):
    # 1. Determine Input Source (Mic takes priority if both are present, or just whichever is not None)
    # Note: Gradio might pass None for the inactive tab, so check both
    user_audio = mic_audio if mic_audio else upload_audio
    
    if user_audio is None:
        yield None, history_state, "⚠️ No audio input detected! Please speak or upload a file."
        return

    # 2. Transcribe
    try:
        yield None, history_state, "👂 Listening..."
        transcription = stt_pipeline(user_audio)
        user_text = transcription["text"].strip()
        if not user_text:
            yield None, history_state, "⚠️ Audio was silent."
            return
    except Exception as e:
        yield None, history_state, f"❌ STT Error: {e}"
        return

    # 3. Prepare Chat History
    messages = []
    for entry in history_state[-2:]: # Keep context short
        messages.append({"role": "user", "content": entry[0]})
        messages.append({"role": "assistant", "content": entry[1]})
    
    messages.append({"role": "system", "content": "You are a helpful voice assistant. Keep answers concise and conversational."})
    messages.append({"role": "user", "content": user_text})

    # 4. Determine Voice Reference
    # Priority: Uploaded File > Dropdown Selection > First file in folder
    ref_wav = None
    
    # Check upload first
    if voice_upload:
        ref_wav = voice_upload
        print(f"Using uploaded voice clone: {ref_wav}")
    # Check dropdown second
    elif voice_dropdown and voice_dropdown != "No voices found in 'Voices' folder":
        ref_wav = os.path.join(VOICE_CLONE_PATH, voice_dropdown)
        print(f"Using selected voice: {ref_wav}")
    
    # Fallback check
    if not ref_wav or not os.path.exists(ref_wav):
        # Try to find *any* wav in the folder
        defaults = get_voice_list()
        if defaults and defaults[0] != "No voices found in 'Voices' folder":
            ref_wav = os.path.join(VOICE_CLONE_PATH, defaults[0])
            yield None, history_state, f"⚠️ No voice selected. Using default: {defaults[0]}"
        else:
             yield None, history_state, "❌ Error: No voice found in Voices/ folder or uploaded!"
             return

    # 5. Start Threads
    sentence_queue = queue.Queue()
    stop_event = threading.Event()
    t = threading.Thread(target=llm_worker, args=(messages, sentence_queue, stop_event))
    t.start()

    # 6. Generate & Stream Audio
    full_response = ""
    yield None, history_state + [[user_text, "..."]], "🧠 Thinking..."

    try:
        while True:
            try:
                sentence = sentence_queue.get(timeout=10)
            except queue.Empty:
                break
            
            if sentence is None: break
            
            full_response += sentence + " "
            
            # Update Chat UI immediately
            current_history = history_state + [[user_text, full_response]]
            yield None, current_history, f"🗣️ Speaking: {sentence}"
            
            # Generate Audio
            wav_chunks = tts_model.tts(
                text=sentence, 
                speaker_wav=ref_wav, 
                language=language,
                speed=1.1
            )
            
            # FIX: Convert to int16 manually
            wav_int16 = float_to_int16(np.array(wav_chunks, dtype=np.float32))
            
            # Yield Audio chunk (24000 Hz is standard for XTTS)
            yield (24000, wav_int16), current_history, "🟢 Streaming..."

    except Exception as e:
        print(f"Stream Error: {e}")
        yield None, history_state, f"Error: {e}"
    
    finally:
        stop_event.set()
        t.join()
        # Final Save
        history_state.append([user_text, full_response.strip()])
        save_conversation_log(history_state)
        yield None, history_state, "✅ Ready"


# --- 5. Gradio UI Layout ---
theme = gr.themes.Soft(
    primary_hue="cyan",
    neutral_hue="slate",
)

# CSS to hide the footer for a cleaner look
css = """
footer {visibility: hidden}
.container {max_width: 1200px; margin: auto; padding-top: 20px}
"""

with gr.Blocks(title="Jarvis STS", theme=theme, css=css) as demo:
    
    # Memory State
    history = gr.State(value=[])

    with gr.Row(elem_classes="container"):
        gr.Markdown("""
        # 🎙️ Jarvis: Real-Time Speech-to-Speech
        *Local AI. Zero Latency. Total Privacy.*
        """)
    
    with gr.Row(elem_classes="container"):
        
        # --- LEFT COLUMN (Inputs) ---
        with gr.Column(scale=1):
            
            gr.Markdown("### 1. Your Input")
            # Tabs for Input Method
            with gr.Tabs():
                with gr.TabItem("🎙️ Microphone"):
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Query")
                
                with gr.TabItem("📁 Upload Audio"):
                    upload_input = gr.Audio(sources=["upload"], type="filepath", label="Upload Query File")
                    process_btn = gr.Button("🚀 Process Uploaded File", variant="primary")

            gr.Markdown("### 2. Voice Settings")
            with gr.Group():
                lang_input = gr.Dropdown(
                    choices=["en", "es", "fr", "de", "it", "pt"], 
                    value="en", 
                    label="Language"
                )
                
                # Dropdown for default voices
                with gr.Row():
                    voice_dropdown = gr.Dropdown(
                        choices=get_voice_list(), 
                        value=get_voice_list()[0] if get_voice_list() else None,
                        label="Select Existing Voice", 
                        interactive=True,
                        scale=3
                    )
                    refresh_btn = gr.Button("🔄", scale=0)
                
                # Upload override
                voice_upload = gr.Audio(
                    label="Or Clone New Voice (Overrides selection)", 
                    type="filepath",
                    sources=["upload"]
                )

        # --- RIGHT COLUMN (Outputs) ---
        with gr.Column(scale=1):
            gr.Markdown("### 3. Response")
            
            # Status Bar
            status_box = gr.Textbox(label="System Status", value="Ready", interactive=False)
            
            # Audio Player (Streaming)
            audio_output = gr.Audio(
                label="AI Voice", 
                autoplay=True, 
                streaming=True, 
                type="numpy"
            )
            
            # Chat History
            chatbot = gr.Chatbot(label="Conversation Log", height=400, bubble_full_width=False)

    # --- Events ---
    
    # Function to refresh voice dropdown
    def update_voices():
        new_choices = get_voice_list()
        return gr.Dropdown(choices=new_choices, value=new_choices[0] if new_choices else None)

    refresh_btn.click(fn=update_voices, outputs=voice_dropdown)

    # Logic for Mic Input (Auto-submit on stop recording)
    mic_input.stop_recording(
        fn=run_pipeline,
        inputs=[mic_input, upload_input, voice_dropdown, voice_upload, lang_input, history],
        outputs=[audio_output, chatbot, status_box]
    )

    # Logic for File Upload (Button Click)
    process_btn.click(
        fn=run_pipeline,
        inputs=[mic_input, upload_input, voice_dropdown, voice_upload, lang_input, history],
        outputs=[audio_output, chatbot, status_box]
    )

if __name__ == "__main__":
    # queue() is required for streaming to work reliably
    demo.queue().launch()