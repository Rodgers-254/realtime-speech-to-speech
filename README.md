🎙️ Jarvis: Real-Time Speech-to-Speech AI

A local, privacy-focused AI voice assistant that listens, thinks, and speaks in real-time. Built with Whisper (Hearing), Ollama/Qwen (Brain), and Coqui XTTS (Voice).

🚀 Features

Real-Time Pipeline: Threaded architecture ensures low-latency responses.

No Cloud APIs: Runs 100% locally on your PC (Privacy focused).

Voice Cloning: Clone any voice using a 3-second audio clip.

Multilingual: Supports English, Spanish, French, German, Italian, and Portuguese.

Persistent Memory: Automatically saves conversation logs to conversations/.

⚡ Quick Start (Copy & Paste)

1. Prerequisites

Install Ollama: Download from ollama.com.

Install FFmpeg: Required for audio processing.

Windows: winget install ffmpeg

Mac: brew install ffmpeg

Linux: sudo apt install ffmpeg

2. Installation & Setup

Copy and paste the commands below into your terminal to set up everything at once.

🪟 Windows (PowerShell)

# 1. Pull the AI Model
ollama pull qwen2.5:7b # or model of your choice

# 2. Create Virtual Environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install Dependencies (This may take a few minutes)
pip install -r requirements.txt

# Note: If you have an NVIDIA GPU, install PyTorch with CUDA support for speed.


🍎 Mac / 🐧 Linux

# 1. Pull the AI Model
ollama pull qwen2.5:7b

# 2. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt


💻 Usage

Once installed, you can run Jarvis in two modes. (Make sure your environment is active!)

Option A: Web Interface (Recommended)

Launches a beautiful UI in your browser.

python app.py


Custom Voices: Drag & drop a short .wav file to clone a voice.

Default Voice: Place .wav files in the Voices/ folder to have them appear in the dropdown list automatically.

Option B: Terminal Mode

Runs the assistant directly in your command line.

python terminal_app.py


🏗️ Architecture

User Audio -> Whisper (STT) -> Ollama (LLM) -> Sentence Queue -> XTTS (TTS) -> Audio Output


Whisper (STT): Transcribes your voice to text instantly.

Ollama (LLM): Generates a smart response using Qwen 2.5.

Parallel Pipeline: As soon as the AI finishes the first sentence, it is sent to the voice engine.

Coqui XTTS (TTS): Clones the reference voice and speaks the response in real-time.

📂 Data & Logs

conversations/: All chat history is automatically saved here as JSON files.

Voices/: Drop your favorite voice samples here to use them as defaults.
