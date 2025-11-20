Jarvis: Real-Time Speech-to-Speech AI

A local, privacy-focused AI voice assistant that listens, thinks, and speaks in real-time. Built with Whisper (Hearing), Ollama/Qwen (Brain), and Coqui XTTS (Voice).


 Features

Real-Time Pipeline: Threaded architecture for low-latency response.

No Cloud APIs: Runs 100% locally on your GPU.

Voice Cloning: Upload a 3-second audio clip to clone any voice.

Multilingual: Supports English, Spanish, French, German, Italian, and Portuguese.

Installation steps

1. Install Ollama from ollama.com.

2. Pull the model:

   ollama pull qwen2.5:7b


3. Install Python dependencies:

   pip install -r requirements.txt


(Note: For GPU support, ensure you have CUDA-enabled PyTorch installed)

Usage

Run the optimized pipeline:

4. python app.py        # runs the model on a user friendly graphical user interface using gradio on a web browser

or

5. terminal_app.py      # runs the model in the terminal

 Architecture

User Audio -> Whisper STT -> Ollama (LLM) -> XTTS (TTS) -> Gradio Output