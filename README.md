🎙️ Jarvis: Real-Time Speech-to-Speech AI



[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/rodgers.amani._/)  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rogers-ogindo-a943352a8/)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](LICENSE)



**Jarvis** is a local, privacy-focused AI voice assistant that listens, thinks, and speaks in real-time. It combines **Whisper** (Hearing), **Ollama** (Brain), and **Coqui XTTS** (Voice) into a multi-threaded pipeline for ultra-low latency performance.



- Features-  **Real-Time Pipeline:** Threaded architecture ensures immediate response.-  **No Cloud APIs:** Runs 100% locally on your GPU (Privacy focused).-  **Voice Cloning:** Clone any voice using a 3-second audio clip.-  **Multilingual:** Supports English, Spanish, French, German, Italian, and Portuguese.-  **Persistent Memory:** Automatically saves conversation logs to `conversations/`.



- Installation & Setup### 1. Prerequisites

Before you begin, ensure you have the following installed:* **Python 3.10+**: [Download Here](https://www.python.org/downloads/)* **Ollama**: [Download Here](https://ollama.com)* **FFmpeg**: Required for audio processing.    * *Windows:* `winget install ffmpeg`    * *Mac:* `brew install ffmpeg`    * *Linux:* `sudo apt install ffmpeg`### 2. Setup (Copy & Paste)



Open your terminal and run these commands one by one:```bash

 1. Clone the repository

git clone [https://github.com/Rodgers-254/realtime-speech-to-speech.git](https://github.com/Rodgers-254/realtime-speech-to-speech.git)

cd realtime-speech-to-speech



 2. Pull the AI Brain (Qwen 2.5)

ollama pull qwen2.5:7b



 3. Create a Virtual Environment (Recommended)

  Windows:

python -m venv venv

.\venv\Scripts\activate



  Mac/Linux:

  python3 -m venv venv

  source venv/bin/activate



  4. Install Dependencies

pip install -r requirements.txt

Note: If you have an NVIDIA GPU, ensure you have the CUDA version of PyTorch installed for maximum speed.

  Usage

You can run Jarvis in two modes. (Make sure your environment is active!)

Option A: Web Interface (Recommended)

Launches a modern UI in your browser with Voice Cloning features.

Bash



python app.py

Custom Voices: Drag & drop a .wav file to clone a voice instantly.

Default Voices: Place files in the Voices/ folder to have them appear in the dropdown list.

Option B: Terminal Mode

Runs the assistant directly in your command line (Push-to-talk).

Bash



python terminal_app.py

  Architecture

Code snippet



graph LR

    A[User Audio] -->|Whisper STT| B(Transcribed Text)

    B -->|Ollama LLM| C{Sentence Queue}

    C -->|Stream Sentence 1| D[XTTS Voice Engine]

    C -->|Stream Sentence 2| D

    D -->|Audio Output| E[User Heats]

Whisper (STT): Transcribes your voice to text instantly.

Ollama (LLM): Generates a smart response using Qwen 2.5.

Parallel Pipeline: As soon as the AI finishes the first sentence, it is sent to the voice engine.

Coqui XTTS (TTS): Clones the reference voice and speaks the response in real-time.

  Data & Logs

conversations/: All chat history is automatically saved here as JSON files.

Voices/: Drop your favorite voice samples here to use them as defaults.

 Feedback is welcomed.
 Lets Connect

Created by Rodgers Ogindo.
