# 1. Base Image: Python 3.10 on Linux
FROM python:3.10-slim

# 2. Install System Tools & Audio Drivers (Needed for SoundFile/Ollama)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    git \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Ollama (The AI Brain)
RUN curl -fsSL https://ollama.com/install.sh | sh

# 4. Set up folder
WORKDIR /app

# 5. Install Python Libraries
# We copy requirements first to cache the installation layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your App code and Voices
COPY . .

# 7. Create the "Startup Script"
# This script starts Ollama in the background, downloads the model, then starts your app
RUN echo '#!/bin/bash' > start.sh && \
    echo 'echo "🔴 Starting Ollama Server..."' >> start.sh && \
    echo 'ollama serve &' >> start.sh && \
    echo 'echo "🟡 Waiting for Ollama to wake up..."' >> start.sh && \
    echo 'sleep 10' >> start.sh && \
    echo 'echo "🟢 Downloading Qwen 2.5 (This runs once)..."' >> start.sh && \
    echo 'ollama pull qwen2.5:7b' >> start.sh && \
    echo 'echo "🚀 Starting Jarvis..."' >> start.sh && \
    echo 'python app.py' >> start.sh && \
    chmod +x start.sh

# 8. Start command
CMD ["./start.sh"]