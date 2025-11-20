# 1. Use Python 3.10 Base Image
FROM python:3.10-slim

# 2. Install system dependencies (ffmpeg is required for whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Ollama (Linux version)
RUN curl -fsSL https://ollama.com/install.sh | sh

# 4. Set Working Directory
WORKDIR /app

# 5. Copy Requirements and Install Python Libs
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the Application Code
COPY . .

# 7. Create a start script to handle Ollama background process
RUN echo '#!/bin/bash' > start.sh && \
    echo 'ollama serve &' >> start.sh && \
    echo 'echo "Waiting for Ollama..."' >> start.sh && \
    echo 'sleep 10' >> start.sh && \
    echo 'ollama pull qwen2.5:7b' >> start.sh && \
    echo 'python app.py' >> start.sh && \
    chmod +x start.sh

# 8. Start the application
CMD ["./start.sh"]