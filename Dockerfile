FROM python:3.13-slim

WORKDIR /app

# ffmpeg is required by openai-whisper to decode audio formats from the browser
# (WebM, Opus, MP4, etc.). Must be installed before the pip layer.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying code — so Docker can cache this layer
# and skip reinstalling when only your code changes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY . .

# Make the project root importable — equivalent to running with PYTHONPATH=. locally
ENV PYTHONPATH=/app

# Start the API server.
# --host 0.0.0.0 makes uvicorn listen on all interfaces, not just localhost,
# so Docker's port forwarding can reach it from outside the container.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
