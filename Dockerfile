FROM python:3.10-slim

# Install system dependencies required for OpenCV and MediaPipe on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend files
COPY ./frameforensics-backend/ /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-app.txt

EXPOSE 7860

CMD ["python", "app/app.py"]
