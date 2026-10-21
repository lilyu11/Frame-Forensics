FROM python:3.10-slim

# Install system dependencies and build tools (bao gồm libgl1 và libegl1 cho MediaPipe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY frameforensics-backend/requirements-app.txt /app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-app.txt

COPY frameforensics-backend/ /app/

EXPOSE 7860

CMD ["python", "app/app.py"]