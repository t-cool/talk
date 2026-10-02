# GPU 1 専用 Audio Service (faster-whisper + Kokoro-82M)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# システム依存パッケージのインストール (ffmpeg, espeak-ng, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY engine.py app.py ./

EXPOSE 8000

# 起動コマンド
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
