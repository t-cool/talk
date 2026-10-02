#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  RTX 3060 12GB x 2 GPU 環境確認スクリプト"
echo "=========================================================="

# 1. nvidia-smi の確認
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ エラー: nvidia-smi が見つかりません。NVIDIA GPU ドライバをインストールしてください。"
    exit 1
fi

echo "✅ NVIDIA ドライバ検出成功:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# 2. GPU枚数の検証
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
echo ""
echo "検出された GPU 数: $GPU_COUNT"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "⚠️ 警告: 検出された GPU が 2 枚未満です ($GPU_COUNT 枚)。"
    echo "本システムは GPU 0 (LLM) と GPU 1 (音声I/O) の物理分離を前提としています。"
else
    echo "✅ 2枚以上の GPU が正常に認識されています！"
    echo "  - GPU 0: vLLM (Qwen 2.5 7B-Instruct AWQ) に割り当て"
    echo "  - GPU 1: faster-whisper + Kokoro-82M に割り当て"
fi

# 3. Docker & NVIDIA Container Toolkit の確認
echo ""
echo "--- Docker & NVIDIA Container Toolkit 確認 ---"
if ! command -v docker &> /dev/null; then
    echo "❌ エラー: docker が見つかりません。"
    exit 1
fi

echo "Docker Version: $(docker --version)"

echo "Docker からの GPU 認識テスト実行中..."
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✅ Docker から NVIDIA GPU へのアクセスが正常に機能しています！"
else
    echo "❌ 警告: Docker から GPU にアクセスできません。"
    echo "NVIDIA Container Toolkit (nvidia-docker2) の設定を確認してください:"
    echo "  sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo nvidia-ctk runtime configure --runtime=docker"
    echo "  sudo systemctl restart docker"
fi

echo ""
echo "確認完了！"
