.PHONY: help check-gpu build up down restart logs benchmark-vllm benchmark-audio load-test clean

help:
	@echo "=========================================================="
	@echo " RTX 3060 12GB x 2 リアルタイム音声対話システム 管理コマンド"
	@echo "=========================================================="
	@echo "  make check-gpu       : ホストの GPU 2枚と Docker GPU ランタイムを確認"
	@echo "  make build           : 全 Docker コンテナをビルド"
	@echo "  make up              : システム全体をバックグラウンドで起動"
	@echo "  make down            : システム全体を停止"
	@echo "  make restart         : システム全体を再起動"
	@echo "  make logs            : 全コンテナのリアルタイムログを表示"
	@echo "  make logs-agent      : LiveKit Agent の対話ログを表示"
	@echo "  make logs-vllm       : vLLM (GPU 0) の推論ログを表示"
	@echo "  make logs-audio      : audio-service (GPU 1) の音声ログを表示"
	@echo "  make benchmark-vllm  : [フェーズ1] GPU 0 (vLLM) 10並列 TTFT 計測"
	@echo "  make benchmark-audio : [フェーズ1] GPU 1 (STT + TTS) 10並列遅延・VRAM計測"
	@echo "  make load-test       : [フェーズ3] 10人同時接続 WebRTC 負荷テスト"
	@echo "=========================================================="

check-gpu:
	bash scripts/verify_gpu_setup.sh

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

logs-agent:
	docker compose logs -f livekit-agent

logs-vllm:
	docker compose logs -f vllm

logs-audio:
	docker compose logs -f audio-service

benchmark-vllm:
	python3 scripts/benchmark_gpu0_vllm.py

benchmark-audio:
	python3 scripts/benchmark_gpu1_audio.py

load-test:
	python3 scripts/load_test_10_clients.py

clean:
	docker compose down -v
