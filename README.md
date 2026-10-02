# RTX 3060 12GB x 2 リアルタイム音声対話システム 構築メモ

RTX 3060 12GB を2枚載せたLinuxマシンで、10人同時接続できるリアルタイム音声対話AIを動かすための構成メモ。Docker Compose で一括起動できるようにしてある。

Moshi などの Speech-to-Speech モデルは単一セッションでGPUを占有してしまい多人数接続で破綻するため、LiveKit Agents + faster-whisper + vLLM + Kokoro TTS によるカスケード方式を採用した。GPU 0 を言語モデル専用、GPU 1 を音声処理専用に物理分離して割り当てることで、12GB VRAM の制約下でもキュー詰まりや OOM を起こさずに安定して回す。

---

## 1. アーキテクチャとリソース配分

2枚の GPU を Tensor Parallel で束ねると通信オーバーヘッドとコンテキスト競合が発生するため、役割ごとに物理分離している。

```
[ クライアント (最大10人: PCブラウザ / スマホ) ]
         │ (WebRTC: 音声通話)
         ▼
[ LiveKit Server / LiveKit Agents (CPU / ホストOS) ]
         │  - Silero VAD (発話検知 & 200msで割り込み中断)
         │  - JapaneseSentenceTokenizer (句読点ごとにTTSへ送出)
         │
         ├──▶ 音声入力ストリーム (10人分) ──┐
         │                                   │
         │   ┌───────────────────────────────┘
         │   ▼
         │ [ GPU 1: 音声 I/O 専用 (12GB) ]
         │   - STT: faster-whisper (medium: 約1.5GB)
         │   - TTS: Kokoro-82M 日本語 (約0.5GB)
         │   │ (文字起こしテキスト)
         │   ▼
         │ [ GPU 0: LLM 専用 (12GB) ]
         │   - エンジン: vLLM (Continuous Batching)
         │   - モデル: Qwen 2.5 7B-Instruct (AWQ 4bit: 約4.5GB + KV Cache 約6.0GB)
         │   │ (生成テキストをストリーミング送出)
         │   ▼
         │ [ GPU 1: Kokoro TTS へパイプ ]
         │   │ (音声チャンクを即座に生成)
         │   ▼
         └──◀ 音声出力ストリーム (10人分) ──┘
```

### VRAM 配分設計

| デバイス | 担当機能 | 稼働プロセス・モデル | VRAM目安 | 設計とチューニングのメモ |
| :--- | :--- | :--- | :--- | :--- |
| GPU 0 (12GB) | LLM 推論専用 | vLLM + Qwen 2.5 7B-Instruct (AWQ 4bit) | 約 11.0 GB | モデル本体は約4.5GB。残りの約6.5GBをKVキャッシュに充てる。Continuous Batching で10人がバラバラに話してもメモリ切れを起こさない。 |
| GPU 1 (12GB) | 音声 I/O 専用 | faster-whisper + Kokoro-82M (FastAPI) | 約 2.5〜3.5 GB | faster-whisper (medium) に約1.5GB、Kokoro-82M に約0.5GB。残りの約8〜9GBは発話が重なったときのバッファとして保持。 |
| CPU / ホスト | 通信・制御 | LiveKit SFU + Python Agent Worker | メモリ 4〜8 GB | WebRTC の暗号化・パケット中継、Silero VAD（発話検出・割り込み判定）を処理。 |

---

## 2. ディレクトリ構成

```
.
├── docker-compose.yml          # 全5コンテナの定義（GPU 0/1 を個別に割り当て）
├── Makefile                    # 起動、ログ確認、テスト用コマンド集
├── .env.example / .env         # 接続先やモデルの設定
├── config/
│   └── livekit.yaml            # LiveKit Server 設定（ポート、APIキー）
├── services/
│   ├── audio-service/          # [GPU 1] faster-whisper + Kokoro-82M（FastAPI）
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── engine.py           # 音声認識・合成の推論処理
│   │   └── app.py              # OpenAI互換エンドポイント（/transcriptions, /speech）
│   ├── livekit-agent/          # [CPU] 対話を統括する LiveKit エージェント
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── agent.py            # パイプライン制御（VAD・バージイン・プロンプト）
│   │   └── japanese_tokenizer.py # 句読点（、。！？）単位でテキストを区切るトークナイザー
│   └── web-client/             # [Web画面] ブラウザからすぐ試せるUI
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── server.py           # トークン発行サーバー
│       └── static/             # 音声通話画面（波形表示付き）
└── scripts/
    ├── verify_gpu_setup.sh     # GPU 2枚と Docker の認識テスト
    ├── benchmark_gpu0_vllm.py  # [フェーズ1] LLM 単体の10並列ベンチマーク
    ├── benchmark_gpu1_audio.py # [フェーズ1] 音声処理単体の10並列ベンチマーク
    └── load_test_10_clients.py # [フェーズ3] WebRTC 10人同時接続テスト
```

---

## 3. RTX 3060 マシンでの準備

### 前提環境
- OS: Ubuntu 22.04 LTS または 24.04 LTS
- GPU: NVIDIA GeForce RTX 3060 12GB x 2枚
- ドライバ: NVIDIA Driver 535 以上（550 推奨）
- ツール: Docker, Docker Compose (v2.20+), NVIDIA Container Toolkit

NVIDIA Container Toolkit の導入:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

GPU 認識の確認:
```bash
make check-gpu
# または: bash scripts/verify_gpu_setup.sh
```

---

## 4. 検証・起動手順

### フェーズ 1: 単体ベンチマーク

各レイヤー単体で 10 並列リクエストを投げて性能を計測する。

#### 1. GPU 0: vLLM (Qwen 2.5 7B) のテスト
```bash
# vLLM コンテナのみ起動
docker compose up -d vllm

# モデルの初回DLとロード完了を待つ（初回は数分）
docker compose logs -f vllm
# ログに「Application startup complete」が出たら抜ける

# 10並列ベンチマーク実行
make benchmark-vllm
```
確認項目: 10並列同時リクエスト時の P90 TTFT が 200ms 以内に収まること。

#### 2. GPU 1: faster-whisper & Kokoro TTS のテスト
```bash
# audio-service コンテナを起動
docker compose up -d audio-service

# ログでモデルのロード完了を確認
docker compose logs -f audio-service

# 10並列音声ベンチマーク実行
make benchmark-audio
```
確認項目:
- Kokoro-82M の音声合成速度（RTF: 実時間比）が 0.1 以下（発話時間の10倍以上の速度で生成できること）。
- GPU 1 の VRAM 使用量が 2.5〜3.5 GB に収まること。

---

### フェーズ 2: 全体の起動

すべてのコンテナを一括起動する。

```bash
make up
```

起動するコンテナ:
1. voice_livekit_server : WebRTC 通信基盤（ポート 7880, 7881, 7882）
2. voice_vllm_gpu0 : GPU 0 の LLM 推論サーバー（ポート 8000）
3. voice_audio_gpu1 : GPU 1 の 音声認識・合成サーバー（ポート 8001）
4. voice_livekit_agent : CPU 上で会話の流れを制御するワーカー
5. voice_web_client : ブラウザ用通話UI（ポート 3000）

ログ確認:
```bash
make logs
# エージェントの会話ログのみ確認する場合
make logs-agent
```

---

### フェーズ 3: 接続テスト & 負荷テスト

#### 1. ブラウザから通話テスト
ブラウザでアクセスする:
```
http://<サーバーのIPアドレス>:3000
```
(ローカルマシンのブラウザなら http://localhost:3000)

- 「会話を開始する」を押してマイクを許可すると、AIから挨拶が返ってくる。
- 話しかけると、句読点（「、」「。」）単位でストリーミング音声生成され、発話終了から約 300〜500ms で返答が始まる。
- 割り込み（バージイン）のテスト: AIが話している途中に声を被せると、約200msで発話が中断されてこちらの発話の聞き取りに切り替わる。
- 複数タブやスマホから同時に開くことで複数人同時通話のテストも可能。

#### 2. 10人同時接続の WebRTC 負荷テスト
スクリプトから 10 本の WebRTC 接続を張り、擬似音声を流し込んで負荷テストを行う。
```bash
make load-test
```
確認項目:
- 10本すべての接続がエラーなく成立すること。
- 発話終了から応答開始までの体感遅延（Turnaround Time）が 300〜600ms に収まること。
- GPU 0 / GPU 1 で OOM が発生しないこと。

---

## 5. チューニングのメモ

### vLLM 起動オプション (GPU 0)
docker-compose.yml で以下のパラメータを設定している。
- --gpu-memory-utilization 0.90: 12GB中、約10.8GBをvLLMに確保。
- --max-model-len 2048: 音声対話では長大な文脈は不要なため、2048トークンに制限して10人分のKVキャッシュを安全に保持する。
- --max-num-seqs 16: 10人が一斉に発話してリクエストが重なった場合でも、キュー落ちせずさばけるように16系列まで並行処理を許可。

### 句読点ストリーミング (japanese_tokenizer.py)
LLM が全文を生成し終えるのを待つと2〜3秒の沈黙が発生するため、LLMのストリーミング出力から「。」「！」「？」「\n」が出た瞬間（または12文字以上の「、」）にテキストを分割し、即座に Kokoro TTS に投入する。これによって発話終了後 300〜500ms で最初の音声が出始める。

### 200ms バージイン制御
LiveKit の Silero VAD を使い、AIの発話中であってもユーザーの音声を 200ms 検知した段階で再生中パケットを破棄（Flush）し、実行中の LLM / TTS 生成タスクを即座にキャンセル（abort）する。

---

## 6. ハマりどころ・備忘録

### スマホや別PCから開くとマイクが動かない問題
ブラウザのセキュリティ仕様により、http://localhost 以外では HTTPS でないとマイク使用がブロックされる。
- 対処法1: Chrome のフラグ設定（chrome://flags/#unsafely-treat-insecure-origin-as-secure）に対象URL（例: http://192.168.1.100:3000）を登録して例外許可する。
- 対処法2: Nginx 等で自己署名証明書を入れて HTTPS 化する。
- WebRTC ポート（7880/tcp, 7881/tcp, 7882/udp, 50000-50020/udp）のファイヤーウォール開放も必要。

### GPU 1 の VRAM をさらに削りたい場合
STT モデルを一段階小さくする。
services/audio-service/app.py または .env の WHISPER_MODEL を Systran/faster-whisper-medium（約1.5GB）から Systran/faster-whisper-small（約0.9GB）に変更すれば約0.6GB浮く。日本語認識精度も small で十分実用水準。

### 停止とリセット
```bash
# コンテナ停止
make down

# キャッシュやボリュームも含めて全削除
make clean
```
