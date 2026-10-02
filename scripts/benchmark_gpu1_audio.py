#!/usr/bin/env python3
"""
フェーズ 1: GPU 1 (faster-whisper + Kokoro-82M) 独立ベンチマーク
同時 10 並列バッチ処理時の推論遅延、RTF (Real-Time Factor)、VRAM使用量を計測します。
"""

import sys
import io
import time
import wave
import json
import asyncio
import statistics
from typing import List, Dict, Any

try:
    import aiohttp
    import numpy as np
except ImportError:
    print("aiohttp, numpy が必要です: pip install aiohttp numpy")
    sys.exit(1)

AUDIO_SERVICE_URL = "http://localhost:8001"
CONCURRENT_REQUESTS = 10

TEST_SENTENCES = [
    "こんにちは！今日はどのようなご用件でしょうか？",
    "承知いたしました。ただいま確認しておりますので少々お待ちください。",
    "最新の天気予報によりますと、明日は晴れ時々曇りの見込みです。",
    "音声対話AIの応答速度を改善するために、句読点単位でストリーミング送出を行っています。",
    "GPUを役割ごとに物理分離することで、メモリ競合と遅延を防ぎます。",
    "何か他にお手伝いできることはございますでしょうか？",
    "富士山の標高は3776メートルで、日本最高峰の独立峰です。",
    "WebRTCプロトコルを採用することで、低ジッターかつ超低遅延な通話が可能です。",
    "ユーザーが話し始めた瞬間に、エージェントの発話を即座に中断します。",
    "本日はご利用いただき誠にありがとうございました。またお声がけください。",
]

async def check_health(session: aiohttp.ClientSession) -> Dict[str, Any]:
    async with session.get(f"{AUDIO_SERVICE_URL}/health", timeout=5) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Healthcheck failed: HTTP {resp.status}")
        return await resp.json()

async def benchmark_single_tts(
    session: aiohttp.ClientSession,
    req_id: int,
    text: str,
) -> Dict[str, Any]:
    payload = {
        "model": "hexgrad/Kokoro-82M",
        "input": text,
        "voice": "jf_alpha",
        "response_format": "wav",
        "speed": 1.0,
    }

    t0 = time.perf_counter()
    async with session.post(f"{AUDIO_SERVICE_URL}/v1/audio/speech", json=payload, timeout=20) as resp:
        if resp.status != 200:
            err = await resp.text()
            return {"id": req_id, "error": f"HTTP {resp.status}: {err}"}
        audio_bytes = await resp.read()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Parse WAV duration
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration_sec = frames / float(rate)
    except Exception:
        duration_sec = 1.0

    rtf = (elapsed_ms / 1000.0) / max(duration_sec, 0.001)

    return {
        "id": req_id,
        "text": text,
        "elapsed_ms": elapsed_ms,
        "duration_sec": duration_sec,
        "rtf": rtf,
        "audio_bytes": audio_bytes,
        "error": None,
    }

async def benchmark_single_stt(
    session: aiohttp.ClientSession,
    req_id: int,
    audio_bytes: bytes,
    expected_text: str,
) -> Dict[str, Any]:
    data = aiohttp.FormData()
    data.add_field(
        "file",
        audio_bytes,
        filename="test.wav",
        content_type="audio/wav",
    )
    data.add_field("model", "Systran/faster-whisper-medium")
    data.add_field("language", "ja")

    t0 = time.perf_counter()
    async with session.post(f"{AUDIO_SERVICE_URL}/v1/audio/transcriptions", data=data, timeout=20) as resp:
        if resp.status != 200:
            err = await resp.text()
            return {"id": req_id, "error": f"HTTP {resp.status}: {err}"}
        res_json = await resp.json()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    recognized_text = res_json.get("text", "")
    return {
        "id": req_id,
        "elapsed_ms": elapsed_ms,
        "recognized_text": recognized_text,
        "expected_text": expected_text,
        "error": None,
    }

async def run_benchmark():
    print("=" * 70)
    print("  GPU 1: faster-whisper + Kokoro-82M 10並列ベンチマーク")
    print(f"  URL: {AUDIO_SERVICE_URL}")
    print(f"  同時リクエスト数: {CONCURRENT_REQUESTS}")
    print("=" * 70)

    async with aiohttp.ClientSession() as session:
        # 1. 接続確認 & 初期 VRAM
        try:
            initial_stats = await check_health(session)
            print(f"✅ Audio Service 接続成功 (Device: {initial_stats.get('device')})")
            if "vram_allocated_mb" in initial_stats:
                print(f"  初期 VRAM 使用量: {initial_stats['vram_allocated_mb']} MB (Reserved: {initial_stats['vram_reserved_mb']} MB)")
        except Exception as e:
            print(f"❌ Audio Service に接続できません: {e}")
            print("  docker compose logs audio-service でログを確認してください。")
            return

        # 2. Kokoro-82M TTS 10並列ベンチマーク
        print("\n🔊 [1/2] Kokoro-82M (TTS) 10並列同時合成テスト実行中...")
        tts_tasks = [
            benchmark_single_tts(session, i + 1, TEST_SENTENCES[i % len(TEST_SENTENCES)])
            for i in range(CONCURRENT_REQUESTS)
        ]
        tts_results = await asyncio.gather(*tts_tasks)

        tts_success = [r for r in tts_results if not r.get("error")]
        if not tts_success:
            print("❌ TTS リクエストが全て失敗しました。")
            return

        print(f"{'Req ID':<8} | {'遅延 (ms)':<10} | {'音声長 (s)':<10} | {'RTF (推論/音声比)':<18} | 合成テキスト")
        print("-" * 70)
        for r in sorted(tts_success, key=lambda x: x["id"]):
            print(
                f"{r['id']:<8} | {r['elapsed_ms']:<10.1f} | {r['duration_sec']:<10.2f} | "
                f"{r['rtf']:<18.3f} | {r['text'][:18]}..."
            )

        mean_tts_ms = statistics.mean([r["elapsed_ms"] for r in tts_success])
        mean_rtf = statistics.mean([r["rtf"] for r in tts_success])

        # 3. faster-whisper STT 10並列ベンチマーク (TTSで合成したWAVを使用)
        print("\n🎤 [2/2] faster-whisper (STT) 10並列同時文字起こしテスト実行中...")
        stt_tasks = [
            benchmark_single_stt(session, r["id"], r["audio_bytes"], r["text"])
            for r in tts_success
        ]
        stt_results = await asyncio.gather(*stt_tasks)

        stt_success = [r for r in stt_results if not r.get("error")]
        print(f"{'Req ID':<8} | {'遅延 (ms)':<10} | 認識結果")
        print("-" * 70)
        for r in sorted(stt_success, key=lambda x: x["id"]):
            print(f"{r['id']:<8} | {r['elapsed_ms']:<10.1f} | {r['recognized_text'][:30]}...")

        mean_stt_ms = statistics.mean([r["elapsed_ms"] for r in stt_success])

        # 4. 実行後 VRAM 確認
        final_stats = await check_health(session)

    # 5. 集計レポート
    print("\n" + "=" * 70)
    print("  📊 GPU 1 音声 I/O ベンチマーク集計サマリー")
    print("=" * 70)
    print(f"TTS 10並列 平均遅延 : {mean_tts_ms:.1f} ms")
    print(f"TTS 平均 RTF         : {mean_rtf:.3f} (※0.1以下で極めて優秀)")
    print(f"STT 10並列 平均遅延 : {mean_stt_ms:.1f} ms")
    if "vram_allocated_mb" in final_stats:
        print(f"GPU 1 VRAM 使用量    : {final_stats['vram_allocated_mb']} MB (推定設計: 2500〜3500MB)")
        print(f"GPU 1 VRAM 予約量    : {final_stats['vram_reserved_mb']} MB")
    print("-" * 70)

    if mean_rtf < 0.2:
        print("🎉 [判定: 合格] Kokoro-82M は10並列時でも実時間より圧倒的に高速(RTF < 0.2)です！")
    else:
        print(f"⚠️ [判定: 注意] RTF ({mean_rtf:.3f}) がやや高めです。GPU負荷を確認してください。")

    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
