#!/usr/bin/env python3
"""
フェーズ 1: GPU 0 (vLLM + Qwen 2.5 7B Instruct AWQ) 独立ベンチマーク
同時 10 リクエストのストリーミング推論を行い、TTFT (Time To First Token) < 200ms を検証します。
"""

import sys
import time
import json
import asyncio
import statistics
from typing import List, Dict, Any

try:
    import aiohttp
except ImportError:
    print("aiohttp が必要です: pip install aiohttp")
    sys.exit(1)

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
CONCURRENT_REQUESTS = 10

TEST_PROMPTS = [
    "こんにちは、今日の天気を教えてください。",
    "日本の首都について1文で教えて。",
    "人工知能の歴史について簡単に説明して。",
    "美味しいカレーライスの作り方のコツは？",
    "WebRTCとWebSocketの違いを一言で。",
    "GPUメモリの最適化手法を教えて。",
    "音声対話AIの低遅延化で重要なポイントは？",
    "おはようございます！自己紹介をお願いします。",
    "富士山の高さは何メートルですか？",
    "リラックスできる音楽の特徴は何ですか？",
]

async def send_streaming_request(
    session: aiohttp.ClientSession,
    request_id: int,
    prompt: str,
) -> Dict[str, Any]:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "あなたは親切なAI音声アシスタントです。1文〜2文で短く回答してください。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": True,
    }

    t_start = time.perf_counter()
    ttft = None
    first_chunk_text = ""
    full_text = []
    token_count = 0

    try:
        async with session.post(VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"id": request_id, "error": f"HTTP {resp.status}: {text}"}

            async for line_bytes in resp.content:
                line = line_bytes.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        if ttft is None:
                            ttft = (time.perf_counter() - t_start) * 1000.0  # ms
                            first_chunk_text = content
                        full_text.append(content)
                        token_count += 1
                except Exception:
                    continue

        t_end = time.perf_counter()
        total_time = (t_end - t_start) * 1000.0  # ms
        tokens_per_sec = (token_count / ((t_end - t_start) or 0.001))

        return {
            "id": request_id,
            "prompt": prompt,
            "ttft_ms": ttft or total_time,
            "total_ms": total_time,
            "token_count": token_count,
            "tokens_per_sec": tokens_per_sec,
            "response": "".join(full_text).strip(),
            "error": None,
        }
    except Exception as e:
        return {"id": request_id, "error": str(e)}

async def run_benchmark():
    print("=" * 65)
    print(f"  GPU 0: vLLM (Continuous Batching) 10並列ベンチマーク")
    print(f"  URL: {VLLM_URL}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  同時リクエスト数: {CONCURRENT_REQUESTS}")
    print("=" * 65)

    # 1. ヘルスチェック
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/v1/models", timeout=5) as resp:
                if resp.status != 200:
                    print(f"❌ vLLM サーバーが正常に応答しません (HTTP {resp.status})")
                    return
                print("✅ vLLM サーバー接続成功！")
    except Exception as e:
        print(f"❌ vLLM サーバーに接続できません: {e}")
        print("  docker compose logs vllm でログを確認してください。")
        return

    print("\n🚀 10並列同時リクエストを送信中...")
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_streaming_request(session, i + 1, TEST_PROMPTS[i % len(TEST_PROMPTS)])
            for i in range(CONCURRENT_REQUESTS)
        ]
        results = await asyncio.gather(*tasks)

    # 2. 結果集計
    errors = [r for r in results if r.get("error")]
    successes = [r for r in results if not r.get("error")]

    print("\n--- 個別リクエスト結果 ---")
    print(f"{'Req ID':<8} | {'TTFT (ms)':<12} | {'Total (ms)':<12} | {'Tokens':<8} | {'Tok/s':<8} | 応答冒頭")
    print("-" * 65)
    for r in sorted(successes, key=lambda x: x["id"]):
        resp_snippet = r['response'][:22].replace("\n", " ")
        print(
            f"{r['id']:<8} | {r['ttft_ms']:<12.1f} | {r['total_ms']:<12.1f} | "
            f"{r['token_count']:<8} | {r['tokens_per_sec']:<8.1f} | {resp_snippet}..."
        )

    if errors:
        print("\n❌ 失敗リクエスト:")
        for err in errors:
            print(f"  Req {err['id']}: {err['error']}")

    if not successes:
        print("すべてのリクエストが失敗しました。")
        return

    ttft_list = [r["ttft_ms"] for r in successes]
    tps_list = [r["tokens_per_sec"] for r in successes]
    total_list = [r["total_ms"] for r in successes]

    mean_ttft = statistics.mean(ttft_list)
    p50_ttft = statistics.median(ttft_list)
    p90_ttft = sorted(ttft_list)[int(len(ttft_list) * 0.9)]
    max_ttft = max(ttft_list)
    mean_tps = statistics.mean(tps_list)

    print("\n" + "=" * 65)
    print("  📊 ベンチマーク集計サマリー (10並列同時実行)")
    print("=" * 65)
    print(f"成功リクエスト数  : {len(successes)} / {CONCURRENT_REQUESTS}")
    print(f"平均 TTFT         : {mean_ttft:.1f} ms")
    print(f"P50 TTFT          : {p50_ttft:.1f} ms")
    print(f"P90 TTFT          : {p90_ttft:.1f} ms")
    print(f"最大 TTFT         : {max_ttft:.1f} ms")
    print(f"平均スループット  : {mean_tps:.1f} tokens/sec")
    print(f"平均総レイテンシ  : {statistics.mean(total_list):.1f} ms")
    print("-" * 65)

    if p90_ttft <= 200.0:
        print("🎉 [判定: 合格] P90 TTFT が目標の 200ms 以内を達成しています！")
    else:
        print(f"⚠️ [判定: 要注意] P90 TTFT ({p90_ttft:.1f}ms) が目標の 200ms を超過しています。")
        print("  vLLM の --gpu-memory-utilization または --max-model-len を調整してください。")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
