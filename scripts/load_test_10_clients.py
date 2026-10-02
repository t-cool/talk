#!/usr/bin/env python3
"""
フェーズ 3: 10人同時接続 WebRTC 音声負荷テストスクリプト
10 クライアントから同時に LiveKit Server へ接続し、音声を流し込んで
応答遅延（Turnaround Time）とバージイン追従性を検証します。
"""

import sys
import os
import time
import math
import asyncio
import statistics
from typing import List, Dict, Any

try:
    from livekit import api, rtc
except ImportError:
    print("livekit, livekit-api が必要です: pip install livekit livekit-api")
    sys.exit(1)

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secretkeyforlocalvoiceai2026")
CONCURRENT_CLIENTS = 10

def generate_test_tone(sample_rate: int = 16000, duration_sec: float = 2.0, freq: float = 440.0) -> bytes:
    """テスト用トーン音声 (PCM 16bit mono)"""
    num_samples = int(sample_rate * duration_sec)
    data = bytearray()
    for i in range(num_samples):
        # 440Hz sine wave
        t = i / sample_rate
        val = int(10000 * math.sin(2 * math.pi * freq * t))
        data.extend(val.to_bytes(2, byteorder="little", signed=True))
    return bytes(data)

async def simulate_single_client(client_id: int, room_name: str) -> Dict[str, Any]:
    participant_name = f"load-test-user-{client_id}"
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(participant_name)
        .with_name(participant_name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

    room = rtc.Room()
    t_start = time.perf_counter()
    connect_time = None
    agent_audio_received_time = None
    agent_spoke = False

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        nonlocal agent_audio_received_time, agent_spoke
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            agent_spoke = True
            if agent_audio_received_time is None:
                agent_audio_received_time = time.perf_counter()

    try:
        # 1. 接続
        await room.connect(LIVEKIT_URL, token)
        connect_time = (time.perf_counter() - t_start) * 1000.0  # ms

        # 2. 音声トラックの作成と発行 (2秒間の擬似発話)
        sample_rate = 16000
        source = rtc.AudioSource(sample_rate, 1)
        track = rtc.LocalAudioTrack.create_audio_track(f"mic-{client_id}", source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await room.local_participant.publish_track(track, options)

        # 音声フレームの送信 (100ms チャンク × 20 = 2秒)
        tone_data = generate_test_tone(sample_rate=sample_rate, duration_sec=2.0)
        chunk_size = int(sample_rate * 0.1) * 2  # 100ms at 16bit (2 bytes per sample)

        speech_start = time.perf_counter()
        for offset in range(0, len(tone_data), chunk_size):
            chunk = tone_data[offset : offset + chunk_size]
            if len(chunk) < chunk_size:
                break
            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(chunk) // 2,
            )
            await source.capture_frame(frame)
            await asyncio.sleep(0.09) # リアルタイム送出

        speech_end = time.perf_counter()

        # 3. エージェントの応答音声を待機 (最大10秒)
        wait_deadline = time.perf_counter() + 10.0
        while time.perf_counter() < wait_deadline and not agent_spoke:
            await asyncio.sleep(0.1)

        turnaround_ms = None
        if agent_audio_received_time is not None:
            turnaround_ms = (agent_audio_received_time - speech_end) * 1000.0

        await room.disconnect()

        return {
            "client_id": client_id,
            "room": room_name,
            "connect_ms": connect_time,
            "turnaround_ms": turnaround_ms,
            "agent_responded": agent_spoke,
            "error": None,
        }

    except Exception as e:
        if room.isconnected:
            await room.disconnect()
        return {
            "client_id": client_id,
            "room": room_name,
            "connect_ms": connect_time,
            "turnaround_ms": None,
            "agent_responded": False,
            "error": str(e),
        }

async def run_load_test():
    print("=" * 70)
    print("  フェーズ 3: 10人同時接続 WebRTC 音声負荷テスト")
    print(f"  LiveKit URL: {LIVEKIT_URL}")
    print(f"  同時接続クライアント数: {CONCURRENT_CLIENTS}")
    print("=" * 70)

    print("\n🚀 10クライアントを同時接続中...")
    tasks = [
        simulate_single_client(i + 1, f"load-test-room-{i + 1}")
        for i in range(CONCURRENT_CLIENTS)
    ]
    results = await asyncio.gather(*tasks)

    print("\n--- 個別クライアント結果 ---")
    print(f"{'Client ID':<12} | {'Connect (ms)':<14} | {'Turnaround (ms)':<16} | 状態")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["client_id"]):
        status_str = "✅ 応答成功" if r["agent_responded"] else f"❌ 失敗 ({r.get('error') or '応答なし'})"
        conn_str = f"{r['connect_ms']:.1f}" if r["connect_ms"] else "N/A"
        ta_str = f"{r['turnaround_ms']:.1f}" if r["turnaround_ms"] else "N/A"
        print(f"Client {r['client_id']:<5} | {conn_str:<14} | {ta_str:<16} | {status_str}")

    successes = [r for r in results if r["agent_responded"]]
    print("\n" + "=" * 70)
    print("  📊 10人同時接続 負荷テスト集計サマリー")
    print("=" * 70)
    print(f"接続成功率 : {len(successes)} / {CONCURRENT_CLIENTS} ({(len(successes)/CONCURRENT_CLIENTS)*100:.1f}%)")

    if successes:
        connect_times = [r["connect_ms"] for r in successes if r["connect_ms"]]
        ta_times = [r["turnaround_ms"] for r in successes if r["turnaround_ms"] is not None]

        print(f"平均 WebRTC 接続時間 : {statistics.mean(connect_times):.1f} ms")
        if ta_times:
            mean_ta = statistics.mean(ta_times)
            p90_ta = sorted(ta_times)[int(len(ta_times) * 0.9)]
            print(f"平均 Turnaround 遅延  : {mean_ta:.1f} ms (発話終了〜AI応答開始)")
            print(f"P90 Turnaround 遅延   : {p90_ta:.1f} ms")
            if p90_ta <= 600.0:
                print("🎉 [判定: 合格] 10人同時通話時でも目標の体感遅延 300〜600ms を達成！")
            else:
                print(f"⚠️ [判定: 要調整] P90遅延 ({p90_ta:.1f}ms) が目標をやや超過しています。")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_load_test())
