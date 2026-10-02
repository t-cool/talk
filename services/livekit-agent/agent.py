import os
import logging
import asyncio
from dotenv import load_dotenv

from livekit.agents import (
    AgentServer,
    WorkerOptions,
    JobContext,
    JobProcess,
    AgentSession,
    Agent,
    tts,
    cli,
)
from livekit.plugins import openai, silero
from japanese_tokenizer import JapaneseSentenceTokenizer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("livekit-voice-agent")

# システムプロンプト: 音声対話向けに最適化（短文・口語・記号排除）
INSTRUCTIONS = """
あなたは親切でフレンドリーな音声対話AIアシスタントです。
WebRTCによるリアルタイム音声対話を行っています。
以下のルールを厳格に守って応答してください：
1. 回答は短く端的に、1〜2文程度の自然な話し言葉（口語）で返してください。
2. マークダウン記号（**太字**、# 見出し、箇条書きなど）や絵文字は読み上げの邪魔になるため絶対に出力しないでください。
3. 句読点（「、」「。」「！」「？」）を適切に入れて、自然な区切りで発話してください。
4. ユーザーが話しかけやすいように、親身で親しみやすいトーンを保ってください。
"""

async def entrypoint(ctx: JobContext):
    """
    1セッションごとの音声対話パイプライン
    - VAD: Silero VAD (CPU)
    - STT: faster-whisper (GPU 1 / audio-service)
    - LLM: Qwen 2.5 7B-Instruct AWQ (GPU 0 / vLLM)
    - TTS: Kokoro-82M (GPU 1 / audio-service) + JapaneseSentenceTokenizer
    """
    logger.info(f"Agent worker starting session for room: {ctx.room.name}")
    await ctx.connect()

    vllm_base_url = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
    audio_base_url = os.getenv("AUDIO_BASE_URL", "http://audio-service:8000/v1")
    llm_model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
    whisper_model = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-medium")
    kokoro_voice = os.getenv("KOKORO_VOICE", "jf_alpha")

    # 1. Voice Activity Detection (Silero VAD)
    vad = silero.VAD.load()

    # 2. STT: faster-whisper on GPU 1
    stt = openai.STT(
        base_url=audio_base_url,
        api_key="none",
        model=whisper_model,
        language="ja",
    )

    # 3. LLM: vLLM (Qwen 2.5 7B AWQ) on GPU 0
    llm = openai.LLM(
        base_url=vllm_base_url,
        api_key="none",
        model=llm_model,
        temperature=0.7,
    )

    # 4. TTS: Kokoro-82M on GPU 1 + 句読点ストリーミング Chunking
    base_tts = openai.TTS(
        base_url=audio_base_url,
        api_key="none",
        model="hexgrad/Kokoro-82M",
        voice=kokoro_voice,
        response_format="wav", # WAVで無圧縮・超低遅延ストリーミング
    )
    streaming_tts = tts.StreamAdapter(
        tts=base_tts,
        sentence_tokenizer=JapaneseSentenceTokenizer(min_sentence_len=4, comma_split_len=12),
    )

    # 5. AgentSession 設定 (バージイン・割り込み & 句読点即時送出)
    session = AgentSession(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=streaming_tts,
        turn_handling={
            "endpointing": {
                "min_delay": 0.25, # ユーザーの発話停止から応答生成開始までの閾値 (250ms)
                "max_delay": 1.5,
            },
            "interruption": {
                "enabled": True, # バージイン有効
                "min_duration": 0.2, # 200msのユーザー発話を検知した瞬間にAI発話を即時キャンセル
            },
            "preemptive_generation": {
                "preemptive_tts": True, # 句読点チャンク単位の先行TTS生成
            },
        },
    )

    # イベントハンドラーの設定 (モニタリング・ログ)
    @session.on("user_input_transcribed")
    def on_user_speech(ev):
        logger.info(f"[{ctx.room.name}] User: {ev.transcript}")

    @session.on("agent_state_changed")
    def on_agent_state(ev):
        logger.info(f"[{ctx.room.name}] Agent state: {ev.new_state}")

    # セッション開始
    agent = Agent(instructions=INSTRUCTIONS)
    await session.start(room=ctx.room, agent=agent)

    # 初回挨拶の発話 (初動遅延テストを兼ねる)
    try:
        await session.say("こんにちは！何かお手伝いできることはありますか？", allow_interruptions=True)
    except Exception as e:
        logger.warning(f"Failed to play initial greeting: {e}")

if __name__ == "__main__":
    # 10人同時接続を安定処理するためのワーカープロセス数設定
    num_idle_processes = int(os.getenv("NUM_IDLE_PROCESSES", "4"))
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            num_idle_processes=num_idle_processes,
            job_memory_limit_mb=0, # プロセスごとの不要なOOM Killを防止
        )
    )
