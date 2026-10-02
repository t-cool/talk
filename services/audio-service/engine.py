import io
import os
import time
import wave
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

logger = logging.getLogger("audio-engine")

class AudioEngine:
    """
    GPU 1 専用の 音声 I/O (STT + TTS) 推論エンジン
    - STT: faster-whisper (small / medium) with CTranslate2 CUDA float16
    - TTS: Kokoro-82M with Japanese phonemizer (misaki[ja]) on CUDA
    """
    def __init__(
        self,
        whisper_model_id: str = "Systran/faster-whisper-medium",
        default_voice: str = "jf_alpha",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        self.whisper_model_id = whisper_model_id
        self.default_voice = default_voice
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info(
            f"Initializing AudioEngine on device: {self.device}, compute_type: {self.compute_type}"
        )

        self._init_stt()
        self._init_tts()
        self._warmup()

    def _init_stt(self):
        """faster-whisper モデルのロード (GPU 1)"""
        logger.info(f"Loading faster-whisper model: {self.whisper_model_id} on {self.device}...")
        t0 = time.time()
        try:
            from faster_whisper import WhisperModel
            # num_workers allows concurrent transcription requests within CTranslate2
            self.whisper = WhisperModel(
                self.whisper_model_id,
                device=self.device,
                compute_type=self.compute_type,
                num_workers=4,
                download_root="/root/.cache/huggingface",
            )
            logger.info(f"faster-whisper loaded in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load faster-whisper: {e}", exc_info=True)
            raise

    def _init_tts(self):
        """Kokoro-82M 日本語パイプラインのロード (GPU 1)"""
        logger.info(f"Loading Kokoro-82M (Japanese) on {self.device}...")
        t0 = time.time()
        try:
            from kokoro import KPipeline
            # lang_code 'j' activates Japanese phonemizer with misaki[ja]
            self.kokoro = KPipeline(lang_code="j", device=self.device)
            logger.info(f"Kokoro-82M loaded in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load Kokoro TTS: {e}", exc_info=True)
            raise

    def _warmup(self):
        """推論エンジンの事前ウォームアップ (初回遅延の解消)"""
        logger.info("Warming up STT & TTS engines...")
        try:
            # 1. TTS Warmup
            _ = self.synthesize_sync("起動完了しました。", voice=self.default_voice, output_format="wav")
            # 2. STT Warmup with 0.5s dummy audio
            dummy_pcm = np.zeros(8000, dtype=np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(dummy_pcm.tobytes())
            buf.seek(0)
            _ = self.transcribe_sync(buf.read(), language="ja")
            logger.info("Warmup complete. Ready for real-time traffic!")
        except Exception as e:
            logger.warning(f"Warmup encountered an issue (ignorable if downloading models): {e}")

    def transcribe_sync(self, audio_bytes: bytes, language: str = "ja", temperature: float = 0.0) -> str:
        """
        音声をテキストに変換 (faster-whisper)
        """
        t0 = time.time()
        audio_stream = io.BytesIO(audio_bytes)
        segments, info = self.whisper.transcribe(
            audio_stream,
            language=language,
            temperature=temperature,
            beam_size=5,
            vad_filter=False, # LiveKit handles VAD upstream
        )
        text = "".join(segment.text for segment in segments).strip()
        elapsed = time.time() - t0
        logger.debug(f"Transcribed {len(audio_bytes)} bytes -> '{text}' in {elapsed * 1000:.1f}ms")
        return text

    async def transcribe(self, audio_bytes: bytes, language: str = "ja", temperature: float = 0.0) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self.transcribe_sync,
            audio_bytes,
            language,
            temperature,
        )

    def synthesize_sync(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        output_format: str = "wav",
    ) -> bytes:
        """
        テキストを音声に変換 (Kokoro-82M)
        WAV (24kHz 16-bit PCM Mono) または MP3
        """
        t0 = time.time()
        v = voice or self.default_voice
        clean_text = text.strip()
        if not clean_text:
            clean_text = "…"

        # Kokoro generator emits (graphemes, phonemes, audio_samples)
        generator = self.kokoro(clean_text, voice=v, speed=speed)
        audio_chunks = []
        for _, _, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)

        if not audio_chunks:
            # Silence fallback (0.1s)
            full_audio = np.zeros(2400, dtype=np.float32)
        else:
            full_audio = np.concatenate(audio_chunks)

        # Kokoro sample rate is 24000 Hz
        sample_rate = 24000

        # Encode to WAV (fastest, uncompressed for lowest latency)
        pcm_int16 = (full_audio * 32767.0).clip(-32768, 32767).astype(np.int16)
        out_buf = io.BytesIO()

        if output_format.lower() == "mp3":
            try:
                import soundfile as sf
                sf.write(out_buf, pcm_int16, sample_rate, format="MP3")
                out_bytes = out_buf.getvalue()
            except Exception:
                # Fallback to WAV if soundfile lacks MP3 encoder
                out_buf.seek(0)
                out_buf.truncate(0)
                with wave.open(out_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm_int16.tobytes())
                out_bytes = out_buf.getvalue()
        else:
            with wave.open(out_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_int16.tobytes())
            out_bytes = out_buf.getvalue()

        elapsed = time.time() - t0
        audio_sec = len(full_audio) / sample_rate
        rtf = elapsed / max(audio_sec, 0.001)
        logger.debug(
            f"Synthesized '{clean_text[:20]}...' ({audio_sec:.2f}s) in {elapsed * 1000:.1f}ms (RTF: {rtf:.3f})"
        )
        return out_bytes

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        output_format: str = "wav",
    ) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self.synthesize_sync,
            text,
            voice,
            speed,
            output_format,
        )

    def get_stats(self) -> Dict[str, Any]:
        """GPU 1 の VRAM 使用統計を取得"""
        stats: Dict[str, Any] = {
            "device": self.device,
            "whisper_model": self.whisper_model_id,
            "default_voice": self.default_voice,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            stats.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
                "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2),
                "vram_total_mb": round(
                    torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2
                ),
            })
        return stats
