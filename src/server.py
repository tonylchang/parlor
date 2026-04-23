"""Parlor — on-device, real-time multimodal AI (voice + vision) via Ollama."""

import asyncio
import base64
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import stt
import tts
from ollama_client import OllamaClient

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
#OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "InternVL3_5:8b")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE") or None  # None = auto-detect

SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone (their speech has already been transcribed) and may show "
    "you their camera. Keep replies to 1-4 short sentences — your words will be "
    "spoken aloud by text-to-speech. Do not use emojis as text-to-speech will make that output sound silly."
)

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

stt_backend = None
tts_backend = None
ollama: OllamaClient | None = None


def load_models():
    global stt_backend, tts_backend, ollama
    stt_backend = stt.load(model_size=WHISPER_MODEL)
    tts_backend = tts.load()
    ollama = OllamaClient(host=OLLAMA_HOST, model=OLLAMA_MODEL)
    print(f"Ollama: {OLLAMA_HOST} model={OLLAMA_MODEL}")


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield
    if ollama is not None:
        await ollama.close()


app = FastAPI(lifespan=lifespan)


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.get("/config")
async def config():
    return {
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "whisper_model": WHISPER_MODEL,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Text-only rolling history (images are sent only with the current turn)
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    interrupted = asyncio.Event()
    msg_queue: asyncio.Queue = asyncio.Queue()

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            interrupted.clear()

            audio_b64 = msg.get("audio")
            image_b64 = msg.get("image")
            text_override = msg.get("text")

            # STT
            transcription = ""
            if audio_b64:
                t0 = time.time()
                wav_bytes = base64.b64decode(audio_b64)
                transcription = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: stt_backend.transcribe(wav_bytes, language=WHISPER_LANGUAGE)
                )
                stt_time = time.time() - t0
                print(f"STT ({stt_time:.2f}s): {transcription!r}")

            user_text = transcription or text_override
            if not user_text and not image_b64:
                continue
            if not user_text:
                user_text = "The user is showing you their camera. Describe what you see."

            if interrupted.is_set():
                continue

            # LLM via Ollama — include current image only; don't persist it in history
            current_msg = {"role": "user", "content": user_text}
            if image_b64:
                current_msg["images"] = [image_b64]
            request_messages = history + [current_msg]

            t0 = time.time()
            try:
                reply = await ollama.chat(request_messages)
            except Exception as e:
                print(f"Ollama error: {e}")
                await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
                continue
            llm_time = time.time() - t0
            print(f"LLM ({llm_time:.2f}s): {reply}")

            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})

            if interrupted.is_set():
                continue

            reply_msg = {"type": "text", "text": reply, "llm_time": round(llm_time, 2)}
            if transcription:
                reply_msg["transcription"] = transcription
            await ws.send_text(json.dumps(reply_msg))

            if interrupted.is_set():
                continue

            # Streaming TTS
            sentences = split_sentences(reply) or [reply]
            tts_start = time.time()

            await ws.send_text(json.dumps({
                "type": "audio_start",
                "sample_rate": tts_backend.sample_rate,
                "sentence_count": len(sentences),
            }))

            for i, sentence in enumerate(sentences):
                if interrupted.is_set():
                    print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
                    break

                pcm = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sentence: tts_backend.generate(s)
                )

                if interrupted.is_set():
                    break

                pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                await ws.send_text(json.dumps({
                    "type": "audio_chunk",
                    "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                    "index": i,
                }))

            tts_time = time.time() - tts_start
            print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

            if not interrupted.is_set():
                await ws.send_text(json.dumps({
                    "type": "audio_end",
                    "tts_time": round(tts_time, 2),
                }))

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
