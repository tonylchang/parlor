"""Speech-to-text via faster-whisper (CTranslate2)."""

import ctypes
import glob
import io
import os
import sys


def _preload_cuda_libs() -> None:
    """Preload pip-installed CUDA runtime libs so ctranslate2 can dlopen them.

    The nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels drop .so files under
    site-packages/nvidia/*/lib, which aren't on the linker's search path.
    ctypes.CDLL loads them into the process's namespace by name.
    """
    if sys.platform != "linux":
        return
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    # `nvidia` is a PEP 420 namespace package — use __path__, not __file__.
    for base in nvidia.__path__:
        for so in sorted(glob.glob(os.path.join(base, "*", "lib", "lib*.so*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_cuda_libs()

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


class WhisperSTT:
    def __init__(self, model_size: str, device: str, compute_type: str):
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model_size = model_size
        self.device = device

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str:
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"expected 16kHz audio, got {sr}")
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return "".join(s.text for s in segments).strip()


def _try_backend(model_size: str, device: str, compute_type: str) -> WhisperSTT:
    """Instantiate and warm up so CUDA errors surface now, not on first request."""
    backend = WhisperSTT(model_size, device=device, compute_type=compute_type)
    warmup = np.zeros(16000, dtype=np.float32)  # 1s of silence
    segments, _ = backend._model.transcribe(warmup, language="en", beam_size=1, vad_filter=False)
    list(segments)
    return backend


def load(model_size: str = "medium") -> WhisperSTT:
    try:
        backend = _try_backend(model_size, "cuda", "float16")
        print(f"STT: faster-whisper {model_size} (CUDA, fp16)")
        return backend
    except Exception as e:
        print(f"STT: CUDA unavailable ({e}); falling back to CPU int8")
        backend = _try_backend(model_size, "cpu", "int8")
        print(f"STT: faster-whisper {model_size} (CPU, int8)")
        return backend
