"""CrispASR Python wrapper via ctypes.

Provides speech-to-text transcription using ggml inference.
Wraps the whisper.h C API from whisper.cpp / CrispASR.
"""

import ctypes
import os
import platform
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


@dataclass
class Segment:
    """A transcription segment with timing information."""
    text: str
    start: float  # seconds
    end: float    # seconds
    no_speech_prob: float = 0.0


def _find_lib():
    """Find the crispasr / whisper shared library.

    As of CrispASR 0.4.0 the build produces both `libcrispasr.*`
    (preferred — all 10 backends linked) and the historical
    `libwhisper.*` (alias). We probe candidates in order so any build
    layout is picked up transparently.
    """
    system = platform.system()
    if system == "Darwin":
        candidates = ["libcrispasr.dylib", "libwhisper.dylib"]
    elif system == "Windows":
        candidates = ["crispasr.dll", "whisper.dll"]
    else:
        candidates = ["libcrispasr.so", "libwhisper.so"]

    search = [
        Path(__file__).parent,
        Path(__file__).parent.parent.parent / "build",
        Path(__file__).parent.parent.parent / "build" / "src",
        Path(__file__).parent.parent.parent / "build" / "lib",
        Path.cwd() / "build",
        Path.cwd() / "build" / "src",
    ]
    for d in search:
        for name in candidates:
            p = d / name
            if p.exists():
                return str(p)
    return candidates[0]


# Whisper sampling strategies
WHISPER_SAMPLING_GREEDY = 0
WHISPER_SAMPLING_BEAM_SEARCH = 1


class CrispASR:
    """Speech-to-text model using ggml inference.

    Usage:
        model = CrispASR("ggml-base.en.bin")
        segments = model.transcribe("audio.wav")
        for seg in segments:
            print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")

        # Or from raw PCM data
        segments = model.transcribe_pcm(pcm_f32, sample_rate=16000)

        model.close()
    """

    def __init__(self, model_path: str, lib_path: Optional[str] = None,
                 helpers_lib_path: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        self._setup_signatures()

        # Load helpers library (provides pointer-based wrappers for by-value struct APIs)
        helpers_search = [
            helpers_lib_path,
            str(Path(lib_path).parent / "libcrispasr_helpers.so") if lib_path else None,
            str(Path(__file__).parent.parent.parent / "build" / "libcrispasr_helpers.so"),
        ]
        self._helpers = None
        for hp in helpers_search:
            if hp and Path(hp).exists():
                self._helpers = ctypes.CDLL(hp)
                break

        if self._helpers:
            # Use pointer-based wrappers (avoids by-value struct issues)
            self._helpers.whisper_init_from_file_ptr.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
            self._helpers.whisper_init_from_file_ptr.restype = ctypes.c_void_p
            self._helpers.whisper_full_ptr.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ]
            self._helpers.whisper_full_ptr.restype = ctypes.c_int

            cparams = self._lib.whisper_context_default_params_by_ref()
            self._ctx = self._helpers.whisper_init_from_file_ptr(
                model_path.encode("utf-8"), cparams
            )
            self._lib.whisper_free_context_params(cparams)
        else:
            # Fallback: use deprecated simple init (no params)
            self._lib.whisper_init_from_file.argtypes = [ctypes.c_char_p]
            self._lib.whisper_init_from_file.restype = ctypes.c_void_p
            self._ctx = self._lib.whisper_init_from_file(model_path.encode("utf-8"))

        if not self._ctx:
            raise RuntimeError(f"Failed to load model: {model_path}")

    def _setup_signatures(self):
        lib = self._lib

        # Free
        lib.whisper_free.argtypes = [ctypes.c_void_p]
        lib.whisper_free.restype = None

        # Context params (by ref)
        lib.whisper_context_default_params_by_ref.argtypes = []
        lib.whisper_context_default_params_by_ref.restype = ctypes.c_void_p

        lib.whisper_free_context_params.argtypes = [ctypes.c_void_p]
        lib.whisper_free_context_params.restype = None

        # Full params (by ref)
        lib.whisper_full_default_params_by_ref.argtypes = [ctypes.c_int]
        lib.whisper_full_default_params_by_ref.restype = ctypes.c_void_p

        lib.whisper_free_params.argtypes = [ctypes.c_void_p]
        lib.whisper_free_params.restype = None

        # whisper_full (takes params by value — needs helpers lib for pointer variant)
        lib.whisper_full.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        lib.whisper_full.restype = ctypes.c_int

        # Results (ctx-based variants)
        lib.whisper_full_n_segments.argtypes = [ctypes.c_void_p]
        lib.whisper_full_n_segments.restype = ctypes.c_int

        lib.whisper_full_get_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.whisper_full_get_segment_text.restype = ctypes.c_char_p

        lib.whisper_full_get_segment_t0.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.whisper_full_get_segment_t0.restype = ctypes.c_int64

        lib.whisper_full_get_segment_t1.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.whisper_full_get_segment_t1.restype = ctypes.c_int64

        lib.whisper_full_get_segment_no_speech_prob.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.whisper_full_get_segment_no_speech_prob.restype = ctypes.c_float

        # Language
        lib.whisper_full_lang_id.argtypes = [ctypes.c_void_p]
        lib.whisper_full_lang_id.restype = ctypes.c_int

        lib.whisper_lang_str.argtypes = [ctypes.c_int]
        lib.whisper_lang_str.restype = ctypes.c_char_p

        # 0.4.2 — VAD + tdrz param setters on whisper_full_params.
        for _sym, _argtypes in [
            ("crispasr_params_set_vad", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_vad_model_path", [ctypes.c_void_p, ctypes.c_char_p]),
            ("crispasr_params_set_vad_threshold", [ctypes.c_void_p, ctypes.c_float]),
            ("crispasr_params_set_vad_min_speech_ms", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_vad_min_silence_ms", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_tdrz", [ctypes.c_void_p, ctypes.c_int]),
        ]:
            if hasattr(lib, _sym):
                getattr(lib, _sym).argtypes = _argtypes
                getattr(lib, _sym).restype = None

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        strategy: int = WHISPER_SAMPLING_GREEDY,
    ) -> List[Segment]:
        """Transcribe an audio file (WAV, 16kHz mono recommended).

        Args:
            audio_path: Path to audio file.
            language: Language code (e.g. "en", "de") or "auto" for detection.
            strategy: WHISPER_SAMPLING_GREEDY or WHISPER_SAMPLING_BEAM_SEARCH.

        Returns:
            List of Segment objects with text and timing.
        """
        pcm = self._load_audio(audio_path)
        return self.transcribe_pcm(pcm, language=language, strategy=strategy)

    def transcribe_pcm(
        self,
        pcm: np.ndarray,
        sample_rate: int = 16000,
        language: str = "auto",
        strategy: int = WHISPER_SAMPLING_GREEDY,
        vad: bool = False,
        vad_model_path: Optional[str] = None,
        vad_threshold: float = 0.5,
        vad_min_speech_ms: int = 250,
        vad_min_silence_ms: int = 100,
        tdrz: bool = False,
    ) -> List[Segment]:
        """Transcribe raw PCM audio data.

        Args:
            pcm: Float32 mono PCM samples.
            sample_rate: Sample rate (will be resampled to 16kHz if different).
            language: Language code or "auto".
            strategy: Sampling strategy.
            vad: Enable Silero VAD to skip silent regions (0.4.2+ dylibs).
            vad_model_path: Path to Silero VAD GGML model. Required when vad=True.
            vad_threshold: Speech detection threshold (0.0-1.0, default 0.5).
            vad_min_speech_ms: Minimum speech span to keep (default 250ms).
            vad_min_silence_ms: Minimum silence span to split on (default 100ms).
            tdrz: Enable tinydiarize speaker-turn markers (requires .en.tdrz model).

        Returns:
            List of Segment objects.
        """
        if sample_rate != 16000:
            # Simple resampling via linear interpolation
            ratio = 16000 / sample_rate
            new_len = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, new_len)
            pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)

        pcm = pcm.astype(np.float32)
        samples_ptr = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Get default params
        params_ptr = self._lib.whisper_full_default_params_by_ref(strategy)

        # 0.4.2: VAD + tdrz. Setters are optional — older dylibs don't
        # have them, the lookup-time hasattr() guard skipped the argtypes
        # declaration so these calls no-op silently.
        if vad and hasattr(self._lib, "crispasr_params_set_vad"):
            self._lib.crispasr_params_set_vad(params_ptr, 1)
            if hasattr(self._lib, "crispasr_params_set_vad_threshold"):
                self._lib.crispasr_params_set_vad_threshold(params_ptr, vad_threshold)
            if hasattr(self._lib, "crispasr_params_set_vad_min_speech_ms"):
                self._lib.crispasr_params_set_vad_min_speech_ms(params_ptr, vad_min_speech_ms)
            if hasattr(self._lib, "crispasr_params_set_vad_min_silence_ms"):
                self._lib.crispasr_params_set_vad_min_silence_ms(params_ptr, vad_min_silence_ms)
            if vad_model_path and hasattr(self._lib, "crispasr_params_set_vad_model_path"):
                self._lib.crispasr_params_set_vad_model_path(
                    params_ptr, vad_model_path.encode("utf-8")
                )
        if tdrz and hasattr(self._lib, "crispasr_params_set_tdrz"):
            self._lib.crispasr_params_set_tdrz(params_ptr, 1)

        # Run inference
        if self._helpers:
            ret = self._helpers.whisper_full_ptr(self._ctx, params_ptr, samples_ptr, len(pcm))
        else:
            ret = self._lib.whisper_full(self._ctx, params_ptr, samples_ptr, len(pcm))
        self._lib.whisper_free_params(params_ptr)

        if ret != 0:
            raise RuntimeError(f"Transcription failed (error code {ret})")

        # Collect segments
        n_segments = self._lib.whisper_full_n_segments(self._ctx)
        segments = []
        for i in range(n_segments):
            text_bytes = self._lib.whisper_full_get_segment_text(self._ctx, i)
            text = text_bytes.decode("utf-8") if text_bytes else ""
            t0 = self._lib.whisper_full_get_segment_t0(self._ctx, i) / 100.0
            t1 = self._lib.whisper_full_get_segment_t1(self._ctx, i) / 100.0
            nsp = float(self._lib.whisper_full_get_segment_no_speech_prob(self._ctx, i))
            segments.append(Segment(text=text, start=t0, end=t1, no_speech_prob=nsp))

        return segments

    @property
    def detected_language(self) -> str:
        """Language detected during the last transcription."""
        lang_id = self._lib.whisper_full_lang_id(self._ctx)
        lang_str = self._lib.whisper_lang_str(lang_id)
        return lang_str.decode("utf-8") if lang_str else "unknown"

    @staticmethod
    def _load_audio(path: str) -> np.ndarray:
        """Load audio file to float32 mono PCM."""
        if path.endswith(".wav"):
            with wave.open(path, "rb") as wf:
                assert wf.getsampwidth() in (1, 2, 4), "Unsupported sample width"
                assert wf.getnchannels() in (1, 2), "Unsupported channel count"
                frames = wf.readframes(wf.getnframes())
                if wf.getsampwidth() == 2:
                    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif wf.getsampwidth() == 4:
                    pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    pcm = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                # Convert stereo to mono
                if wf.getnchannels() == 2:
                    pcm = pcm.reshape(-1, 2).mean(axis=1)
                # Resample if needed
                if wf.getframerate() != 16000:
                    ratio = 16000 / wf.getframerate()
                    new_len = int(len(pcm) * ratio)
                    indices = np.linspace(0, len(pcm) - 1, new_len)
                    pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)
                return pcm
        else:
            raise ValueError(f"Unsupported audio format: {path}. Use .wav or pass raw PCM via transcribe_pcm().")

    def close(self):
        """Release all resources."""
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.whisper_free(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =========================================================================
# Unified session — works for every backend libcrispasr was built with
# =========================================================================

@dataclass
class SessionWord:
    """Word-level timing from a session transcribe (backends that produce it)."""
    text: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class SessionSegment:
    """A transcription segment from Session.transcribe."""
    text: str
    start: float  # seconds
    end: float    # seconds
    words: List[SessionWord]


class Session:
    """Backend-agnostic transcription session over any CrispASR-supported GGUF.

    The backend is auto-detected from the file's `general.architecture`
    metadata. `Session.available_backends()` lists which backends the
    bundled libcrispasr was actually compiled with — a model whose
    backend isn't in that list will fail to open.

    Usage:
        with crispasr.Session("model.gguf") as s:
            print(f"backend: {s.backend}")
            for seg in s.transcribe(pcm_f32):
                print(f"[{seg.start:.1f}-{seg.end:.1f}s] {seg.text}")
    """

    def __init__(self, model_path: str, lib_path: Optional[str] = None,
                 n_threads: int = 4, backend: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        self._setup_session_signatures()

        path_bytes = model_path.encode("utf-8")
        if backend:
            self._handle = self._lib.crispasr_session_open_explicit(
                path_bytes, backend.encode("utf-8"), n_threads
            )
        else:
            self._handle = self._lib.crispasr_session_open(path_bytes, n_threads)

        if not self._handle:
            avail = Session.available_backends(lib_path=lib_path)
            raise RuntimeError(
                f"Failed to open {model_path!r} — backend not supported. "
                f"libcrispasr was built with: {avail}"
            )
        be = self._lib.crispasr_session_backend(self._handle)
        self.backend = be.decode("utf-8") if be else ""

    def _setup_session_signatures(self):
        lib = self._lib
        # Missing symbol ⇒ pre-0.4.0 dylib.
        for name in (
            "crispasr_session_open", "crispasr_session_transcribe",
            "crispasr_session_available_backends", "crispasr_session_close",
        ):
            if not hasattr(lib, name):
                raise RuntimeError(
                    "Unified session API not found in loaded library — "
                    "rebuild CrispASR with 0.4.0+ helpers."
                )

        lib.crispasr_session_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispasr_session_open.restype = ctypes.c_void_p
        lib.crispasr_session_open_explicit.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
        ]
        lib.crispasr_session_open_explicit.restype = ctypes.c_void_p
        lib.crispasr_session_backend.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_backend.restype = ctypes.c_char_p
        lib.crispasr_session_available_backends.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispasr_session_available_backends.restype = ctypes.c_int
        lib.crispasr_session_transcribe.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        lib.crispasr_session_transcribe.restype = ctypes.c_void_p
        lib.crispasr_session_result_n_segments.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_result_n_segments.restype = ctypes.c_int
        lib.crispasr_session_result_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.crispasr_session_result_segment_text.restype = ctypes.c_char_p
        lib.crispasr_session_result_segment_t0.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.crispasr_session_result_segment_t0.restype = ctypes.c_int64
        lib.crispasr_session_result_segment_t1.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.crispasr_session_result_segment_t1.restype = ctypes.c_int64
        lib.crispasr_session_result_n_words.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.crispasr_session_result_n_words.restype = ctypes.c_int
        lib.crispasr_session_result_word_text.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.crispasr_session_result_word_text.restype = ctypes.c_char_p
        lib.crispasr_session_result_word_t0.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.crispasr_session_result_word_t0.restype = ctypes.c_int64
        lib.crispasr_session_result_word_t1.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.crispasr_session_result_word_t1.restype = ctypes.c_int64
        lib.crispasr_session_result_free.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_result_free.restype = None
        lib.crispasr_session_close.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_close.restype = None

    @staticmethod
    def available_backends(lib_path: Optional[str] = None) -> List[str]:
        """List the backend names the loaded CrispASR library was built with."""
        lib = ctypes.CDLL(lib_path or _find_lib())
        if not hasattr(lib, "crispasr_session_available_backends"):
            return []
        lib.crispasr_session_available_backends.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispasr_session_available_backends.restype = ctypes.c_int
        buf = ctypes.create_string_buffer(256)
        lib.crispasr_session_available_backends(buf, 256)
        csv = buf.value.decode("utf-8")
        return [s.strip() for s in csv.split(",") if s.strip()]

    def transcribe(
        self, pcm: np.ndarray, sample_rate: int = 16000,
    ) -> List[SessionSegment]:
        """Transcribe 16 kHz mono float32 PCM. Dispatches via crispasr_session."""
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, new_len)
            pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)
        pcm = np.asarray(pcm, dtype=np.float32)
        samples_ptr = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        res = self._lib.crispasr_session_transcribe(self._handle, samples_ptr, len(pcm))
        if not res:
            raise RuntimeError(f"crispasr_session_transcribe failed for backend {self.backend!r}")

        try:
            n_seg = self._lib.crispasr_session_result_n_segments(res)
            out: List[SessionSegment] = []
            for i in range(n_seg):
                t = self._lib.crispasr_session_result_segment_text(res, i)
                text = t.decode("utf-8") if t else ""
                t0 = self._lib.crispasr_session_result_segment_t0(res, i) / 100.0
                t1 = self._lib.crispasr_session_result_segment_t1(res, i) / 100.0
                wn = self._lib.crispasr_session_result_n_words(res, i)
                words: List[SessionWord] = []
                for j in range(wn):
                    wt = self._lib.crispasr_session_result_word_text(res, i, j)
                    words.append(SessionWord(
                        text=wt.decode("utf-8") if wt else "",
                        start=self._lib.crispasr_session_result_word_t0(res, i, j) / 100.0,
                        end=self._lib.crispasr_session_result_word_t1(res, i, j) / 100.0,
                    ))
                out.append(SessionSegment(text=text.strip(), start=t0, end=t1, words=words))
            return out
        finally:
            self._lib.crispasr_session_result_free(res)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.crispasr_session_close(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
