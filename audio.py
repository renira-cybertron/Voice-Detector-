from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .model import ModelConfig


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}
VIDEO_EXTENSIONS = {".mp4"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class AudioError(ValueError):
    """Raised when an uploaded audio file cannot be decoded."""


@dataclass
class AudioSegment:
    index: int
    start_sec: float
    end_sec: float
    valid_duration_sec: float
    samples: np.ndarray
    rms: float


def find_ffmpeg() -> Path | None:
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    candidates = (
        Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe",
        Path(sys.prefix) / "Library" / "bin" / "ffmpeg",
        Path(sys.prefix) / "bin" / "ffmpeg.exe",
        Path(sys.prefix) / "bin" / "ffmpeg",
        Path(sys.executable).resolve().parent / "ffmpeg.exe",
        Path(sys.executable).resolve().parent / "ffmpeg",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _finalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise AudioError("ไฟล์เสียงไม่มีข้อมูล")
    if not np.isfinite(audio).all():
        raise AudioError("ไฟล์เสียงมี sample ที่ไม่ถูกต้อง")
    return audio


def _load_with_librosa(
    path: Path,
    sample_rate: int,
    max_duration_sec: float | None = None,
) -> np.ndarray:
    try:
        audio, _ = librosa.load(
            path,
            sr=sample_rate,
            mono=True,
            duration=max_duration_sec,
        )
    except Exception as exc:
        raise AudioError(
            "อ่านไฟล์เสียงไม่สำเร็จ กรุณาตรวจสอบไฟล์หรือแปลงเป็น WAV"
        ) from exc
    return _finalize_audio(audio)


def _load_audio_from_video(
    path: Path,
    sample_rate: int,
    max_duration_sec: float | None = None,
) -> np.ndarray:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise AudioError(
            "แปลง MP4 ไม่ได้ เพราะไม่พบ FFmpeg กรุณาแปลงเป็น WAV ก่อน"
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        wav_path = Path(temp_file.name)

    command = [
        str(ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
    ]
    if max_duration_sec is not None:
        command.extend(["-t", f"{float(max_duration_sec):.3f}"])
    command.append(str(wav_path))

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            creationflags=creationflags,
        )
        if (
            completed.returncode != 0
            or not wav_path.exists()
            or wav_path.stat().st_size == 0
        ):
            raise AudioError(
                "แปลง MP4 เป็นเสียงไม่สำเร็จ ไฟล์อาจไม่มีแทร็กเสียง"
            )
        return _load_with_librosa(wav_path, sample_rate, max_duration_sec)
    finally:
        wav_path.unlink(missing_ok=True)


def load_audio_file(
    path: Path,
    sample_rate: int,
    max_duration_sec: float | None = None,
) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return _load_audio_from_video(path, sample_rate, max_duration_sec)
    return _load_with_librosa(path, sample_rate, max_duration_sec)


def pad_or_trim(audio: np.ndarray, sample_count: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < sample_count:
        audio = np.pad(audio, (0, sample_count - audio.size), mode="constant")
    else:
        audio = audio[:sample_count]
    return audio.astype(np.float32, copy=False)


def limit_audio_duration(
    audio: np.ndarray,
    sample_rate: int,
    max_duration_sec: float,
) -> np.ndarray:
    """Keep only the beginning of an audio file up to the requested duration."""
    max_samples = int(sample_rate * max_duration_sec)
    if max_samples <= 0:
        raise ValueError("ระยะเวลาสูงสุดต้องมากกว่า 0 วินาที")
    return np.asarray(audio, dtype=np.float32)[:max_samples]


def split_audio(audio: np.ndarray, config: ModelConfig) -> list[AudioSegment]:
    """Split into consecutive training-sized windows and pad only the tail."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise AudioError("ไฟล์เสียงไม่มีข้อมูล")

    segments: list[AudioSegment] = []
    for zero_index, start in enumerate(
        range(0, audio.size, config.sample_count)
    ):
        raw = audio[start : start + config.sample_count]
        valid_samples = raw.size
        segments.append(
            AudioSegment(
                index=zero_index + 1,
                start_sec=start / config.sample_rate,
                end_sec=(start + valid_samples) / config.sample_rate,
                valid_duration_sec=valid_samples / config.sample_rate,
                samples=pad_or_trim(raw, config.sample_count),
                rms=float(np.sqrt(np.mean(raw**2))),
            )
        )
    return segments
