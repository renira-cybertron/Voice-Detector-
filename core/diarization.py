from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeechSegment:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    rms: float


def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    min_speech_sec: float = 0.25,
    min_silence_sec: float = 0.3,
    threshold_ratio: float = 0.35,
) -> tuple[SpeechSegment, ...]:
    """Energy-based speech activity detection (local lightweight diarization aid)."""
    if audio.size == 0:
        return ()

    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_size = max(1, int(sample_rate * hop_ms / 1000.0))
    if audio.size < frame_size:
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return (
            SpeechSegment(
                index=1,
                start_sec=0.0,
                end_sec=audio.size / sample_rate,
                duration_sec=audio.size / sample_rate,
                rms=rms,
            ),
        )

    energies: list[float] = []
    for start in range(0, audio.size - frame_size + 1, hop_size):
        frame = audio[start : start + frame_size]
        energies.append(float(np.sqrt(np.mean(np.square(frame)))))

    energy = np.asarray(energies, dtype=np.float64)
    peak = float(np.max(energy)) if energy.size else 0.0
    if peak <= 1e-8:
        return ()

    threshold = peak * threshold_ratio
    voiced = energy >= threshold

    min_speech_frames = max(1, int(min_speech_sec * 1000.0 / hop_ms))
    min_silence_frames = max(1, int(min_silence_sec * 1000.0 / hop_ms))

    # Merge short silence gaps inside speech.
    cleaned = voiced.copy()
    index = 0
    while index < cleaned.size:
        if cleaned[index]:
            index += 1
            continue
        start = index
        while index < cleaned.size and not cleaned[index]:
            index += 1
        if start > 0 and index < cleaned.size and (index - start) < min_silence_frames:
            cleaned[start:index] = True

    segments: list[SpeechSegment] = []
    index = 0
    while index < cleaned.size:
        if not cleaned[index]:
            index += 1
            continue
        start = index
        while index < cleaned.size and cleaned[index]:
            index += 1
        if (index - start) < min_speech_frames:
            continue
        start_sec = start * hop_ms / 1000.0
        end_sec = min(audio.size / sample_rate, index * hop_ms / 1000.0)
        sample_start = int(start_sec * sample_rate)
        sample_end = max(sample_start + 1, int(end_sec * sample_rate))
        chunk = audio[sample_start:sample_end]
        segments.append(
            SpeechSegment(
                index=len(segments) + 1,
                start_sec=round(start_sec, 2),
                end_sec=round(end_sec, 2),
                duration_sec=round(end_sec - start_sec, 2),
                rms=float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0,
            )
        )

    return tuple(segments)
