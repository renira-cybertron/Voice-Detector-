from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .audio import AudioSegment, split_audio
from .features import compute_lfcc, scale_features
from .model import ModelBundle


@dataclass(frozen=True)
class SegmentPrediction:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    label: str
    real_probability: float
    fake_probability: float
    confidence: float
    rms: float


@dataclass(frozen=True)
class AnalysisResult:
    label: str
    real_probability: float
    fake_probability: float
    confidence: float
    max_fake_probability: float
    duration_sec: float
    segments: tuple[SegmentPrediction, ...]


def _predict_batch(
    segments: list[AudioSegment],
    bundle: ModelBundle,
) -> np.ndarray:
    features = np.stack(
        [compute_lfcc(segment.samples, bundle.config) for segment in segments]
    )
    scaled = scale_features(features, bundle)
    tensor = torch.from_numpy(scaled).unsqueeze(1).to(bundle.device)
    with torch.inference_mode():
        logits = bundle.model(tensor)
        return torch.softmax(logits, dim=1).cpu().numpy()


def analyze_audio(
    audio: np.ndarray,
    bundle: ModelBundle,
    batch_size: int = 64,
) -> AnalysisResult:
    segments = split_audio(audio, bundle.config)
    probability_batches = [
        _predict_batch(segments[start : start + batch_size], bundle)
        for start in range(0, len(segments), batch_size)
    ]
    probabilities = np.concatenate(probability_batches, axis=0)

    real_index = bundle.labels.index("real")
    fake_index = bundle.labels.index("fake")
    predictions: list[SegmentPrediction] = []
    for segment, probs in zip(segments, probabilities):
        label = bundle.labels[int(np.argmax(probs))]
        predictions.append(
            SegmentPrediction(
                index=segment.index,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                duration_sec=segment.valid_duration_sec,
                label=label,
                real_probability=float(probs[real_index]),
                fake_probability=float(probs[fake_index]),
                confidence=float(np.max(probs)),
                rms=segment.rms,
            )
        )

    weights = np.asarray(
        [segment.valid_duration_sec for segment in segments],
        dtype=np.float64,
    )
    overall = np.average(probabilities, axis=0, weights=weights)
    real_probability = float(overall[real_index])
    fake_probability = float(overall[fake_index])
    label = bundle.labels[int(np.argmax(overall))]

    return AnalysisResult(
        label=label,
        real_probability=real_probability,
        fake_probability=fake_probability,
        confidence=max(real_probability, fake_probability),
        max_fake_probability=max(
            prediction.fake_probability for prediction in predictions
        ),
        duration_sec=float(audio.size / bundle.config.sample_rate),
        segments=tuple(predictions),
    )
