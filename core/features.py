from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np
from scipy.fftpack import dct

from .model import CheckpointError, ModelBundle, ModelConfig


@lru_cache(maxsize=16)
def linear_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
) -> np.ndarray:
    """Build the same linearly spaced triangular filterbank as the notebook."""
    center_frequencies = np.linspace(0, sample_rate / 2, n_filters + 2)
    bins = np.floor((n_fft + 1) * center_frequencies / sample_rate).astype(int)
    filterbank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float64)

    for index in range(n_filters):
        left, center, right = bins[index : index + 3]
        if center > left:
            filterbank[index, left:center] = np.linspace(
                0.0,
                1.0,
                center - left,
                endpoint=False,
            )
        if right > center:
            filterbank[index, center:right] = np.linspace(
                1.0,
                0.0,
                right - center,
                endpoint=False,
            )
    return filterbank


def compute_lfcc(audio: np.ndarray, config: ModelConfig) -> np.ndarray:
    """Replicate the notebook LFCC extraction without substituting MFCC."""
    power_spectrum = (
        np.abs(
            librosa.stft(
                audio,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
            )
        )
        ** 2
    )
    filterbank = linear_filterbank(
        config.n_lfcc,
        config.n_fft,
        config.sample_rate,
    )
    energies = np.dot(filterbank, power_spectrum)
    logged = np.log(energies + 1e-10)
    lfcc = dct(logged, type=2, axis=0, norm="ortho")[: config.n_lfcc]
    result = lfcc.astype(np.float32)

    expected_shape = (config.n_lfcc, config.frame_count)
    if result.shape != expected_shape:
        raise CheckpointError(
            f"LFCC shape ไม่ตรงกับ checkpoint: {result.shape} != {expected_shape}"
        )
    return result


def scale_features(features: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError("features ต้องมี shape (batch, LFCC, time)")
    count, n_lfcc, n_frames = features.shape
    flattened = features.reshape(count, -1)
    try:
        scaled = bundle.scaler.transform(flattened)
    except Exception as exc:
        raise CheckpointError(f"ใช้ scaler จาก checkpoint ไม่สำเร็จ: {exc}") from exc
    return scaled.reshape(count, n_lfcc, n_frames).astype(np.float32)
