from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DEFAULT_LABELS = ("real", "fake")
REQUIRED_CONFIG_KEYS = ("SR", "DURATION", "N_LFCC", "N_FFT", "HOP_LENGTH")


class CheckpointError(ValueError):
    """Raised when a checkpoint is incompatible with this application."""


@dataclass(frozen=True)
class ModelConfig:
    sample_rate: int
    duration: float
    n_lfcc: int
    n_fft: int
    hop_length: int

    @property
    def sample_count(self) -> int:
        return int(self.sample_rate * self.duration)

    @property
    def frame_count(self) -> int:
        # librosa.stft defaults to center=True, producing one frame at t=0.
        return 1 + self.sample_count // self.hop_length

    @property
    def feature_count(self) -> int:
        return self.n_lfcc * self.frame_count


@dataclass
class ModelBundle:
    path: Path
    model: nn.Module
    scaler: Any
    config: ModelConfig
    labels: tuple[str, str]
    device: torch.device
    metrics: dict[str, float]


class LFCC_CNN(nn.Module):
    """Architecture used by the English S2 + Thai S2.1 LFCC notebook."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.head(x.reshape(x.size(0), -1))


def discover_models(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    return sorted(
        (path for path in models_dir.glob("*.pt") if path.is_file()),
        key=lambda path: path.name.lower(),
    )


def _parse_config(raw: Any) -> ModelConfig:
    if not isinstance(raw, dict):
        raise CheckpointError("checkpoint ไม่มี config สำหรับ preprocessing")
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in raw]
    if missing:
        raise CheckpointError(f"checkpoint config ขาดค่า: {', '.join(missing)}")

    config = ModelConfig(
        sample_rate=int(raw["SR"]),
        duration=float(raw["DURATION"]),
        n_lfcc=int(raw["N_LFCC"]),
        n_fft=int(raw["N_FFT"]),
        hop_length=int(raw["HOP_LENGTH"]),
    )
    if min(
        config.sample_rate,
        config.n_lfcc,
        config.n_fft,
        config.hop_length,
    ) <= 0 or config.duration <= 0:
        raise CheckpointError("checkpoint config มีค่าที่ไม่ถูกต้อง")
    return config


def _extract_metrics(checkpoint: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in ("best_val_loss", "best_val_acc", "test_acc"):
        value = checkpoint.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)

    for prefix, checkpoint_key in (
        ("validation", "validation_accuracy"),
        ("test", "test_accuracy"),
    ):
        values = checkpoint.get(checkpoint_key)
        if not isinstance(values, dict):
            continue
        for language in ("combined", "english", "thai"):
            value = values.get(language)
            if isinstance(value, (int, float)):
                metrics[f"{prefix}_{language}_acc"] = float(value)
    return metrics


@lru_cache(maxsize=8)
def _load_checkpoint_cached(
    path_string: str,
    modified_ns: int,
    file_size: int,
    device_name: str,
) -> ModelBundle:
    del modified_ns, file_size  # Values form the cache key.
    path = Path(path_string)
    device = torch.device(device_name)

    try:
        # This project stores an sklearn scaler in the checkpoint, so
        # weights_only=False is required. Load trusted local files only.
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"เปิด checkpoint ไม่สำเร็จ: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise CheckpointError("checkpoint ต้องเป็น dictionary")
    if "model_state_dict" not in checkpoint or "scaler" not in checkpoint:
        raise CheckpointError("checkpoint ต้องมี model_state_dict และ scaler")

    config = _parse_config(checkpoint.get("config"))
    scaler = checkpoint["scaler"]
    scaler_feature_count = getattr(scaler, "n_features_in_", None)
    if scaler_feature_count != config.feature_count:
        raise CheckpointError(
            "จำนวนฟีเจอร์ของ scaler ไม่ตรงกับ config "
            f"({scaler_feature_count} != {config.feature_count})"
        )
    if not callable(getattr(scaler, "transform", None)):
        raise CheckpointError("scaler ใน checkpoint ไม่มี transform()")

    model = LFCC_CNN().to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as exc:
        raise CheckpointError(
            "โครงสร้างโมเดลไม่ตรงกับ LFCC_CNN ที่แอปรองรับ"
        ) from exc
    model.eval()

    raw_labels = checkpoint.get("label_names", DEFAULT_LABELS)
    labels = tuple(str(label).lower() for label in raw_labels)
    if len(labels) != 2 or set(labels) != set(DEFAULT_LABELS):
        raise CheckpointError("label_names ต้องประกอบด้วย real และ fake")

    return ModelBundle(
        path=path,
        model=model,
        scaler=scaler,
        config=config,
        labels=(labels[0], labels[1]),
        device=device,
        metrics=_extract_metrics(checkpoint),
    )


def load_checkpoint(path: Path) -> ModelBundle:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ไม่พบโมเดล: {resolved}")
    stat = resolved.stat()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return _load_checkpoint_cached(
        str(resolved),
        stat.st_mtime_ns,
        stat.st_size,
        device_name,
    )


def clear_model_cache() -> None:
    _load_checkpoint_cached.cache_clear()
