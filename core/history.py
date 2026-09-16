from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


HISTORY_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_PATH = HISTORY_DIR / "analysis_history.json"


def _ensure_store() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text("[]", encoding="utf-8")


def load_history() -> list[dict[str, Any]]:
    _ensure_store()
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return data


def save_history(entries: list[dict[str, Any]]) -> None:
    _ensure_store()
    HISTORY_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
    entries = load_history()
    record = {
        "id": entry.get("id") or str(uuid.uuid4()),
        "timestamp": entry.get("timestamp")
        or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **{key: value for key, value in entry.items() if key not in {"id", "timestamp"}},
    }
    entries.insert(0, record)
    save_history(entries)
    return record


def delete_history_entry(entry_id: str) -> None:
    entries = [item for item in load_history() if item.get("id") != entry_id]
    save_history(entries)


def clear_history() -> None:
    save_history([])


def delete_history_by_filename(file_name: str) -> int:
    entries = load_history()
    kept = [item for item in entries if item.get("file_name") != file_name]
    removed = len(entries) - len(kept)
    save_history(kept)
    return removed
