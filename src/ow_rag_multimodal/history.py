"""Persistence helpers for player hero usage history."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "data" / "player_history.json"


def _default_history() -> dict[str, object]:
    """Builds the default empty history payload.

    Returns:
        A dictionary with initialized counters, sequence, and timestamp fields.
    """

    return {
        "played_counts": {},
        "played_sequence": [],
        "updated_at": None,
    }


def load_history(path: Path = DEFAULT_HISTORY_PATH) -> dict[str, object]:
    """Loads and normalizes history data from disk.

    Args:
        path: Path to the history JSON file.

    Returns:
        A normalized history dictionary. If loading fails, returns defaults.
    """

    if not path.exists():
        return _default_history()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_history()

    if not isinstance(raw, dict):
        return _default_history()

    counts = raw.get("played_counts")
    sequence = raw.get("played_sequence")

    if not isinstance(counts, dict):
        counts = {}
    if not isinstance(sequence, list):
        sequence = []

    normalized_counts = {str(k): int(v) for k, v in counts.items() if str(k).strip()}
    normalized_sequence = [str(item) for item in sequence if str(item).strip()]
    updated_at = raw.get("updated_at")
    updated = str(updated_at) if updated_at else None

    return {
        "played_counts": normalized_counts,
        "played_sequence": normalized_sequence,
        "updated_at": updated,
    }


def save_history(history: dict[str, object], path: Path = DEFAULT_HISTORY_PATH) -> None:
    """Persists history data to disk and stamps update time.

    Args:
        history: History dictionary to serialize.
        path: Destination JSON path.

    Returns:
        ``None``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "played_counts": history.get("played_counts", {}),
        "played_sequence": history.get("played_sequence", []),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_played(slugs: list[str], path: Path = DEFAULT_HISTORY_PATH) -> dict[str, object]:
    """Records played hero slugs into history counters and sequence.

    Args:
        slugs: Hero slugs to append.
        path: History JSON path.

    Returns:
        The updated normalized history payload after persistence.
    """

    history = load_history(path)
    counts = Counter(history.get("played_counts", {}))
    sequence = list(history.get("played_sequence", []))

    for slug in slugs:
        clean = slug.strip().lower()
        if not clean:
            continue
        counts[clean] += 1
        sequence.append(clean)

    history["played_counts"] = dict(counts)
    history["played_sequence"] = sequence[-500:]
    save_history(history, path)
    return history


def top_played_slugs(limit: int = 8, path: Path = DEFAULT_HISTORY_PATH) -> list[str]:
    """Returns the most frequently played hero slugs.

    Args:
        limit: Maximum number of slugs to return.
        path: History JSON path.

    Returns:
        Ordered list of slugs by descending play count.
    """

    history = load_history(path)
    counts_raw = history.get("played_counts", {})
    if not isinstance(counts_raw, dict):
        return []

    counts = Counter({str(k): int(v) for k, v in counts_raw.items()})
    return [slug for slug, _ in counts.most_common(max(limit, 0))]


def clear_history(path: Path = DEFAULT_HISTORY_PATH) -> dict[str, object]:
    """Resets history storage to its default empty state.

    Args:
        path: History JSON path.

    Returns:
        The reset history payload.
    """

    history = _default_history()
    save_history(history, path)
    return history
