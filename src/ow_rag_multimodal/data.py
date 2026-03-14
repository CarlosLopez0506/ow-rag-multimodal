"""Dataset loading and normalization utilities for hero documents."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import HeroDoc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEROES_PATH = PROJECT_ROOT / "data" / "heroes.json"
VALID_ROLES = {"Tank", "Damage", "Support"}


def _slugify(text: str) -> str:
    """Converts free text into a lowercase slug.

    Args:
        text: Source text to convert.

    Returns:
        A normalized slug. Returns ``"unknown"`` when the normalized result is empty.
    """

    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "unknown"


def _first_nonempty(row: dict[str, object], keys: tuple[str, ...]) -> str:
    """Returns the first non-empty string value found in a row.

    Args:
        row: Input mapping representing one hero record.
        keys: Candidate field names evaluated in order.

    Returns:
        The first non-empty stripped string found, or an empty string if none exists.
    """

    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _combined_text(slug: str, row: dict[str, object]) -> str:
    """Builds the descriptive text used for embeddings.

    Args:
        slug: Default identifier used as fallback for missing names.
        row: Raw hero record.

    Returns:
        A consolidated text string. Uses pre-combined fields when available,
        otherwise composes one from individual metadata fields.
    """

    combined = _first_nonempty(row, ("combined_text", "text", "description"))
    if combined:
        return combined

    name = _first_nonempty(row, ("name",)) or slug
    role = _first_nonempty(row, ("role",)) or "Unknown"
    overview = _first_nonempty(row, ("overview", "bio", "summary"))
    abilities = _first_nonempty(row, ("abilities", "kit"))
    tags = _first_nonempty(row, ("playstyle_tags", "tags"))

    parts = [
        f"{name} is a {role} hero.",
        overview,
        abilities,
        f"Playstyle: {tags}" if tags else "",
    ]
    return " ".join(part for part in parts if part).strip()


def load_heroes(path: Path = DEFAULT_HEROES_PATH) -> list[HeroDoc]:
    """Loads hero records from JSON and returns normalized ``HeroDoc`` objects.

    The loader supports dictionary and list JSON formats. Output is sorted by
    role and name to keep a stable processing order.

    Args:
        path: Path to the heroes JSON dataset.

    Returns:
        A sorted list of normalized hero documents.

    Raises:
        ValueError: If the input JSON root is not a supported type.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        rows = [(slug, row) for slug, row in raw.items() if isinstance(row, dict)]
    elif isinstance(raw, list):
        rows = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            slug = _first_nonempty(row, ("slug",)) or _slugify(_first_nonempty(row, ("name",)))
            rows.append((slug, row))
    else:
        raise ValueError(f"Unsupported heroes format in {path}: {type(raw).__name__}")

    heroes: list[HeroDoc] = []
    for slug, row in rows:
        name = _first_nonempty(row, ("name",)) or slug
        role = _first_nonempty(row, ("role",)).title() or "Unknown"
        text = _combined_text(slug, row)
        heroes.append(HeroDoc(slug=slug, name=name, role=role, text=text))

    heroes.sort(key=lambda h: (h.role, h.name))
    return heroes


def resolve_heroes_by_ref(heroes: list[HeroDoc], refs: list[str]) -> list[HeroDoc]:
    """Resolves hero references by slug or name without duplicates.

    Args:
        heroes: Catalog of available heroes.
        refs: User-provided hero references.

    Returns:
        A list of resolved heroes preserving reference order and uniqueness.
    """

    by_slug = {hero.slug.lower(): hero for hero in heroes}
    by_name = {hero.name.lower(): hero for hero in heroes}

    resolved: list[HeroDoc] = []
    seen: set[str] = set()

    for ref in refs:
        key = ref.strip().lower()
        if not key:
            continue

        hero = by_slug.get(key) or by_name.get(key)
        if hero and hero.slug not in seen:
            resolved.append(hero)
            seen.add(hero.slug)

    return resolved
