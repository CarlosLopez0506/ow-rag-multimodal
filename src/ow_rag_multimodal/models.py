"""Core data models used across the OW RAG recommender."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeroDoc:
    """Represents a hero document used for indexing and retrieval.

    Attributes:
        slug: Stable machine-friendly hero identifier.
        name: Human-readable hero name.
        role: Hero role label, for example ``Tank``, ``Damage``, or ``Support``.
        text: Consolidated descriptive text used for embedding generation.
    """

    slug: str
    name: str
    role: str
    text: str


@dataclass(frozen=True)
class RetrievedContext:
    """Represents one retrieved context item from vector similarity search.

    Attributes:
        slug: Hero identifier for the retrieved item.
        name: Hero display name.
        role: Hero role.
        score: Similarity score in percentage points.
        text: Text payload associated with the retrieved hero.
    """

    slug: str
    name: str
    role: str
    score: float
    text: str


@dataclass(frozen=True)
class PlayerProfile:
    """Stores the inferred player profile produced by the RAG pipeline.

    Attributes:
        played_heroes: Tuple of resolved played hero names.
        dominant_roles: Most represented role labels across played heroes.
        signature_traits: Extracted style terms that characterize the player.
        summary: Human-readable profile summary.
        retrieved_context: Supporting retrieved items used for profile generation.
    """

    played_heroes: tuple[str, ...]
    dominant_roles: tuple[str, ...]
    signature_traits: tuple[str, ...]
    summary: str
    retrieved_context: tuple[RetrievedContext, ...]


@dataclass(frozen=True)
class Recommendation:
    """Represents a ranked recommendation candidate.

    Attributes:
        slug: Hero identifier.
        name: Hero display name.
        role: Hero role.
        score: Ranking score in percentage points.
    """

    slug: str
    name: str
    role: str
    score: float
