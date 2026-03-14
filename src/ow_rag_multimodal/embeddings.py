"""Embedding generation, normalization, and on-disk index caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import HeroDoc


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Applies L2 normalization row-wise to a matrix.

    Args:
        matrix: Input matrix where each row is a vector.

    Returns:
        A float32 matrix with normalized rows. Zero-norm rows are preserved.
    """

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return (matrix / safe).astype(np.float32)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Applies L2 normalization to a vector.

    Args:
        vector: Input vector.

    Returns:
        A float32 normalized vector. Zero vectors are returned unchanged.
    """

    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class OpenAIEmbeddingClient:
    """Minimal batch embedding client around the OpenAI embeddings API."""

    def __init__(self, client: object, text_model: str, batch_size: int = 32) -> None:
        """Initializes the embedding client.

        Args:
            client: OpenAI client instance exposing ``embeddings.create``.
            text_model: Embedding model name.
            batch_size: Number of texts per API call.
        """

        self.client = client
        self.text_model = text_model
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generates embeddings for a list of texts.

        Args:
            texts: Input texts to embed.

        Returns:
            A 2D float32 array with one row per input text.
        """

        vectors: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(model=self.text_model, input=batch)
            vectors.append(np.array([row.embedding for row in response.data], dtype=np.float32))
        return np.vstack(vectors)


@dataclass(frozen=True)
class MultimodalIndex:
    """Vector index manager with cache metadata validation.

    Attributes:
        heroes: Hero documents to index.
        cache_dir: Directory where vectors and metadata are stored.
        embedding_client: Embedding provider used to build vectors.
    """

    heroes: list[HeroDoc]
    cache_dir: Path
    embedding_client: OpenAIEmbeddingClient

    @property
    def vectors_path(self) -> Path:
        """Returns the path for serialized hero vectors.

        Returns:
            Path to ``hero_vectors.npy``.
        """

        return self.cache_dir / "hero_vectors.npy"

    @property
    def meta_path(self) -> Path:
        """Returns the path for index metadata.

        Returns:
            Path to ``index_meta.json``.
        """

        return self.cache_dir / "index_meta.json"

    def _signature(self) -> str:
        """Computes a stable content signature for cache validation.

        Returns:
            SHA-256 hex digest derived from hero slugs and text payloads.
        """

        payload = "\n".join(f"{h.slug}|{h.text}" for h in self.heroes)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _is_cache_compatible(self) -> bool:
        """Checks whether existing cached files are valid for current inputs.

        Returns:
            ``True`` when cache metadata matches current signature, model, and hero count.
            Otherwise ``False``.
        """

        if not self.vectors_path.exists() or not self.meta_path.exists():
            return False

        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        return (
            meta.get("signature") == self._signature()
            and meta.get("text_model") == self.embedding_client.text_model
            and int(meta.get("hero_count", -1)) == len(self.heroes)
        )

    def build(self, force_refresh: bool = False) -> np.ndarray:
        """Builds vectors or reuses a compatible cache.

        Args:
            force_refresh: When ``True``, bypasses cache compatibility checks.

        Returns:
            A normalized float32 matrix of hero vectors.
        """

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not force_refresh and self._is_cache_compatible():
            cached = np.load(self.vectors_path)
            if cached.shape[0] == len(self.heroes):
                return normalize_rows(cached)

        text_vectors = self.embedding_client.embed_texts([hero.text for hero in self.heroes])
        fused = normalize_rows(text_vectors)
        np.save(self.vectors_path, fused)
        self._write_meta()
        return fused

    def _write_meta(self) -> None:
        """Writes index metadata to disk.

        Returns:
            ``None``.
        """

        self.meta_path.write_text(
            json.dumps(
                {
                    "signature": self._signature(),
                    "hero_count": len(self.heroes),
                    "text_model": self.embedding_client.text_model,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
