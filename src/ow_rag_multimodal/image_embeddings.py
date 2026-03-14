"""CLIP image embedding index with on-disk cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import normalize_rows, normalize_vector
from .models import HeroDoc


@dataclass(frozen=True)
class CLIPImageIndex:
    """CLIP image embedding index with cache metadata validation.

    Encodes hero portrait images with CLIP ViT-B/32 and caches the resulting
    normalized float32 matrix.  Also exposes a text encoder for cross-modal
    query encoding (CLIP text and image vectors share the same 512-dim space).

    Attributes:
        heroes: Hero catalog aligned with the image matrix rows.
        cache_dir: Directory where vectors and metadata are stored.
        images_dir: Directory containing ``{slug}.png`` portrait files.
    """

    heroes: list[HeroDoc]
    cache_dir: Path
    images_dir: Path

    @property
    def vectors_path(self) -> Path:
        """Returns path for cached image vectors."""
        return self.cache_dir / "image_vectors.npy"

    @property
    def meta_path(self) -> Path:
        """Returns path for image index metadata."""
        return self.cache_dir / "image_meta.json"

    def _signature(self) -> str:
        """Computes a content signature based on image file sizes and slugs.

        Uses file size as a lightweight proxy for content change detection.
        Invalidates cache when images are replaced.

        Returns:
            SHA-256 hex digest.
        """
        parts: list[str] = []
        for hero in self.heroes:
            img_path = self.images_dir / f"{hero.slug}.png"
            size = img_path.stat().st_size if img_path.exists() else 0
            parts.append(f"{hero.slug}|{size}")
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    def _is_cache_compatible(self) -> bool:
        """Checks whether cached files match current image inputs."""
        if not self.vectors_path.exists() or not self.meta_path.exists():
            return False
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            meta.get("signature") == self._signature()
            and int(meta.get("hero_count", -1)) == len(self.heroes)
        )

    def _available_heroes(self) -> list[HeroDoc]:
        """Returns heroes whose portrait PNG exists in images_dir."""
        return [h for h in self.heroes if (self.images_dir / f"{h.slug}.png").exists()]

    def build(self, force_refresh: bool = False) -> np.ndarray:
        """Builds image vectors or reuses a compatible cache.

        Heroes with missing images are assigned zero vectors so the matrix
        shape always matches ``len(heroes)``.

        Args:
            force_refresh: When ``True``, ignores cache.

        Returns:
            L2-normalized float32 matrix of shape ``[n_heroes, 512]``.

        Raises:
            RuntimeError: If ``sentence-transformers`` is not installed.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not force_refresh and self._is_cache_compatible():
            cached = np.load(self.vectors_path)
            if cached.shape[0] == len(self.heroes):
                return cached.astype(np.float32)

        try:
            from PIL import Image
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for image embeddings. "
                "Install with: pip install -e '.[multimodal]'"
            ) from exc

        model = SentenceTransformer("clip-ViT-B-32")

        # Build a slug→index map so we can place vectors in the right rows
        slug_to_idx = {hero.slug: i for i, hero in enumerate(self.heroes)}
        n = len(self.heroes)
        matrix = np.zeros((n, 512), dtype=np.float32)

        available = self._available_heroes()
        if not available:
            print("Warning: no hero images found in", self.images_dir)
        else:
            images = [Image.open(self.images_dir / f"{h.slug}.png").convert("RGB") for h in available]
            vecs = model.encode(images, convert_to_numpy=True, show_progress_bar=True)
            for hero, vec in zip(available, vecs):
                matrix[slug_to_idx[hero.slug]] = vec.astype(np.float32)

        matrix = normalize_rows(matrix)
        np.save(self.vectors_path, matrix)
        self._write_meta()
        return matrix

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a text query into CLIP's shared 512-dim embedding space.

        Because CLIP's text and image encoders share the same embedding space,
        this vector can be compared directly against image vectors via dot product.

        Args:
            query: Free-text query string.

        Returns:
            L2-normalized float32 vector of shape ``[512]``.

        Raises:
            RuntimeError: If ``sentence-transformers`` is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for image embeddings. "
                "Install with: pip install -e '.[multimodal]'"
            ) from exc

        model = SentenceTransformer("clip-ViT-B-32")
        vec = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
        return normalize_vector(vec)

    def _write_meta(self) -> None:
        """Writes image index metadata to disk."""
        self.meta_path.write_text(
            json.dumps(
                {
                    "signature": self._signature(),
                    "hero_count": len(self.heroes),
                    "clip_model": "clip-ViT-B-32",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
