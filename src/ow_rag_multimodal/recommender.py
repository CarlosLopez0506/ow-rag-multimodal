"""Top-level recommendation orchestration using embeddings and RAG profile signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import DEFAULT_HEROES_PATH, VALID_ROLES, load_heroes, resolve_heroes_by_ref
from .embeddings import (
    MultimodalIndex,
    OpenAIEmbeddingClient,
    SentenceTransformerEmbeddingClient,
    normalize_vector,
)
from .image_embeddings import CLIPImageIndex
from .models import PlayerProfile, Recommendation
from .rag import HeroRAG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "images"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class RecommenderResult:
    """Container for recommendation output and optional profile details.

    Attributes:
        recommendations: Ranked recommendation list.
        profile: Inferred player profile when played heroes are provided.
    """

    recommendations: tuple[Recommendation, ...]
    profile: PlayerProfile | None


class OWRAGMultimodalRecommender:
    """Main service that builds the index and computes hero recommendations."""

    def __init__(
        self,
        heroes_path: Path = DEFAULT_HEROES_PATH,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        images_dir: Path = DEFAULT_IMAGES_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        force_refresh_cache: bool = False,
        alpha_image: float = 0.3,
    ) -> None:
        """Initializes OpenAI client, vector index, RAG helper, and optionally CLIP.

        Args:
            heroes_path: Path to heroes dataset JSON.
            cache_dir: Directory for vector cache artifacts.
            images_dir: Directory containing hero portrait PNGs.
            embedding_model: Text embedding model name.
            force_refresh_cache: Whether to force index rebuild.
            alpha_image: Image signal weight in [0, 1]. 0 = text-only (default).

        Raises:
            RuntimeError: If the ``openai`` dependency is not installed.
        """

        self.heroes = load_heroes(heroes_path)
        self.embedding_client = SentenceTransformerEmbeddingClient(
            model_name=embedding_model,
        )

        index = MultimodalIndex(
            heroes=self.heroes,
            cache_dir=cache_dir,
            embedding_client=self.embedding_client,
        )
        self.hero_vectors = index.build(force_refresh=force_refresh_cache)
        self.rag = HeroRAG(
            heroes=self.heroes,
            hero_vectors=self.hero_vectors,
            embedding_client=self.embedding_client,
        )
        self.index_by_slug = {hero.slug: i for i, hero in enumerate(self.heroes)}

        self._alpha_image = alpha_image
        self.clip_index: CLIPImageIndex | None = None
        self.image_vectors: np.ndarray | None = None
        if alpha_image > 0.0:
            self.clip_index = CLIPImageIndex(
                heroes=self.heroes,
                cache_dir=cache_dir,
                images_dir=images_dir,
            )
            self.image_vectors = self.clip_index.build(force_refresh=force_refresh_cache)

    def recommend(
        self,
        query: str,
        played_refs: list[str],
        top_k: int = 5,
        role_filter: str | None = None,
        profile_top_k: int = 6,
        exclude_played: bool = True,
        w_query: float = 0.6,
        w_played: float = 0.05,
        w_context: float = 0.35,
        alpha_image: float | None = None,
    ) -> RecommenderResult:
        """Computes ranked hero recommendations from query and play history.

        Args:
            query: Optional free-text playstyle query.
            played_refs: Played hero references by slug or name.
            top_k: Maximum number of recommendations to return.
            role_filter: Optional role filter.
            profile_top_k: Number of contexts for profile construction.
            exclude_played: Whether to remove already played heroes from results.
            w_query: Weight for the query signal.
            w_played: Weight for the played-heroes centroid signal.
            w_context: Weight for the retrieved-context centroid signal.
            alpha_image: Image signal weight override. Defaults to value set at init.

        Returns:
            A ``RecommenderResult`` with ranked recommendations and optional profile.

        Raises:
            ValueError: If ``top_k`` is invalid, role filter is invalid, or both
                query and played references are empty.
        """

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        normalized_role = role_filter.title() if role_filter else None
        if normalized_role and normalized_role not in VALID_ROLES:
            valid = ", ".join(sorted(VALID_ROLES))
            raise ValueError(f"Role inválido: {role_filter}. Usa uno de: {valid}")

        profile: PlayerProfile | None = None
        played = resolve_heroes_by_ref(self.heroes, played_refs)
        played_set = {hero.slug for hero in played}

        vectors: list[np.ndarray] = []
        weights: list[float] = []

        if query.strip():
            query_vec = normalize_vector(self.embedding_client.embed_texts([query])[0])
            vectors.append(query_vec)
            weights.append(w_query)

        if played:
            profile = self.rag.build_profile(played_refs=played_refs, extra_context=query, top_k=profile_top_k)

            played_indices = [self.index_by_slug[h.slug] for h in played]
            played_vec = normalize_vector(np.mean(self.hero_vectors[played_indices], axis=0))
            vectors.append(played_vec)
            weights.append(w_played)

            if profile.retrieved_context:
                ctx_indices = [self.index_by_slug[item.slug] for item in profile.retrieved_context]
                ctx_vec = normalize_vector(np.mean(self.hero_vectors[ctx_indices], axis=0))
                vectors.append(ctx_vec)
                weights.append(w_context)

        if not vectors:
            raise ValueError("Debes enviar --query o al menos un héroe en --played")

        total = np.zeros_like(vectors[0])
        for weight, vector in zip(weights, vectors, strict=True):
            total += weight * vector
        final_query = normalize_vector(total)

        text_scores = self.hero_vectors @ final_query

        eff_alpha = self._alpha_image if alpha_image is None else alpha_image
        if eff_alpha > 0.0 and self.clip_index is not None and self.image_vectors is not None:
            image_query = self.clip_index.encode_query(query if query.strip() else " ")
            image_scores = self.image_vectors @ image_query
            scores = (1.0 - eff_alpha) * text_scores + eff_alpha * image_scores
        else:
            scores = text_scores

        ranked = np.argsort(scores)[::-1]

        recommendations: list[Recommendation] = []
        for idx in ranked:
            hero = self.heroes[int(idx)]
            if normalized_role and hero.role != normalized_role:
                continue
            if exclude_played and hero.slug in played_set:
                continue

            recommendations.append(
                Recommendation(
                    slug=hero.slug,
                    name=hero.name,
                    role=hero.role,
                    score=round(float(scores[idx]) * 100, 2),
                )
            )
            if len(recommendations) >= top_k:
                break

        return RecommenderResult(recommendations=tuple(recommendations), profile=profile)
