"""RAG retrieval and player-profile synthesis for hero recommendations."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .data import resolve_heroes_by_ref
from .embeddings import OpenAIEmbeddingClient, normalize_vector
from .models import HeroDoc, PlayerProfile, RetrievedContext

_STOPWORDS = {
    "with",
    "that",
    "this",
    "from",
    "hero",
    "playstyle",
    "their",
    "they",
    "into",
    "while",
    "uses",
    "using",
    "and",
    "the",
    "for",
    "are",
    "who",
    "can",
    "his",
    "her",
    "your",
    "you",
}


class HeroRAG:
    """Performs similarity retrieval and profile construction over hero vectors.

    Attributes:
        heroes: Hero catalog used for retrieval.
        hero_vectors: Normalized hero embedding matrix.
        embedding_client: Embedding provider used for query vectors.
        by_slug: Index mapping from slug to hero document.
        index_by_slug: Index mapping from slug to row position in ``hero_vectors``.
    """

    def __init__(
        self,
        heroes: list[HeroDoc],
        hero_vectors: np.ndarray,
        embedding_client: OpenAIEmbeddingClient,
    ) -> None:
        """Initializes the RAG helper.

        Args:
            heroes: Hero catalog.
            hero_vectors: Embedding matrix aligned with ``heroes``.
            embedding_client: Embedding provider for query encoding.
        """

        self.heroes = heroes
        self.hero_vectors = hero_vectors
        self.embedding_client = embedding_client
        self.by_slug = {hero.slug: hero for hero in heroes}
        self.index_by_slug = {hero.slug: i for i, hero in enumerate(heroes)}

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        exclude_slugs: set[str] | None = None,
    ) -> tuple[RetrievedContext, ...]:
        """Retrieves top-k hero contexts similar to the query.

        Args:
            query: Free-text query used to retrieve context.
            top_k: Maximum number of context items to return.
            exclude_slugs: Optional set of hero slugs to exclude.

        Returns:
            A tuple of retrieved context items ordered by similarity.
        """

        exclude = exclude_slugs or set()
        query_vec = normalize_vector(self.embedding_client.embed_texts([query])[0])
        scores = self.hero_vectors @ query_vec
        ranked = np.argsort(scores)[::-1]

        out: list[RetrievedContext] = []
        for idx in ranked:
            hero = self.heroes[int(idx)]
            if hero.slug in exclude:
                continue
            out.append(
                RetrievedContext(
                    slug=hero.slug,
                    name=hero.name,
                    role=hero.role,
                    score=round(float(scores[idx]) * 100, 2),
                    text=hero.text,
                )
            )
            if len(out) >= top_k:
                break
        return tuple(out)

    def build_profile(
        self,
        played_refs: list[str],
        extra_context: str = "",
        top_k: int = 6,
    ) -> PlayerProfile:
        """Builds a player profile from played heroes and optional text context.

        Args:
            played_refs: Hero references as slugs or names.
            extra_context: Additional free text to enrich retrieval.
            top_k: Number of context items used to infer profile traits.

        Returns:
            A ``PlayerProfile`` inferred from played heroes and retrieved context.

        Raises:
            ValueError: If no played heroes can be resolved from ``played_refs``.
        """

        played = resolve_heroes_by_ref(self.heroes, played_refs)
        if not played:
            raise ValueError("No se pudieron resolver héroes usados en --played.")

        played_slugs = {hero.slug for hero in played}
        query_parts = [hero.text for hero in played]
        if extra_context.strip():
            query_parts.append(extra_context.strip())
        query = "\n".join(query_parts)

        retrieved = self.retrieve(query=query, top_k=top_k, exclude_slugs=played_slugs)
        roles = Counter(hero.role for hero in played)
        dominant_roles = tuple(role for role, _ in roles.most_common(2))

        trait_counter = Counter(self._extract_terms(" ".join([query] + [c.text for c in retrieved])))
        signature_traits = tuple(term for term, _ in trait_counter.most_common(8))

        summary = self._build_summary(played=played, roles=roles, traits=signature_traits)

        return PlayerProfile(
            played_heroes=tuple(hero.name for hero in played),
            dominant_roles=dominant_roles,
            signature_traits=signature_traits,
            summary=summary,
            retrieved_context=retrieved,
        )

    @staticmethod
    def _extract_terms(text: str) -> list[str]:
        """Extracts candidate style terms from text.

        Args:
            text: Source text for token extraction.

        Returns:
            Filtered list of lowercase terms excluding configured stopwords.
        """

        terms = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower())
        return [term for term in terms if term not in _STOPWORDS]

    @staticmethod
    def _build_summary(played: list[HeroDoc], roles: Counter[str], traits: tuple[str, ...]) -> str:
        """Builds a compact natural-language summary for a player profile.

        Args:
            played: Resolved played hero documents.
            roles: Role frequency counts for played heroes.
            traits: Ranked signature terms.

        Returns:
            A human-readable profile summary sentence.
        """

        played_names = ", ".join(hero.name for hero in played)
        dominant = ", ".join(role for role, _ in roles.most_common(2))
        trait_text = ", ".join(traits[:5]) if traits else "adaptable"
        return (
            f"Perfil inferido por RAG a partir de {played_names}. "
            f"Roles dominantes: {dominant}. "
            f"Rasgos de estilo detectados: {trait_text}."
        )
