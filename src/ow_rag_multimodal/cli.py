"""Command-line interface for the OW RAG recommender."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .recommender import (
    DEFAULT_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    OWRAGMultimodalRecommender,
)


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        A fully configured ``argparse.ArgumentParser`` instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Recomendador de héroes OW con perfil RAG por héroes usados "
            "y embeddings de texto."
        )
    )
    parser.add_argument("--query", default="", help="Descripción de playstyle en lenguaje natural.")
    parser.add_argument(
        "--played",
        nargs="*",
        default=[],
        help="Héroes usados (slug o nombre), por ejemplo: --played ana genji tracer",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Número de recomendaciones.")
    parser.add_argument("--role", help="Filtro opcional de rol: Tank, Damage, Support")
    parser.add_argument(
        "--heroes-path",
        type=Path,
        default=Path("data/heroes.json"),
        help="Ruta al dataset de héroes.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Directorio de caché de embeddings (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Modelo de embedding de texto (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--profile-top-k",
        type=int,
        default=6,
        help="Número de contextos RAG para construir el perfil.",
    )
    parser.add_argument(
        "--include-played",
        action="store_true",
        help="Incluye héroes usados en resultados (por defecto se excluyen).",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Regenera embeddings ignorando caché.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Muestra contexto recuperado por RAG para el perfil.",
    )
    return parser


def _print_profile(show_context: bool, result_profile: object) -> None:
    """Prints the inferred player profile to stdout.

    Args:
        show_context: Whether to print retrieved context snippets.
        result_profile: Profile object returned by the recommender, or ``None``.

    Returns:
        ``None``.
    """

    profile = result_profile
    if profile is None:
        return

    print("\nPerfil RAG")
    print("-" * 50)
    print(profile.summary)
    print(f"Heroes usados: {', '.join(profile.played_heroes)}")
    if profile.dominant_roles:
        print(f"Roles dominantes: {', '.join(profile.dominant_roles)}")
    if profile.signature_traits:
        print(f"Rasgos: {', '.join(profile.signature_traits[:8])}")

    if show_context and profile.retrieved_context:
        print("\nContexto recuperado")
        print("-" * 50)
        for i, ctx in enumerate(profile.retrieved_context, start=1):
            snippet = " ".join(ctx.text.split())[:140]
            print(f"{i}. {ctx.name} [{ctx.role}] ({ctx.score}%) :: {snippet}...")


def main() -> int:
    """Executes the CLI workflow.

    Returns:
        Process-like exit code where ``0`` means success and ``1`` means error.
    """

    args = build_parser().parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Falta OPENAI_API_KEY en el entorno.")
        return 1

    if not args.query.strip() and not args.played:
        print("Debes pasar --query o al menos un héroe en --played.")
        return 1

    try:
        recommender = OWRAGMultimodalRecommender(
            heroes_path=args.heroes_path,
            cache_dir=args.cache_dir,
            embedding_model=args.embedding_model,
            force_refresh_cache=args.refresh_cache,
        )
        result = recommender.recommend(
            query=args.query,
            played_refs=args.played,
            top_k=args.top_k,
            role_filter=args.role,
            profile_top_k=args.profile_top_k,
            exclude_played=not args.include_played,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print("Recomendaciones")
    print("-" * 50)
    for i, item in enumerate(result.recommendations, start=1):
        print(f"{i}. {item.name} [{item.role}] - {item.score}%")

    _print_profile(args.show_context, result.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
