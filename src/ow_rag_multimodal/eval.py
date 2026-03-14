"""Offline evaluation protocols for the OW RAG recommender."""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import load_heroes
from .models import HeroDoc
from .recommender import (
    DEFAULT_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HEROES_PATH,
    OWRAGMultimodalRecommender,
)


# ---------------------------------------------------------------------------
# Protocol 1 — Self-retrieval
# ---------------------------------------------------------------------------

def run_self_retrieval(
    recommender: OWRAGMultimodalRecommender,
    heroes: list[HeroDoc],
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Evaluates embedding quality by querying each hero with its own text.

    For every hero the recommender is called with that hero's ``text`` as the
    query and no played references.  A perfect embedding space places the hero
    at rank 1 every time.

    Args:
        recommender: Initialized recommender instance.
        heroes: Full hero catalog.
        top_k: Maximum K for hit-rate computation.

    Returns:
        Nested dict ``{role: {"hit@1": float, "hit@3": float, "hit@K": float},
        "overall": {...}}``.
    """

    ks = sorted({1, 3, top_k})
    counters: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)

    for hero in heroes:
        result = recommender.recommend(
            query=hero.text,
            played_refs=[],
            top_k=max(ks),
            exclude_played=False,
        )
        slugs = [r.slug for r in result.recommendations]
        for k in ks:
            hit = int(hero.slug in slugs[:k])
            counters[hero.role][k] += hit
            counters["Overall"][k] += hit
        totals[hero.role] += 1
        totals["Overall"] += 1

    return _build_rate_dict(counters, totals, ks)


# ---------------------------------------------------------------------------
# Protocol 2 — Leave-one-out by role
# ---------------------------------------------------------------------------

def run_leave_one_out(
    recommender: OWRAGMultimodalRecommender,
    heroes: list[HeroDoc],
    top_k: int,
    n_played: int,
    rng: random.Random,
    w_played: float = 0.3,
    w_context: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Evaluates the RAG profile fusion via leave-one-out within each role.

    For every hero H, ``n_played`` heroes from the same role are sampled as
    played references and H is excluded from recommendations.  Hit-rate
    measures how often the model recovers H.

    Args:
        recommender: Initialized recommender instance.
        heroes: Full hero catalog.
        top_k: Maximum K for hit-rate computation.
        n_played: Number of same-role heroes to use as played references.
        rng: Seeded random instance for reproducibility.
        w_played: Weight for the played-heroes centroid signal.
        w_context: Weight for the retrieved-context centroid signal.

    Returns:
        Nested dict ``{role: {"hit@3": float, "hit@K": float}, "overall": {...}}``.
    """

    ks = sorted({3, top_k})
    by_role: dict[str, list[HeroDoc]] = defaultdict(list)
    for hero in heroes:
        by_role[hero.role].append(hero)

    counters: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)

    for role, group in by_role.items():
        if len(group) < n_played + 1:
            continue
        for hero in group:
            pool = [h for h in group if h.slug != hero.slug]
            played = [h.slug for h in rng.sample(pool, min(n_played, len(pool)))]
            result = recommender.recommend(
                query="",
                played_refs=played,
                top_k=max(ks),
                exclude_played=True,
                w_played=w_played,
                w_context=w_context,
            )
            slugs = [r.slug for r in result.recommendations]
            for k in ks:
                hit = int(hero.slug in slugs[:k])
                counters[role][k] += hit
                counters["Overall"][k] += hit
            totals[role] += 1
            totals["Overall"] += 1

    return _build_rate_dict(counters, totals, ks)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(
    recommender: OWRAGMultimodalRecommender,
    heroes: list[HeroDoc],
    top_k: int,
    n_played: int,
    rng_seed: int,
    mode: str = "loo",
    step: float = 0.1,
) -> tuple[list[dict], dict[str, float]]:
    """Sweeps fusion weights and returns the full results table and best weights.

    LOO mode sweeps a single parameter alpha = w_played / (w_played + w_context)
    over [step, 1-step].  Since normalize_vector is applied after fusion, only
    this ratio affects ranking when query is empty.

    Full mode sweeps (w_query, w_played) on a 2-D grid with step 0.2 and
    constraint w_query + w_played <= 1.0.  Each combination uses the hero's own
    text as the query alongside 2 sampled same-role heroes as played refs.

    Args:
        recommender: Initialized recommender instance.
        heroes: Full hero catalog.
        top_k: K for hit@K objective.
        n_played: Played refs per hero.
        rng_seed: Seed for reproducibility.
        mode: "loo" for 1-D alpha sweep, "full" for 2-D (w_query, w_played) grid.
        step: Grid step size.

    Returns:
        Tuple of (rows list, best_weights dict).
    """

    rows: list[dict] = []
    best_score = -1.0
    best_weights: dict[str, float] = {}

    if mode == "loo":
        alphas = np.round(np.arange(step, 1.0, step), 6).tolist()
        for alpha in alphas:
            w_p = round(alpha, 6)
            w_c = round(1.0 - alpha, 6)
            rng = random.Random(rng_seed)
            result = run_leave_one_out(
                recommender, heroes, top_k, n_played, rng,
                w_played=w_p, w_context=w_c,
            )
            score = result["Overall"].get(f"hit@{top_k}", 0.0)
            hit3 = result["Overall"].get("hit@3", 0.0)
            rows.append({"alpha": w_p, "w_context": w_c, "hit@3": hit3, f"hit@{top_k}": score})
            if score > best_score:
                best_score = score
                best_weights = {"w_played": w_p, "w_context": w_c, f"hit@{top_k}": score}

    else:  # full mode
        grid_step = 0.2
        candidates = np.round(np.arange(grid_step, 1.0, grid_step), 6).tolist()
        by_role: dict[str, list[HeroDoc]] = defaultdict(list)
        for hero in heroes:
            by_role[hero.role].append(hero)

        for w_q in candidates:
            for w_p in candidates:
                if w_q + w_p > 1.0 + 1e-9:
                    continue
                w_c = round(1.0 - w_q - w_p, 6)
                if w_c < 0:
                    w_c = 0.0

                rng = random.Random(rng_seed)
                counters: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
                totals: dict[str, int] = defaultdict(int)
                ks = sorted({3, top_k})

                for role, group in by_role.items():
                    if len(group) < n_played + 1:
                        continue
                    for hero in group:
                        pool = [h for h in group if h.slug != hero.slug]
                        played = [h.slug for h in rng.sample(pool, min(n_played, len(pool)))]
                        res = recommender.recommend(
                            query=hero.text,
                            played_refs=played,
                            top_k=max(ks),
                            exclude_played=True,
                            w_query=w_q,
                            w_played=w_p,
                            w_context=w_c,
                        )
                        slugs = [r.slug for r in res.recommendations]
                        for k in ks:
                            hit = int(hero.slug in slugs[:k])
                            counters[role][k] += hit
                            counters["Overall"][k] += hit
                        totals[role] += 1
                        totals["Overall"] += 1

                total = totals.get("Overall", 0)
                score = round(counters["Overall"][top_k] / total, 4) if total else 0.0
                hit3 = round(counters["Overall"][3] / total, 4) if total else 0.0
                rows.append({"w_query": w_q, "w_played": w_p, "w_context": w_c,
                             "hit@3": hit3, f"hit@{top_k}": score})
                if score > best_score:
                    best_score = score
                    best_weights = {"w_query": w_q, "w_played": w_p, "w_context": w_c,
                                    f"hit@{top_k}": score}

    return rows, best_weights


def _print_grid_table(title: str, rows: list[dict], top_k: int) -> None:
    if not rows:
        print("No results.")
        return

    keys = list(rows[0].keys())
    col_w = 12
    header = "".join(f"{k:>{col_w}}" for k in keys)
    sep = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print("".join(f"{row[k]:>{col_w}.4f}" for k in keys))
    print(sep)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_rate_dict(
    counters: dict[str, dict[int, int]],
    totals: dict[str, int],
    ks: list[int],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for group, total in totals.items():
        out[group] = {
            f"hit@{k}": round(counters[group][k] / total, 4) if total else 0.0
            for k in ks
        }
    return out


def _print_table(title: str, results: dict[str, dict[str, float]]) -> None:
    roles = sorted(r for r in results if r != "Overall") + ["Overall"]
    metrics = list(next(iter(results.values())).keys())

    col_w = 10
    header = f"{'Role':<12}" + "".join(f"{m:>{col_w}}" for m in metrics)
    sep = "-" * len(header)

    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for role in roles:
        row = f"{role:<12}" + "".join(
            f"{results[role].get(m, 0.0):>{col_w}.4f}" for m in metrics
        )
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline hit-rate evaluation for the OW RAG recommender."
    )
    parser.add_argument(
        "--protocols",
        choices=["self", "loo", "both"],
        default="both",
        help="Protocols to run: self-retrieval, leave-one-out, or both (default: both)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="K for hit@K (default: 5)")
    parser.add_argument(
        "--n-played",
        type=int,
        default=2,
        help="Played refs per hero in leave-one-out (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--heroes-path", type=Path, default=DEFAULT_HEROES_PATH, help="Path to heroes.json"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Embedding cache directory"
    )
    parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="OpenAI embedding model"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run weight grid search instead of standard protocols",
    )
    parser.add_argument(
        "--tune-mode",
        choices=["loo", "full"],
        default="loo",
        help="Grid mode: loo (1-D alpha sweep) or full (2-D w_query/w_played grid) (default: loo)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Grid step size for --tune (default: 0.1)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        return 1

    print("Loading heroes and building index...")
    heroes = load_heroes(args.heroes_path)
    recommender = OWRAGMultimodalRecommender(
        heroes_path=args.heroes_path,
        cache_dir=args.cache_dir,
        embedding_model=args.embedding_model,
    )

    if args.tune:
        print(f"Running weight grid search (mode={args.tune_mode}, step={args.step})...")
        rows, best = run_grid_search(
            recommender, heroes, args.top_k, args.n_played,
            rng_seed=args.seed, mode=args.tune_mode, step=args.step,
        )
        _print_grid_table(
            f"Weight tuning ({args.tune_mode} mode, step={args.step})", rows, args.top_k
        )
        print(f"\nBest weights: {best}")
        return 0

    rng = random.Random(args.seed)

    if args.protocols in ("self", "both"):
        print(f"Running self-retrieval on {len(heroes)} heroes...")
        sr = run_self_retrieval(recommender, heroes, args.top_k)
        _print_table("Self-retrieval", sr)

    if args.protocols in ("loo", "both"):
        print(f"\nRunning leave-one-out (n_played={args.n_played}, seed={args.seed})...")
        loo = run_leave_one_out(recommender, heroes, args.top_k, args.n_played, rng)
        _print_table(f"Leave-one-out (n_played={args.n_played}, seed={args.seed})", loo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
