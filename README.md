# OW RAG Multimodal

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-412991?logo=openai&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?logo=gradio&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Vector%20Math-013243?logo=numpy&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented-22C55E)

Overwatch hero recommender built on **semantic embeddings + RAG**. Given a playstyle description and your hero history, it returns the best hero picks ranked by vector similarity.

---

## How it works

The engine operates entirely in normalized vector space. There are no keyword rules, no hardcoded scores.

1. **Embed** every hero's description using OpenAI `text-embedding-3-small`.
2. **Cache** the resulting matrix on disk with SHA-256 content validation — zero unnecessary API calls on re-runs.
3. **Encode the query** into the same vector space.
4. **Build a RAG player profile** from your previously played heroes: retrieve semantically similar context, extract style traits, and produce a summary.
5. **Combine three signals** with fixed weights:
   - Query vector — `0.6`
   - Played-heroes centroid — `0.3`
   - Retrieved-context centroid — `0.1`
6. **Rank** all heroes by cosine similarity (dot product on L2-normalized vectors) and return the top-K, optionally filtered by role.

```
flowchart TD
    A[UI / CLI] --> B[Query + Played Heroes]
    B --> C[Embedding Index]
    C --> D[hero_vectors normalized]
    B --> E[RAG Profile Builder]
    E --> F[Retrieved Context]
    D & F --> G[Weighted Combination]
    G --> H[Cosine Score + Role Filter]
    H --> I[Top-K Recommendations]
```

---

## Project structure

```
ow-rag-multimodal/
├── data/
│   ├── heroes.json           # Hero catalog with role and description
│   ├── player_history.json   # Persisted play counts per hero
│   └── cache/                # .npy vectors + metadata (gitignored)
├── src/ow_rag_multimodal/
│   ├── cli.py                # Argparse CLI entry point
│   ├── data.py               # Hero loading and fuzzy-name resolution
│   ├── embeddings.py         # OpenAI client, L2 normalization, disk cache
│   ├── history.py            # Play history persistence
│   ├── models.py             # Dataclasses: HeroDoc, Recommendation, PlayerProfile
│   ├── rag.py                # HeroRAG: retrieval + profile synthesis
│   ├── recommender.py        # OWRAGMultimodalRecommender: full pipeline
│   └── ui.py                 # Gradio web interface
├── docs/
│   ├── DIAGRAMA_FLUJO_RAG.md
│   └── PRESENTACION_OW_RAG.md
├── .env.example
└── pyproject.toml
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env  # add your OPENAI_API_KEY
```

---

## Usage

### CLI — query only

```bash
ow-rag --query "aggressive frontline, high pressure" --top-k 5
```

### CLI — query + play history

```bash
ow-rag \
  --query "sustain pressure and enable teammates" \
  --played reinhardt zarya \
  --top-k 5 \
  --show-context
```

`--played` accepts slugs or hero names (case-insensitive fuzzy match).

### Web UI

```bash
ow-rag-ui --host 127.0.0.1 --port 7860
```

Select heroes → describe your playstyle → hit **Recomendar**.
Selections are saved automatically to `player_history.json` and fed back into the RAG profile on future runs.

---

## Key design choices

| Decision | Reason |
|---|---|
| L2-normalized dot product instead of raw cosine | Equivalent math, faster batch scoring with `matrix @ vector` |
| SHA-256 cache signature | Detects catalog changes without storing timestamps |
| Weighted centroid fusion | Simpler and faster than late fusion or re-ranking |
| Gradio stateful callbacks | Keeps the recommender object alive across requests — one index build per session |

---

## Requirements

- Python 3.10+
- `openai >= 1.40`
- `numpy >= 1.26`
- `gradio >= 5.0`
