# Decision log

Newest last. Each entry: what we chose, what we rejected, why.

## Phase 0 — Scaffold
- **Chose** a single `backend/pyproject.toml` (uv, `uv_build` backend) with `Makefile` driving everything via `uv --directory backend`. **Rejected** a uv workspace at the repo root: only one Python package exists.
- **Chose** `pgvector/pgvector:pg17` and the official `ghcr.io/mlflow/mlflow` image with a SQLite backend in `./mlflow/` (gitignored). **Rejected** Postgres as MLflow backend: one less moving part locally.
- `make setup` skips `pnpm install` until `frontend/package.json` exists (Phase 8).

## Phase 1 — Ingestion & EDA
- **Feature month = the oldest of the 3 downloaded months (2026-07)**, not the latest. Graph, subset and stats come from it; 2026-08 and 2026-09 are reserved as LTR label months (§4.4). **Rejected** the spec's "most recent month" for the subset: it would leave no future months for leakage-free labels. Spec deviation, noted here.
- **Edge `p` is normalized over link clicks *within the subset*** (rows sum to 1, as the asset check requires). Raw `n` and per-article `onward_clicks` are kept, so Phase 4's absorbing chain can instead normalize by total onward clicks (which treats leaving the subset as leaving) if that proves more honest.
- **DEV subset BFS:** level by level, following each article's top-20 most-clicked links; the overflowing level is trimmed by summed click count. Deterministic and every article is reachable from a seed. **Rejected** global best-first search: it drifts into hubs.
- **Exclusions** (Main Page, disambiguation, lists, year/decade/date and "X in Y" year pages) are applied to BFS edges too, so they never enter the subset.
- **DEV text = lead section only** via `prop=extracts|info` (20 titles per call; `info` gives page byte length for `content_richness`). Chunks in DEV are therefore lead-only. **Rejected** full-page extracts in DEV: the API returns one full extract per request, i.e. ~5,000 requests.
- **Chunk size counted in words, not tokens** (~1.3 tokens/word). Avoids a tokenizer dependency; precise enough for retrieval.
- MediaWiki API: ≥1 s between requests, exponential backoff honouring `Retry-After` on 429/503/maxlag, every response cached to `data/raw/wiki_api/`. We were heavily 429'd with a placeholder contact in `WIKI_USER_AGENT`; it now points at the repo URL.
- FULL-mode HF streaming lives in `ingest/hf_text.py` with pyright in basic mode, because `datasets` lacks complete type hints. Implemented, not yet run.
- Added `marimo` (notebooks as plain `.py`, no stored outputs) and `matplotlib` (charts) as dev dependencies.

## Phase 2 — Graph & communities
- **Resolution chosen by modularity × coherence** among resolutions with median eligible size 20–200. **Rejected** max modularity alone: it always picked resolution 1, giving 23 communities in DEV (≈ one per seed topic, e.g. all of Rome and Byzantium together), which is too coarse for a rabbit hole. Coherence alone favours ever-smaller clusters. DEV result: resolution 4, 46 communities, 43 eligible, median size 68 (`reports/leiden_sweep.md`).
- **Coherence uses TF-IDF of lead texts for now**, not sentence embeddings (the spec's measure), because embeddings arrive in Phase 3. Re-run the sweep and notebook with embeddings then.
- **Central article = highest summed transition probability `p` from other members**, not raw weighted in-degree (spec deviation, spec updated). With raw counts, one huge edge crowned odd articles: "Hans Albert Einstein" for physics (via Albert Einstein's page), "Android 17" for software. With `p`, each member casts at most one vote: Albert Einstein, Linus Torvalds, The Beatles, Cricket, Charles Darwin.
- **c-TF-IDF has no `min_df`**: a word that appears in only one community is the best keyword, and `min_df=2` was silently removing exactly those (caught by a unit test).
- **Leiden vs HDBSCAN** (`notebooks/02_communities.py`): mostly the same topics (AMI ≈ 0.80). HDBSCAN on text wins raw stickiness (0.85 vs 0.71) but discards 43% of articles as noise (28% of views), mostly the bridging ones. On the same articles, Leiden has higher modularity (0.635 vs 0.602). Kept Leiden: it covers every article and rabbit holes are defined by reader movement. An honest "it's closer than expected" result.
- **Pruned serving graph:** top-10 outgoing edges per article by `p`; `cost = −log p + 0.1·log(1 + degree)` so hub pages cost more as stepping stones. Hub exclusion as bridge *endpoints* is left to Phase 5, where bridge paths are built.
- **Pyright:** switched to `typeCheckingMode = "strict"` with only the "unknown type from a library without stubs" rules off (igraph, scipy, scikit-learn and datasets have no complete hints). Everything else stays strict; the per-file `basic` override on `hf_text.py` is gone.
