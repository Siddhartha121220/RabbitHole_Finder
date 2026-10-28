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
- MediaWiki API: ≥1 s between requests, exponential backoff honouring `Retry-After` on 429/503/maxlag, every response cached to `data/raw/wiki_api/`. We were heavily 429'd with the placeholder contact in `WIKI_USER_AGENT`; set a real contact in `.env`.
- FULL-mode HF streaming lives in `ingest/hf_text.py` with pyright in basic mode, because `datasets` lacks complete type hints. Implemented, not yet run.
- Added `marimo` (notebooks as plain `.py`, no stored outputs) and `matplotlib` (charts) as dev dependencies.
