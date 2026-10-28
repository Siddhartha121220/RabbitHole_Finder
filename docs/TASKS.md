# TASKS — Phased Build Plan

Rules: one phase at a time; DEV dataset first; finish each phase with `make lint && make test`, tick the boxes, add a DECISIONS.md entry. Each phase should end in a commit (or a few small ones).

Rough timeline for a student working part-time with Claude Code: 5–7 weeks.

---

## Phase 0 — Scaffold & tooling
- [x] Create repo layout from ARCHITECTURE.md §3; `git init`; `.gitignore` (data/, mlruns/, .env, caches, node_modules, .venv).
- [x] `backend/pyproject.toml` with uv; ruff + pyright config; pytest config.
- [x] `rabbithole.config` (pydantic-settings) reading all env vars from CLAUDE.md; `.env.example`.
- [x] `docker-compose.yml`: postgres with pgvector (official `pgvector/pgvector` image), mlflow server (sqlite backend, local artifact dir).
- [x] `Makefile` with every command in CLAUDE.md (stubs allowed for later phases, printing "not implemented yet").
- [x] pre-commit (ruff, ruff-format, end-of-file, trailing whitespace, no large files).
- [x] GitHub Actions: backend lint + types + tests; frontend job added in Phase 8.
- [x] Empty `docs/DECISIONS.md` and `docs/RESULTS.md` with headings.
- [x] Minimal README stub.

**Acceptance:** `make setup && make up && make lint && make test` all succeed on a clean clone; one trivial test passes; CI green.

## Phase 1 — Ingestion & EDA
- [ ] Clickstream: list available months, download (with size check), parse with DuckDB → `clickstream_<month>.parquet`.
- [ ] Title normalization + `article_id` mapping; exclusion rules (DATA_AND_ML §1.3).
- [ ] DEV subset builder (deterministic BFS from `dev_seeds.py`); FULL subset with threshold.
- [ ] Per-article stats and edges tables (§1.1).
- [ ] Article text: DEV via MediaWiki API (cached, polite); FULL via HF streaming (implement, run later).
- [ ] Chunking (≈ 300 tokens, 50 overlap, keep section titles).
- [ ] Dagster assets for all of the above; `make pipeline-dev` materializes them.
- [ ] Data quality checks (Dagster asset checks): no null titles, n > 0, p rows sum to ~1, text match rate ≥ 90%.
- [ ] Notebook `notebooks/01_eda.*`: distributions (views, onward_ratio, degree), top transitions, entry sources share. Save key charts to `reports/figures/`.
- [ ] Tests on a tiny fixture clickstream (≈ 30 rows).

**Acceptance:** DEV parquet files exist with logged row counts; asset checks pass; EDA notebook runs top to bottom.

## Phase 2 — Graph & communities
- [ ] Undirected weighted graph; Leiden with resolution sweep; metrics table (§2.3).
- [ ] Community labeling via c-TF-IDF + central article.
- [ ] `reports/communities_sample.md` (20 random communities).
- [ ] Pruned directed graph for serving (top-k out edges, hub cost penalty).
- [ ] Notebook `02_communities`: Leiden vs HDBSCAN comparison.
- [ ] Tests: Leiden deterministic with fixed seed on fixture graph; labeling picks central article.

**Acceptance:** sample communities look coherent (show the user `communities_sample.md` and ask for a thumbs up before continuing).

## Phase 3 — Embeddings, vector DB & map
- [ ] Embed article leads and chunks with `EMBEDDING_MODEL`; batch, show progress, resume on crash (write in shards).
- [ ] Optional GPU export/import path (Parquet in/out) documented in README.
- [ ] SQLAlchemy models + Alembic migrations for all tables; HNSW indexes.
- [ ] Loader asset: Parquet → Postgres (idempotent upsert, records `pipeline_runs`).
- [ ] UMAP 2D for articles and community centroids; stored in DB.
- [ ] `/seeds/search` logic as a library function with tests.

**Acceptance:** semantic search for "mechanical keyboard", "Byzantine", "cricket" returns sensible articles in DEV; DB loaded via `make pipeline-dev`.

## Phase 4 — Rabbit-holeness score
- [ ] Feature computation per community (§3.1); absorbing chain with sparse solver (§3.2) + hand-computed unit test.
- [ ] Score + percentile + breakdown (§3.3); weights in config.
- [ ] `reports/rh_extremes.md`; feature correlation heatmap.
- [ ] Labeling CLI: `uv run rabbithole label-communities --n 100` (saves to `data/labels/rh_labels.csv`).
- [ ] Notebook `03_rh_validation`: Spearman vs labels (when labels exist), ridge-learned weights comparison.

**Acceptance:** extremes report looks right to the user; tests pass. Ask the user to do the 100 labels (≈ 20 minutes) — continue without blocking if they postpone.

## Phase 5 — Recommender v1 + evaluation harness
- [ ] Seeds from free text (§4.1); candidate features (§4.2) with shortlist for latency.
- [ ] Heuristic ranker with Goldilocks band + MMR (§4.3).
- [ ] Bridge path explanation (§4.5).
- [ ] Baselines: random, popularity, nearest neighbour.
- [ ] Ground-truth labels from future months (§4.4 label definition) — requires ≥ 2 clickstream months; download them now.
- [ ] Eval runner `make eval` with all metrics, bootstrap CIs, MLflow logging, report + charts (§6).
- [ ] Unit tests for metrics (NDCG, novelty, diversity on toy inputs with known answers).

**Acceptance:** `make eval` produces `reports/eval_<date>.md`; heuristic v1 compared against baselines with CIs. It is fine if v1 does not win on every metric; write down why.

## Phase 6 — API
- [ ] FastAPI app with routers and Pydantic schemas (ARCHITECTURE §5).
- [ ] Startup: load pruned graph + model artifacts; `/health` reports readiness.
- [ ] Structured logging with request id + latency; CORS; rate limit on trail endpoint.
- [ ] Recommendation request log (no personal data).
- [ ] API tests with `httpx` test client against a test DB loaded from fixtures (use testcontainers or the compose DB with a separate schema).
- [ ] Latency check script: p50/p95 for `/recommend` over 100 random queries.

**Acceptance:** all endpoints documented in `/docs`; `/recommend` p95 < 800 ms on DEV locally.

## Phase 7 — RAG trail
- [ ] LLM provider interface; Anthropic provider with structured JSON output + Pydantic validation; template provider.
- [ ] Retrieval + ordering + generation + validation + caching (§5).
- [ ] `/rabbit-holes/{id}/trail` wired up.
- [ ] Trail eval: link validity test, LLM-as-judge faithfulness script, 30-trail hand-check sheet (`reports/trail_check.csv`).
- [ ] Cost guard: log tokens per trail; refuse to run batch judge > 200 trails without user confirmation.

**Acceptance:** trails work with and without `ANTHROPIC_API_KEY`; link validity 100% in tests; faithfulness numbers in RESULTS.md.

## Phase 8 — Frontend
- [ ] Next.js app, TypeScript strict, Tailwind, typed API client generated from OpenAPI (`openapi-typescript`).
- [ ] Pages and components from ARCHITECTURE §7; distinctive visual direction; light/dark; mobile.
- [ ] Map with deck.gl; seeds highlighted; click a community → detail.
- [ ] Feedback buttons wired; anonymous session id.
- [ ] Frontend CI job: typecheck, lint, build; a couple of Playwright smoke tests.

**Acceptance:** full flow works locally against the DEV API; screenshot set saved to `reports/screens/`.

## Phase 9 — FULL data + LTR v2
- [ ] Run FULL pipeline (ask user before big downloads; offer GPU path for embeddings).
- [ ] Re-tune Leiden resolution for FULL; re-check extremes report.
- [ ] LTR training with temporal + community split (§4.4); MLflow tracking; model registry.
- [ ] Serve v2 behind a config flag; re-run `make eval` comparing all systems and ablations.

**Acceptance:** RESULTS.md has the full comparison table, the accuracy-vs-novelty chart, and an honest discussion.

## Phase 10 — Deploy, study & write-up
- [ ] Dockerfiles for API and web; compose profile for full local stack.
- [ ] Deploy (ask user before creating accounts/resources): DB (Neon/Supabase with pgvector), API (Render/Fly/Railway), web (Vercel). Load FULL data, or a capped subset if the free tier is too small — document the cap.
- [ ] `/study` page + blind v1/v2 assignment (§7).
- [ ] README: hook + demo GIF, architecture diagram (Mermaid), how rabbit-holeness works, results table, what failed, how to run, tech stack, attribution (CC BY-SA).
- [ ] `docs/INTERVIEW_NOTES.md`: 10 likely interview questions about this project with crisp answers (metric design without ground truth, leakage, Leiden vs HDBSCAN, LTR, RAG validation, scaling).

**Acceptance:** public URL works; README renders well on GitHub; study results (if collected) in RESULTS.md.

## Stretch (only after Phase 10)
- [ ] Rabbit hole autopsy (paste URLs → path on map).
- [ ] Personal re-ranking from feedback.
- [ ] "Ask this rabbit hole" scoped RAG chat.
- [ ] Monthly refresh scheduled in Dagster.
