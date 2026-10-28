# Decision log

Newest last. Each entry: what we chose, what we rejected, why.

## Phase 0 — Scaffold
- **Chose** a single `backend/pyproject.toml` (uv, `uv_build` backend) with `Makefile` driving everything via `uv --directory backend`. **Rejected** a uv workspace at the repo root: only one Python package exists.
- **Chose** `pgvector/pgvector:pg17` and the official `ghcr.io/mlflow/mlflow` image with a SQLite backend in `./mlflow/` (gitignored). **Rejected** Postgres as MLflow backend: one less moving part locally.
- `make setup` skips `pnpm install` until `frontend/package.json` exists (Phase 8).
