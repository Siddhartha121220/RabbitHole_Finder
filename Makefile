UV := uv --directory backend
.PHONY: load setup up down lint test migrate pipeline-dev pipeline dagster-ui api web eval api-types

setup:
	$(UV) sync
	if [ -f frontend/package.json ]; then pnpm --dir frontend install; fi
	$(UV) run pre-commit install

up:
	docker compose up -d --wait

down:
	docker compose down

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run pyright

test:
	$(UV) run pytest -q

migrate:
	$(UV) run alembic upgrade head

DAGSTER := mkdir -p .dagster && DAGSTER_HOME=$(CURDIR)/.dagster $(UV) run dagster asset materialize \
	-m rabbithole.pipelines.definitions --select '*'

pipeline-dev:
	DATASET_MODE=dev $(DAGSTER)

pipeline:
	DATASET_MODE=full $(DAGSTER)

# Load processed Parquet into Postgres only (e.g. after `rabbithole unpack` of a Kaggle build).
load:
	mkdir -p .dagster && DAGSTER_HOME=$(CURDIR)/.dagster $(UV) run dagster asset materialize \
		-m rabbithole.pipelines.definitions --select postgres

dagster-ui:
	mkdir -p .dagster && DAGSTER_HOME=$(CURDIR)/.dagster $(UV) run dagster dev -m rabbithole.pipelines.definitions

eval:
	$(UV) run rabbithole eval

api:
	$(UV) run uvicorn rabbithole.api.app:app --reload --port 8000

web:
	cd frontend && pnpm dev

# Regenerate frontend/openapi.json and the typed client from the FastAPI schema.
api-types:
	$(UV) run python -c "import json; from rabbithole.api.app import app; print(json.dumps(app.openapi(), indent=1))" > frontend/openapi.json
	cd frontend && pnpm gen:api
