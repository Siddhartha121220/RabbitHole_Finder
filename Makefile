UV := uv --directory backend
.PHONY: setup up down lint test pipeline-dev pipeline dagster-ui api web eval

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

DAGSTER := mkdir -p .dagster && DAGSTER_HOME=$(CURDIR)/.dagster $(UV) run dagster asset materialize \
	-m rabbithole.pipelines.definitions --select '*'

pipeline-dev:
	DATASET_MODE=dev $(DAGSTER)

pipeline:
	DATASET_MODE=full $(DAGSTER)

dagster-ui:
	mkdir -p .dagster && DAGSTER_HOME=$(CURDIR)/.dagster $(UV) run dagster dev -m rabbithole.pipelines.definitions

api web eval:
	@echo "$@: not implemented yet (see docs/TASKS.md)"
