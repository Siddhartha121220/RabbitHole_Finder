UV := uv --directory backend
.PHONY: setup up down lint test pipeline-dev pipeline api web eval

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

pipeline-dev pipeline api web eval:
	@echo "$@: not implemented yet (see docs/TASKS.md)"
