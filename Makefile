.PHONY: check lint format typecheck test docs docs-notebooks docs-serve docs-theme

## Run every quality gate (CI runs the same target)
check: lint typecheck test docs

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src

test:
	uv run pytest

## Render executed notebooks to Markdown so Zensical can include them
docs-notebooks:
	@if ls notebooks/*.ipynb >/dev/null 2>&1; then \
		uv run jupyter nbconvert --to markdown --output-dir docs/notebooks notebooks/*.ipynb; \
	fi

## The strict site build, then fail on any Sphinx role left raw in the built pages or a stale
## generated theme (the Pages workflow deploys what this target checks)
docs: docs-notebooks
	uv run zensical build --strict
	uv run python -m football_docs.pages site site
	uv run python -m football_docs.theme docs --check

## Regenerate the committed site theme (docs/stylesheets/theme.css, docs/assets/logo.svg) from viz
docs-theme:
	uv run python -m football_docs.theme docs

docs-serve: docs-notebooks
	uv run zensical serve
