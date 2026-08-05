SHELL := /bin/bash

.PHONY: help sync lock run test style docs clean

help:
	@echo "Commands:"
	@echo "  sync   Create/update .venv from uv.lock"
	@echo "  lock   Resolve dependencies and update uv.lock"
	@echo "  run    Launch the Streamlit UI"
	@echo "  test   Run the test suite"
	@echo "  style  Format and lint the codebase"
	@echo "  docs   Preview the documentation"
	@echo "  clean  Remove generated Python and test artifacts"

sync:
	uv sync --locked

lock:
	uv lock

run:
	uv run streamlit run UserInterface/credit_score_app.py

test:
	uv run pytest

style:
	uv run black .
	uv run isort .
	uv run flake8

docs:
	uv run mkdocs serve

clean:
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".DS_Store" \) -delete
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ipynb_checkpoints" \) -prune -exec rm -rf {} +
	rm -f .coverage
