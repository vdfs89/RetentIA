.PHONY: install train run test lint clean run-batch

install:
	uv pip install --system -r requirements.txt

train:
	uv run python -m src.train

run:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

run-batch:
	uv run python scripts/run_batch.py

clean:
	rm -rf models/*.pkl models/*.pt mlflow.db mlruns/ logs/input_samples.jsonl data/processed/*
