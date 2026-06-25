.PHONY: install dev format lint test serve docker

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

format:
	black pdf_table_extractor tests

lint:
	ruff check pdf_table_extractor tests
	black --check pdf_table_extractor tests

test:
	pytest

serve:
	uvicorn pdf_table_extractor.api:app --reload

docker:
	docker build -t pdf-table-extractor .
