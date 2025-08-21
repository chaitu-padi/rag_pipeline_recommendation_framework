.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available commands:"
	@echo "make install          Install project dependencies"
	@echo "make install-dev      Install development dependencies"
	@echo "make test            Run tests"
	@echo "make lint            Run linters"
	@echo "make format          Format code"
	@echo "make clean           Clean build artifacts"

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: install-dev
install-dev:
	pip install -r requirements-dev.txt

.PHONY: test
test:
	pytest

.PHONY: lint
lint:
	flake8 rag_recommender
	mypy rag_recommender
	black --check rag_recommender
	isort --check-only rag_recommender

.PHONY: format
format:
	black rag_recommender
	isort rag_recommender

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
