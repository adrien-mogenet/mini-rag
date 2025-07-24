.PHONY: help install install-dev test test-cov clean lint format check build upload

help:		## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:	## Install the package
	pip install -e .

install-dev:	## Install development dependencies
	pip install -e .
	pip install -r requirements-dev.txt

test:		## Run tests
	pytest

test-cov:	## Run tests with coverage
	pytest --cov=mini_rag --cov-report=term-missing --cov-report=html

lint:		## Run linting
	flake8 mini_rag tests
	isort --check-only mini_rag tests

format:		## Format code
	black mini_rag tests
	isort mini_rag tests

check:		## Run all checks (lint + test)
	make lint
	make test

clean:		## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:		## Build the package
	python setup.py sdist bdist_wheel

upload:		## Upload to PyPI (requires twine)
	twine upload dist/*

example:	## Run the example script
	python example.py