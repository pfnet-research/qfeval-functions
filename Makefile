PROJECT_NAME := qfeval_functions
RUN := uv run

.PHONY: check
check: test lint

.PHONY: install
install:
	uv sync --dev

.PHONY: test
test: doctest pytest

.PHONY: doctest
doctest:
	@echo "To be implemented"
#	$(RUN) pytest --doctest-modules $(PROJECT_NAME)

.PHONY: pytest
pytest:
	$(RUN) pytest --doctest-modules tests

.PHONY: test-cov
test-cov:
	$(RUN) pytest --cov=$(PROJECT_NAME) --cov-report=xml

.PHONY: test-flakiness
test-flakiness:
	$(RUN) pytest --count=10 tests

.PHONY: lint
lint: lint-black lint-isort flake8 mypy

.PHONY: lint-black
lint-black:
	$(RUN) black --check --diff --quiet .

.PHONY: lint-isort
lint-isort:
	$(RUN) isort --check --quiet .

.PHONY: mypy
mypy:
	$(RUN) mypy .

.PHONY: flake8
flake8:
	$(RUN) pflake8 .

.PHONY: format
format: format-black format-isort

.PHONY: format-black
format-black:
	$(RUN) black --quiet .

.PHONY: format-isort
format-isort:
	$(RUN) isort --quiet .

.PHONY: docs
docs:
	$(RUN) sphinx-build -b html docs docs/_build/html

.PHONY: docs-clean
docs-clean:
	rm -rf docs/_build
