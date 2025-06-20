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
	$(RUN) pytest --doctest-modules qfeval_functions/functions/*.py -v

.PHONY: pytest
pytest:
	$(RUN) pytest --doctest-modules tests

.PHONY: test-cov
test-cov:
	$(RUN) pytest --cov=$(PROJECT_NAME) --cov-report=xml

.PHONY: test-flakiness
test-flakiness:
	$(RUN) pytest -m random --count=100 -x

.PHONY: test-flakiness-quick
test-flakiness-quick:
	$(RUN) pytest -m random --count=10 -x

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

# Documentation targets
.PHONY: docs
docs:
	cd docs && $(RUN) python -m sphinx -b html . _build/html

.PHONY: docs-clean
docs-clean:
	rm -rf docs/_build
	rm -rf docs/functions/qfeval_functions.*.rst
