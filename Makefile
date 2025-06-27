PROJECT_NAME := qfeval_functions
RUN := uv run

.PHONY: check
check: test lint

.PHONY: install
install:
	uv sync --dev

.PHONY: test
test: doctest pytest doctest

.PHONY: doctest
doctest:
	$(RUN) pytest --doctest-modules $(PROJECT_NAME) -v

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
docs: docs-apidoc
	$(RUN) sphinx-build -b html docs docs/_build/html/en && \
	$(RUN) sphinx-build -b html -D language='ja' docs docs/_build/html/ja && \
	cp docs/_index.html docs/_build/html/index.html

.PHONY: docs-apidoc
docs-apidoc:
	$(RUN) sphinx-apidoc -f -e -M -o docs/api qfeval_functions
	cd docs && $(RUN) python postprocess_apidoc.py

.PHONY: docs-generate-locale-ja
docs-generate-locale-ja: docs
	@cd docs && \
	$(RUN) sphinx-build -M gettext ./ ./_build/ && \
	$(RUN) sphinx-intl update -p ./_build/gettext -l ja && \
	echo "please check and modify docs/source/locale/ja/LC_MESSAGES/index.po"

.PHONY: docs-clean
docs-clean:
	rm -rf docs/_build docs/api

.PHONY: docs-plamo-translate
docs-plamo-translate: docs-generate-locale-ja
	if [ "$(shell uname -s)" != "Darwin" ]; then \
		echo "This command is only supported on Mac OS"; \
		exit 1; \
	fi
	if [ "$(shell uname -m)" != "arm64" ]; then \
		echo "This command is only supported on Apple Silicon"; \
		exit 1; \
	fi
	if [ "$(shell python --version | cut -d' ' -f2 | cut -d'.' -f1)" -ge 3 ] && [ "$(shell python --version | cut -d' ' -f2 | cut -d'.' -f2)" -ge 13 ]; then \
		if ! command -v cmake &> /dev/null; then \
			brew install cmake; \
		fi; \
		uv pip install git+https://github.com/google/sentencepiece.git@2734490#subdirectory=python; \
	fi
	uv pip install plamo-translate
