# qfeval-functions
[![python](https://img.shields.io/badge/python-%3E=3.9-blue.svg)](https://pypi.org/project/qfeval-functions/)
[![pypi](https://img.shields.io/pypi/v/qfeval-functions.svg)](https://pypi.org/project/qfeval-functions/)
[![CI](https://github.com/pfnet-research/qfeval-functions/actions/workflows/ci-python.yaml/badge.svg)](https://github.com/pfnet-research/qfeval-functions/actions/workflows/ci-python.yaml)
[![codecov](https://codecov.io/gh/pfnet-research/qfeval-functions/graph/badge.svg?token=8U6KIJ10CF)](https://codecov.io/gh/pfnet-research/qfeval-functions)
[![downloads](https://img.shields.io/pypi/dm/qfeval-functions)](https://pypi.org/project/qfeval-functions)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


qfevalは、Preferred Networks 金融チームが開発している、金融時系列処理のためのフレームワークです。
データ形式の仕様定義、金融時系列データを効率的に扱うためのクラス/関数群、および金融時系列モデルの評価フレームワークが含まれます。

qfeval-functionsは、qfevalの中でも、金融時系列データを効率的に扱うための関数群を提供します。

---

qfeval is a framework developed by Preferred Networks' Financial Solutions team for processing financial time series data.
It includes: data format specification definitions, a set of classes/functions for efficiently handling financial time series data, and a framework for evaluating financial time series models.

qfeval-functions specifically provides a collection of functions within qfeval that facilitate efficient processing of financial time series data.


## Installation

```bash
pip install qfeval_functions
```

## Documentation
https://qfeval-functions.readthedocs.io/

# Pitfalls

## Calling qfeval_functions.random.seed without `fast=True` may slow down `qfeval_functions.functions.rand*`

Calling `qfeval_functions.random.seed` without `fast=True` makes `QF.rand*` functions use
reproducible random number generators implemented on CPU.
Setting `fast=True` lets `qfeval_functions.functions.rand*` functions to use random number generators
provided by CUDA via PyTorch.  It is fast, but it is closed source.
It cannot be reproducible on CPUs, and it may not be reproducible with other
PyTorch/CUDA versions.
For reproducibility, calling `qfeval_functions.random.seed` forces to use a reproducible way by
default.

# Contributing
## Testing
 - `make format`: format codes according to the lint rules
 - `make lint`: lint checking
 - `make test`: run pytest and other tests

## Generate documentation
 1. run `make docs-plamo-translate` to translate the documentation on your Mac or GitHub Actions (Auto Translate Documentation workflow).
 2. run `make docs` to build the documentation. (This is not required for pushing to GitHub. It is only required for local development.)

You can also run `make docs-plamo-translate-dry-run` to check if the documentation is translated correctly.
