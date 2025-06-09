# qfeval-functions

qfevalは、Preferred Networks 金融チームが開発している、金融時系列処理のためのフレームワークです。
データ形式の仕様定義、金融時系列データを効率的に扱うためのクラス/関数群、および金融時系列モデルの評価フレームワークが含まれます。

qfeval-functionsは、qfevalの中でも、金融時系列データを効率的に扱うための関数群を提供します。

---

qfeval is a framework developed by Preferred Networks' Financial Solutions team for processing financial time series data.
It includes: data format specification definitions, a set of classes/functions for efficiently handling financial time series data, and a framework for evaluating financial time series models.

qfeval-functions specifically provides a collection of functions within qfeval that facilitate efficient processing of financial time series data.


## Installation

```bash
pip install qfeval-functions
```

## Usage
TBD

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
