# qfeval-functions

qfevalは、Preferred Networks 金融チームが開発している、金融時系列処理のためのフレームワークです。
データ形式の仕様定義、金融時系列データを効率的に扱うためのクラス/関数群、および金融時系列モデルの評価フレームワークが含まれます。

qfeval-functionsは、qfevalの中でも、金融時系列データを効率的に扱うための関数群を提供します。

## インストール

```bash
pip install qfeval-functions
```

## 使用方法
TBD

# Pitfalls

## Calling qfeval.seed without `fast=True` may slow down `QF.rand*`

Calling `qfeval.seed` without `fast=True` makes `QF.rand*` functions use
reproducible random number generators implemented on CPU.
Setting `fast=True` lets `QF.rand*` functions to use random number generators
provided by CUDA via PyTorch.  It is fast, but it is closed source.
It cannot be reproducible on CPUs, and it may not be reproducible with other
PyTorch/CUDA versions.
For reproducibility, calling `qfeval.seed` forces to use a reproducible way by
default.
