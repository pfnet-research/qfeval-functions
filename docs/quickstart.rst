Quick Start
===========

Installation
------------

Install qfeval-functions from source:

.. code-block:: bash

   git clone https://github.com/pfnet-research/qfeval-functions
   cd qfeval-functions
   pip install -e .

Basic Usage
-----------

Import the library and start using the functions:

.. code-block:: python

   import torch
   import qfeval_functions.functions as QF

   # Create sample data
   x = torch.randn(100, 20)
   
   # Calculate moving average
   ma = QF.ma(x, span=5, dim=1)
   
   # Calculate correlation
   y = torch.randn(100, 20)
   corr = QF.correl(x, y, dim=1)

Available Functions
-------------------

The library provides various categories of functions:

* **Moving Window Functions**: ma, msum, mmax, mmin, mstd, mvar
* **Statistical Functions**: correl, covar, nanmean, nansum, nanvar
* **Financial Indicators**: rsi, ema, bollinger_band
* **Array Operations**: fillna, bfill, ffill, shift
* **Linear Algebra**: eigh, pca, orthogonalize

See the :doc:`functions/index` for detailed documentation of each function.