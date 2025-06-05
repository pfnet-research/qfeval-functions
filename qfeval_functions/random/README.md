# qfeval.random

qfeval.random provides utility functions for random value generation.

**CAVEAT: `fast` random generation mode**

When `qfeval.random.is_fast()` returns `False`, random functions should use
random generators implemented on CPUs.  Otherwise, random functions may use
device specific libraries such as cuRAND to speed up random value generation.
