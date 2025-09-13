"""
Model package for Pay‑PrivAuth.

This package exposes baseline and placeholder models used in the behavioural authentication framework.  At present it includes a simple logistic regression classifier for demonstration purposes, a federated averaging utility and stubs for a temporal convolutional network (TCN) encoder and differential privacy mechanisms.

Submodules:

```
baseline.py   – baseline classifier using scikit‑learn
aggregator.py – federated averaging helper
tcn.py        – placeholder for TCN encoder
dp.py         – stubs for differential privacy utilities
```
"""

from .baseline import LogisticBaseline
from .aggregator import FedAvgAggregator

__all__ = [
    "LogisticBaseline",
    "FedAvgAggregator",
]