"""
tabular - Tabular and structured data datasets

This module provides loaders for tabular datasets and CSV files.

Available Datasets
------------------
- Iris: Classic flower classification dataset
- CSVDataset: Generic CSV file loader with auto-detection
"""

from lemon.datasets.tabular.iris import Iris
from lemon.datasets.tabular.csv import CSVDataset

__all__ = [
    "Iris",
    "CSVDataset",
]
