"""
Shared pytest configuration.

Tests that download real datasets are marked with ``@pytest.mark.network``
and are skipped by default. Run them with ``pytest --run-network`` (or set
``LEMON_RUN_NETWORK=1``).

Downloaded datasets are cached in ``LEMON_DATASET_CACHE`` (default:
``~/.cache/lemon/datasets``) so they are fetched only once across tests
and across runs.
"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that download datasets from the internet",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test downloads data from the internet (skipped by default)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network") or os.environ.get("LEMON_RUN_NETWORK") == "1":
        return
    skip_network = pytest.mark.skip(
        reason="needs network; use --run-network or LEMON_RUN_NETWORK=1"
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture(scope="session")
def dataset_root():
    """Persistent directory shared by all tests that download datasets."""
    root = os.environ.get(
        "LEMON_DATASET_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "lemon", "datasets"),
    )
    os.makedirs(root, exist_ok=True)
    return root
