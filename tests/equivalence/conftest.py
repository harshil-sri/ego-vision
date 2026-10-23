"""Pytest plugin for the equivalence harness.

Adds the --impl option: `pytest tests/equivalence --impl core` sets EGO_IMPL
so src.registry loads the owner's core/ implementations for every piece and
the SAME shared test vectors run against them. Default: reference.
"""

from __future__ import annotations

import os


def pytest_addoption(parser):
    parser.addoption("--impl", action="store", default="reference",
                     choices=["reference", "core"],
                     help="implementation under test: reference or core")


def pytest_configure(config):
    os.environ["EGO_IMPL"] = config.getoption("--impl")
