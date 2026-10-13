"""Shared test setup: import paths and the committed processed data."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "data"):
    sys.path.insert(0, str(ROOT / sub))

PROCESSED = ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def occupation_features():
    return pd.read_csv(PROCESSED / "oews_occupation_features.csv")


@pytest.fixture(scope="session")
def industry_features():
    return pd.read_csv(PROCESSED / "oews_industry_features.csv", dtype={"naics": str})


@pytest.fixture(scope="session")
def dashboard():
    """The Dash app module, loaded by file path (app/ is also a namespace package name)."""
    spec = importlib.util.spec_from_file_location("dashboard_app", ROOT / "app" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
