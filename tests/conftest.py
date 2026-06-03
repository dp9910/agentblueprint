"""Shared fixtures — load each output artifact once per session.

Phase 1/2 artifacts are loaded unconditionally (they exist).
Old experiment artifacts (experiment_report, real_experiment_*) are kept
only for the guardrails tests that still reference them (leakage, split).
Phase 3 corrected-experiment artifacts are loaded in their own test modules
with graceful skips if the corrected experiment hasn't run yet.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

OUTPUT = Path(__file__).resolve().parent.parent / "output"


# ── JSON fixtures ──────────────────────────────────────────────────────────

def _load_json(name: str) -> dict:
    with open(OUTPUT / name) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def phase1_summary():
    return _load_json("phase1_summary.json")


@pytest.fixture(scope="session")
def phase1_sanity():
    return _load_json("phase1_sanity_checks.json")


@pytest.fixture(scope="session")
def recluster_summary():
    return _load_json("recluster_summary.json")


@pytest.fixture(scope="session")
def blueprints():
    return _load_json("blueprints.json")


@pytest.fixture(scope="session")
def blueprint_cost():
    return _load_json("blueprint_generation_cost.json")


@pytest.fixture(scope="session")
def blueprint_review():
    return _load_json("blueprint_review_results.json")


@pytest.fixture(scope="session")
def routing_summary():
    return _load_json("routing_summary.json")


# Old experiment artifacts — used only by guardrails tests for split checks
@pytest.fixture(scope="session")
def judge_results():
    return _load_json("judge_results.json")


@pytest.fixture(scope="session")
def real_experiment_report():
    return _load_json("real_experiment_report.json")


@pytest.fixture(scope="session")
def real_experiment_progress():
    return _load_json("real_experiment_progress.json")


# ── Parquet fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def train_df():
    return pd.read_parquet(OUTPUT / "train_split.parquet")


@pytest.fixture(scope="session")
def test_df():
    return pd.read_parquet(OUTPUT / "test_split.parquet")


@pytest.fixture(scope="session")
def combined_train_df():
    return pd.read_parquet(OUTPUT / "combined_train_translated.parquet")


@pytest.fixture(scope="session")
def test_routed_df():
    return pd.read_parquet(OUTPUT / "test_routed.parquet")


# ── Numpy fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def train_embeddings():
    return np.load(OUTPUT / "train_embeddings.npy")


@pytest.fixture(scope="session")
def test_embeddings():
    return np.load(OUTPUT / "test_embeddings.npy")


@pytest.fixture(scope="session")
def combined_embeddings():
    return np.load(OUTPUT / "combined_embeddings.npy")


@pytest.fixture(scope="session")
def cluster_centroids():
    return np.load(OUTPUT / "cluster_centroids.npy")
