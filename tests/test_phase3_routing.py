"""§5 Phase 3 — Router coverage, centroid normalization, sample structure.

These tests validate the routing infrastructure (Phase 1/2 outputs applied
to the test set).  They are process checks — independent of which model
wins the experiment.
"""

import json
from pathlib import Path

import numpy as np
import pytest

OUTPUT = Path(__file__).resolve().parent.parent / "output"


# ── 1. Routing covers all test tickets ────────────────────────────────────

def test_routing_covers_all_test_tickets(routing_summary, test_routed_df):
    assert len(test_routed_df) == routing_summary["test_set_size"]


# ── 2. All 5 clusters receive tickets ────────────────────────────────────

def test_all_clusters_get_tickets(routing_summary):
    dist = routing_summary["route_distribution"]
    for cid in ["0", "2", "4", "7", "9"]:
        assert dist[cid] > 0, f"Cluster {cid} received 0 tickets"


# ── 3. Experiment sample size ≥ 120 (corrected spec) ─────────────────────

def test_experiment_sample_size():
    """The corrected spec requires ≥120 stratified tickets in sample.jsonl."""
    path = OUTPUT / "sample.jsonl"
    if not path.exists():
        # Fall back to the old parquet if the new experiment hasn't run yet
        old_path = OUTPUT / "experiment_sample.parquet"
        if old_path.exists():
            pytest.skip(
                "sample.jsonl not found — corrected experiment not yet run. "
                "Old experiment_sample.parquet exists (n=30) but does not "
                "satisfy the ≥120 requirement."
            )
        pytest.skip("No sample file found")
    with open(path) as f:
        sample = [json.loads(line) for line in f if line.strip()]
    assert len(sample) >= 120, (
        f"Sample has {len(sample)} tickets, need ≥120"
    )


# ── 4. Both languages present in routed test set ─────────────────────────

def test_both_languages_present(test_routed_df):
    if "translated" in test_routed_df.columns:
        has_translated = test_routed_df["translated"].any()
        has_native = (~test_routed_df["translated"]).any()
        assert has_translated and has_native, (
            "Routed test set should contain both native and translated tickets"
        )
    else:
        langs = set(test_routed_df["language"].unique())
        assert len(langs) >= 2, f"Expected ≥2 languages, got {langs}"


# ── 5. Similarity stats plausible ────────────────────────────────────────

def test_similarity_stats_plausible(routing_summary):
    stats = routing_summary["similarity_stats"]
    assert stats["mean"] > 0.5, f"Mean similarity {stats['mean']} <= 0.5"
    assert 0 <= stats["min"] <= 1
    assert 0 <= stats["max"] <= 1


# ── 6. Centroids shape (5, 384) ──────────────────────────────────────────

def test_centroids_shape(cluster_centroids):
    assert cluster_centroids.shape == (5, 384), (
        f"Expected (5, 384), got {cluster_centroids.shape}"
    )


# ── 7. Centroids are L2-normalized (~1.0) ────────────────────────────────

def test_centroids_normalized(cluster_centroids):
    norms = np.linalg.norm(cluster_centroids, axis=1)
    for i, n in enumerate(norms):
        assert abs(n - 1.0) < 0.05, f"Centroid {i} norm={n:.4f}, expected ~1.0"


# ── 8. No degenerate distribution (no cluster > 80% of total) ────────────

def test_no_degenerate_distribution(routing_summary):
    dist = routing_summary["route_distribution"]
    total = sum(dist.values())
    for cid, count in dist.items():
        pct = count / total * 100
        assert pct < 80, f"Cluster {cid} has {pct:.1f}% of tickets (> 80%)"


# ── 9. Required routing columns in routed parquet ────────────────────────

def test_routing_columns_present(test_routed_df):
    required = {"routed_cluster", "route_similarity", "route_confidence"}
    missing = required - set(test_routed_df.columns)
    assert not missing, f"Missing routing columns: {missing}"
