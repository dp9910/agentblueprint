"""§3 Phase 1 — Embeddings, HDBSCAN, concentration curve, decision gate."""

from pathlib import Path

import pytest

OUTPUT = Path(__file__).resolve().parent.parent / "output"


# ── 1. Embedding dimension is 384 (MiniLM) ───────────────────────────────

def test_embedding_dimension_is_384(train_embeddings):
    assert train_embeddings.shape[1] == 384, (
        f"Expected 384-dim embeddings, got {train_embeddings.shape[1]}"
    )


# ── 2. HDBSCAN was used and found clusters ───────────────────────────────

def test_hdbscan_used(phase1_summary):
    hdb = phase1_summary["hdbscan"]
    assert hdb["n_clusters"] > 0, "HDBSCAN should find > 0 clusters"


# ── 3. KMeans comparison present ─────────────────────────────────────────

def test_kmeans_comparison(phase1_summary):
    km = phase1_summary["kmeans"]
    assert km["best_k"] > 0, "KMeans best_k should be > 0"


# ── 4. Concentration curve PNG exists ─────────────────────────────────────

def test_concentration_curve_exists():
    assert (OUTPUT / "concentration_curves.png").exists(), (
        "concentration_curves.png not found"
    )


# ── 5. Decision gate: HDBSCAN coverage ≥ 70% ─────────────────────────────

def test_decision_gate_coverage_above_70(phase1_summary):
    cov = phase1_summary["hdbscan"]["coverage_pct"]
    assert cov >= 70, f"HDBSCAN coverage {cov}% < 70%"


# ── 6. Clusters needed for 70% is ≤ 15 ───────────────────────────────────

def test_clusters_for_70pct_is_small(phase1_summary):
    k = phase1_summary["hdbscan"]["clusters_for_70pct"]
    assert k <= 15, f"clusters_for_70pct={k} > 15"


# ── 7. Recluster coverage ≥ 70% ──────────────────────────────────────────

def test_recluster_coverage(recluster_summary):
    cov = recluster_summary["clustering"]["coverage_pct"]
    assert cov >= 70, f"Recluster coverage {cov}% < 70%"


# ── 8. Recluster produced fewer clusters (14 < 27) ───────────────────────

def test_recluster_fewer_clusters(phase1_summary, recluster_summary):
    original = phase1_summary["hdbscan"]["n_clusters"]
    reclustered = recluster_summary["clustering"]["n_clusters"]
    assert reclustered < original, (
        f"Recluster ({reclustered}) should be < original ({original})"
    )


# ── 9. Recluster clusters_for_70pct ≤ 10 ─────────────────────────────────

def test_recluster_5_for_70pct(recluster_summary):
    k = recluster_summary["clustering"]["clusters_for_70pct"]
    assert k <= 10, f"Recluster clusters_for_70pct={k} > 10"


# ── 10. ≥50% of clusters have type_purity ≥ 0.5 ─────────────────────────

def test_clusters_map_to_real_intents(recluster_summary):
    details = recluster_summary["cluster_details"]
    pure = sum(1 for c in details if c["type_purity"] >= 0.5)
    assert pure / len(details) >= 0.5, (
        f"Only {pure}/{len(details)} clusters have type_purity >= 0.5"
    )


# ── 11. No language blobs ────────────────────────────────────────────────

def test_no_language_blobs(recluster_summary):
    blobs = recluster_summary["language_blobs"]
    assert len(blobs) == 0, f"Language blobs found: {blobs}"


# ── 12. HDBSCAN sweep tested ≥ 8 configurations ─────────────────────────

def test_hdbscan_sweep_breadth(phase1_summary):
    n = len(phase1_summary["hdbscan_sweep"])
    assert n >= 8, f"Only {n} HDBSCAN configs tested; need ≥ 8"
