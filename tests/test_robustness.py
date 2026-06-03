"""§7 Robustness — confidence sweep, cluster sensitivity, coverage decay.

These tests run lightweight analysis on saved embeddings and centroids.
No API calls — only numpy and sklearn operations.
"""

import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.cluster import KMeans


# ─── helpers ──────────────────────────────────────────────────────────────

def _cosine_similarity_matrix(embeddings, centroids):
    """Return (N, K) cosine-similarity matrix."""
    e_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    c_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return e_norm @ c_norm.T


def _route(embeddings, centroids):
    """Return best_sim, confidence_gap arrays."""
    sim = _cosine_similarity_matrix(embeddings, centroids)
    sorted_sim = np.sort(sim, axis=1)[:, ::-1]
    best_sim = sorted_sim[:, 0]
    gap = sorted_sim[:, 0] - sorted_sim[:, 1]
    return best_sim, gap


# ── 1. Confidence sweep is monotonically decreasing ──────────────────────

def test_confidence_sweep_monotonic(test_embeddings, cluster_centroids):
    best_sim, _ = _route(test_embeddings, cluster_centroids)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    coverages = [(best_sim >= t).mean() for t in thresholds]
    for i in range(len(coverages) - 1):
        assert coverages[i] >= coverages[i + 1], (
            f"Coverage not monotonic: {thresholds[i]}->{coverages[i]:.3f}, "
            f"{thresholds[i+1]}->{coverages[i+1]:.3f}"
        )


# ── 2. Coverage spread ≥ 20pp between threshold 0.3 and 0.9 ─────────────

def test_confidence_sweep_range(test_embeddings, cluster_centroids):
    best_sim, _ = _route(test_embeddings, cluster_centroids)
    cov_low = (best_sim >= 0.3).mean() * 100
    cov_high = (best_sim >= 0.9).mean() * 100
    spread = cov_low - cov_high
    assert spread >= 20, f"Spread {spread:.1f}pp < 20pp"


# ── 3. Positive correlation between similarity and confidence gap ────────

def test_confidence_similarity_correlation(test_embeddings, cluster_centroids):
    best_sim, gap = _route(test_embeddings, cluster_centroids)
    r, _ = pearsonr(best_sim, gap)
    assert r > 0.0, f"Pearson(similarity, gap)={r:.3f} ≤ 0"


# ── 4. Varying k changes routing assignments ────────────────────────────

def test_varying_k_changes_routing(combined_embeddings):
    ks = [3, 5, 7, 10, 15]
    labels = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
        labels[k] = km.fit_predict(combined_embeddings)
    # k=3 and k=15 must differ (different number of clusters guarantees this)
    assert not np.array_equal(labels[3], labels[15])


# ── 5. Top-cluster stability: Jaccard of biggest cluster at k=5 vs k=7 ──

def test_top_cluster_stability(combined_embeddings):
    def biggest_cluster_set(k):
        km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
        lbl = km.fit_predict(combined_embeddings)
        counts = np.bincount(lbl)
        biggest = counts.argmax()
        return set(np.where(lbl == biggest)[0])

    s5 = biggest_cluster_set(5)
    s7 = biggest_cluster_set(7)
    jaccard = len(s5 & s7) / len(s5 | s7)
    assert jaccard > 0.3, f"Jaccard(biggest@k=5, biggest@k=7)={jaccard:.3f} ≤ 0.3"


# ── 6. Coverage (sim > 0.6) at k=5 ≥ 50% ────────────────────────────────

def test_coverage_at_different_k(combined_embeddings):
    km = KMeans(n_clusters=5, n_init=3, random_state=42, max_iter=100)
    km.fit(combined_embeddings)
    centroids = km.cluster_centers_
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    best_sim, _ = _route(combined_embeddings, centroids)
    cov = (best_sim >= 0.6).mean() * 100
    assert cov >= 50, f"Coverage at k=5 with sim>0.6 is {cov:.1f}% < 50%"


# ── 7. Coverage decay: centroids from first 70%, tested on last 30% ─────

def test_coverage_decay(combined_embeddings):
    n = len(combined_embeddings)
    split = int(n * 0.7)
    train_part = combined_embeddings[:split]
    test_part = combined_embeddings[split:]

    km = KMeans(n_clusters=5, n_init=3, random_state=42, max_iter=100)
    km.fit(train_part)
    centroids = km.cluster_centers_
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    best_sim, _ = _route(test_part, centroids)
    cov = (best_sim >= 0.6).mean() * 100
    assert cov >= 30, f"New-data coverage {cov:.1f}% < 30%"
