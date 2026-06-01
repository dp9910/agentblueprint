"""
Phase 1 — Concentration Test
=============================
Make-or-break check: does a small number of clusters cover a large share of tickets?
If not, the blueprint premise doesn't hold for this data — stop early.

Steps:
  1. Load dataset, 80/20 stratified train/test split
  2. Embed training tickets with paraphrase-multilingual-MiniLM-L12-v2
  3. UMAP reduce to lower dimensions (HDBSCAN needs this for high-dim data)
  4. Cluster with HDBSCAN (primary) and k-means (comparison)
  5. Plot concentration curve
  6. Validate clusters against existing labels (type, queue, tags)

Outputs saved to ./output/
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import hdbscan
import umap

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ── 1. Load dataset and split ────────────────────────────────────────────────

print("=" * 60)
print("PHASE 1: CONCENTRATION TEST")
print("=" * 60)

print("\n[1/6] Loading dataset...")
ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
df = ds.to_pandas()
print(f"  Total tickets: {len(df):,}")
print(f"  Columns: {list(df.columns)}")
print(f"  Languages: {dict(Counter(df['language']))}")
print(f"  Types: {dict(Counter(df['type'].fillna('Unknown')))}")

# Combine subject + body for embedding (main intent signal)
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
df["text"] = df["text"].str.strip()

# Fill NaN in categorical columns for stratification and later metrics
df["type"] = df["type"].fillna("Unknown")
df["queue"] = df["queue"].fillna("Unknown")
df["tag_1"] = df["tag_1"].fillna("none")

# Stratify by 'type' (Incident, Request, etc.)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["type"]
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"\n  Train set: {len(train_df):,}")
print(f"  Test set:  {len(test_df):,}")
print(f"  Train type distribution: {dict(Counter(train_df['type']))}")

# Save the split for reproducibility
train_df.to_parquet(OUTPUT_DIR / "train_split.parquet")
test_df.to_parquet(OUTPUT_DIR / "test_split.parquet")
print("  Splits saved to output/")


# ── 2. Embed training tickets ────────────────────────────────────────────────

print("\n[2/6] Embedding training tickets...")
embeddings_path = OUTPUT_DIR / "train_embeddings.npy"

if embeddings_path.exists():
    print("  Found cached embeddings, loading...")
    embeddings = np.load(embeddings_path)
    print(f"  Loaded embeddings: {embeddings.shape}")
else:
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    t0 = time.time()
    embeddings = model.encode(
        train_df["text"].tolist(),
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"  Embedded {len(embeddings):,} tickets in {elapsed:.1f}s")
    print(f"  Shape: {embeddings.shape}")
    np.save(embeddings_path, embeddings)
    print(f"  Saved to {embeddings_path}")


# ── 3. UMAP dimensionality reduction ─────────────────────────────────────────

print("\n[3/6] UMAP reduction (384-dim → multiple targets)...")
umap_path = OUTPUT_DIR / "train_umap_embeddings.npz"

# We'll try multiple UMAP dimensions to see what works best for HDBSCAN
umap_dims = [15, 30, 50]
umap_embeddings = {}

if umap_path.exists():
    print("  Found cached UMAP embeddings, loading...")
    data = np.load(umap_path)
    for d in umap_dims:
        key = f"dim_{d}"
        if key in data:
            umap_embeddings[d] = data[key]
            print(f"    dim={d}: {umap_embeddings[d].shape}")
else:
    for d in umap_dims:
        t0 = time.time()
        reducer = umap.UMAP(
            n_components=d,
            n_neighbors=30,
            min_dist=0.0,  # tighter clusters for HDBSCAN
            metric="cosine",
            random_state=RANDOM_SEED,
            low_memory=True,
        )
        umap_embeddings[d] = reducer.fit_transform(embeddings)
        elapsed = time.time() - t0
        print(f"  dim={d}: {umap_embeddings[d].shape} ({elapsed:.1f}s)")

    np.savez(umap_path, **{f"dim_{d}": umap_embeddings[d] for d in umap_dims})
    print(f"  Saved to {umap_path}")

# Also do a 2D reduction for visualization
umap_2d_path = OUTPUT_DIR / "train_umap_2d.npy"
if umap_2d_path.exists():
    umap_2d = np.load(umap_2d_path)
else:
    print("  Computing 2D UMAP for visualization...")
    reducer_2d = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="cosine", random_state=RANDOM_SEED, low_memory=True,
    )
    umap_2d = reducer_2d.fit_transform(embeddings)
    np.save(umap_2d_path, umap_2d)
    print(f"  Saved 2D UMAP to {umap_2d_path}")


# ── 4. Clustering ────────────────────────────────────────────────────────────

print("\n[4/6] Clustering...")

# 4a. HDBSCAN on each UMAP dimension
print("\n  --- HDBSCAN (on UMAP-reduced data) ---")
hdb_results = {}
for d in umap_dims:
    for min_cs in [30, 50, 100, 200]:
        t0 = time.time()
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=10,
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=-1,
        )
        labels = hdb.fit_predict(umap_embeddings[d])
        elapsed = time.time() - t0

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        coverage = 1 - (n_noise / len(labels))

        hdb_results[(d, min_cs)] = {
            "labels": labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "coverage": coverage,
            "time": elapsed,
        }
        print(f"    dim={d:2d} min_cs={min_cs:3d} | clusters={n_clusters:3d} | "
              f"coverage={coverage*100:.1f}% | noise={n_noise:,} | {elapsed:.1f}s")

# Pick the best HDBSCAN config: highest coverage with a reasonable number of clusters (>5)
best_hdb_key = max(
    [(k, v) for k, v in hdb_results.items() if v["n_clusters"] >= 5],
    key=lambda x: x[1]["coverage"],
    default=None,
)

if best_hdb_key is None:
    # Fallback: just pick highest coverage
    best_hdb_key = max(hdb_results.items(), key=lambda x: x[1]["coverage"])
    best_hdb_key = best_hdb_key[0]
else:
    best_hdb_key = best_hdb_key[0]

hdb_best = hdb_results[best_hdb_key]
hdb_labels = hdb_best["labels"]
n_hdb_clusters = hdb_best["n_clusters"]
hdb_coverage = hdb_best["coverage"]
n_hdb_noise = hdb_best["n_noise"]
print(f"\n  Best HDBSCAN: dim={best_hdb_key[0]}, min_cs={best_hdb_key[1]} "
      f"→ {n_hdb_clusters} clusters, {hdb_coverage*100:.1f}% coverage")

# Cluster size distribution
hdb_cluster_sizes = Counter(hdb_labels)
if -1 in hdb_cluster_sizes:
    del hdb_cluster_sizes[-1]
hdb_sizes_sorted = sorted(hdb_cluster_sizes.values(), reverse=True)
print(f"  Top 15 cluster sizes: {hdb_sizes_sorted[:15]}")


# 4b. K-Means on raw embeddings (comparison, sweep k)
print("\n  --- K-Means (on raw 384-dim embeddings) ---")
k_values = [10, 20, 30, 50, 75, 100, 150]
kmeans_results = {}

for k in k_values:
    t0 = time.time()
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, max_iter=300)
    km_labels = km.fit_predict(embeddings)
    elapsed = time.time() - t0

    subsample_idx = np.random.choice(len(embeddings), size=min(5000, len(embeddings)), replace=False)
    sil = silhouette_score(embeddings[subsample_idx], km_labels[subsample_idx])

    kmeans_results[k] = {
        "labels": km_labels,
        "inertia": km.inertia_,
        "silhouette": sil,
        "centroids": km.cluster_centers_,
        "time": elapsed,
    }
    print(f"    k={k:3d} | silhouette={sil:.3f} | inertia={km.inertia_:.0f} | {elapsed:.1f}s")


# ── 5. Concentration curves + visualization ──────────────────────────────────

print("\n[5/6] Building plots...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 5a. HDBSCAN concentration curve
ax = axes[0, 0]
hdb_cumulative = np.cumsum(hdb_sizes_sorted) / len(train_df) * 100
ax.plot(range(1, len(hdb_cumulative) + 1), hdb_cumulative, "b-o", markersize=3)
ax.axhline(y=70, color="r", linestyle="--", alpha=0.7, label="70% target")
ax.set_xlabel("Number of clusters (sorted by size)")
ax.set_ylabel("Cumulative % of tickets covered")
ax.set_title(f"HDBSCAN Concentration\n({n_hdb_clusters} clusters, {hdb_coverage*100:.1f}% total coverage)")
ax.legend()
ax.grid(True, alpha=0.3)

clusters_for_70 = None
for i, cum in enumerate(hdb_cumulative):
    if cum >= 70:
        clusters_for_70 = i + 1
        ax.axvline(x=i + 1, color="g", linestyle=":", alpha=0.7)
        ax.annotate(f"{i+1} clusters\nfor 70%", xy=(i + 1, 70), fontsize=9,
                    xytext=(i + max(3, n_hdb_clusters // 10), 55),
                    arrowprops=dict(arrowstyle="->"))
        break

# 5b. K-Means concentration curves
ax = axes[0, 1]
for k in [20, 50, 100]:
    km_labels_k = kmeans_results[k]["labels"]
    km_sizes = sorted(Counter(km_labels_k).values(), reverse=True)
    km_cumulative = np.cumsum(km_sizes) / len(train_df) * 100
    ax.plot(range(1, len(km_cumulative) + 1), km_cumulative, "-o", markersize=3, label=f"k={k}")
ax.axhline(y=70, color="r", linestyle="--", alpha=0.7, label="70% target")
ax.set_xlabel("Number of clusters (sorted by size)")
ax.set_ylabel("Cumulative % of tickets covered")
ax.set_title("K-Means Concentration Curves")
ax.legend()
ax.grid(True, alpha=0.3)

# 5c. Silhouette scores vs k
ax = axes[0, 2]
ks = sorted(kmeans_results.keys())
sils = [kmeans_results[k]["silhouette"] for k in ks]
ax.plot(ks, sils, "g-o")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette score")
ax.set_title("K-Means: Silhouette vs k")
ax.grid(True, alpha=0.3)

# 5d. 2D UMAP scatter colored by HDBSCAN clusters
ax = axes[1, 0]
noise_mask = hdb_labels == -1
ax.scatter(umap_2d[noise_mask, 0], umap_2d[noise_mask, 1],
           c="lightgray", s=1, alpha=0.3, label=f"Noise ({noise_mask.sum():,})")
scatter = ax.scatter(umap_2d[~noise_mask, 0], umap_2d[~noise_mask, 1],
                     c=hdb_labels[~noise_mask], cmap="tab20", s=1, alpha=0.5)
ax.set_title(f"UMAP 2D — HDBSCAN clusters")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(markerscale=5, loc="upper right")

# 5e. 2D UMAP scatter colored by ticket type
ax = axes[1, 1]
type_map = {t: i for i, t in enumerate(sorted(train_df["type"].unique()))}
type_colors = np.array([type_map[t] for t in train_df["type"]])
scatter2 = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=type_colors, cmap="Set1", s=1, alpha=0.5)
ax.set_title("UMAP 2D — by ticket type")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
legend_handles = [plt.scatter([], [], c=[plt.cm.Set1(type_map[t] / len(type_map))], s=20, label=t)
                  for t in sorted(type_map.keys())]
ax.legend(handles=legend_handles, markerscale=2, loc="upper right", fontsize=7)

# 5f. 2D UMAP scatter colored by language
ax = axes[1, 2]
lang_colors = np.array([0 if l == "en" else 1 for l in train_df["language"]])
ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=lang_colors, cmap="coolwarm", s=1, alpha=0.5)
ax.set_title("UMAP 2D — by language")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='en'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='de'),
], loc="upper right")

plt.suptitle("Phase 1: Concentration Test Results", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "concentration_curves.png", dpi=150, bbox_inches="tight")
print(f"  Saved concentration_curves.png")


# ── 6. Cluster-to-label coherence ────────────────────────────────────────────

print("\n[6/6] Validating cluster coherence against labels...")

type_labels = train_df["type"].values
queue_labels = train_df["queue"].values
tag1_labels = train_df["tag_1"].values

print("\n  HDBSCAN vs ground-truth labels (non-noise points only):")
hdb_mask = hdb_labels != -1
if hdb_mask.sum() > 100:
    for label_name, label_arr in [("type", type_labels), ("queue", queue_labels), ("tag_1", tag1_labels)]:
        ari = adjusted_rand_score(label_arr[hdb_mask], hdb_labels[hdb_mask])
        nmi = normalized_mutual_info_score(label_arr[hdb_mask], hdb_labels[hdb_mask])
        print(f"    vs {label_name:6s}: ARI={ari:.3f}  NMI={nmi:.3f}")

best_k = max(kmeans_results, key=lambda k: kmeans_results[k]["silhouette"])
km_best_labels = kmeans_results[best_k]["labels"]
print(f"\n  K-Means (k={best_k}, best silhouette) vs ground-truth labels:")
for label_name, label_arr in [("type", type_labels), ("queue", queue_labels), ("tag_1", tag1_labels)]:
    ari = adjusted_rand_score(label_arr, km_best_labels)
    nmi = normalized_mutual_info_score(label_arr, km_best_labels)
    print(f"    vs {label_name:6s}: ARI={ari:.3f}  NMI={nmi:.3f}")

# Per-cluster dominant label breakdown (HDBSCAN, top 15 clusters)
if n_hdb_clusters > 0:
    print(f"\n  HDBSCAN: Top clusters breakdown:")
    print(f"  {'Cluster':>8s}  {'Size':>6s}  {'Type':>20s}  {'Queue':>25s}  {'Tag1':>20s}  {'Lang':>12s}")
    print(f"  {'-'*100}")
    top_clusters = sorted(hdb_cluster_sizes.keys(), key=lambda c: hdb_cluster_sizes[c], reverse=True)[:15]
    for cid in top_clusters:
        mask = hdb_labels == cid
        size = mask.sum()
        top_type = Counter(type_labels[mask]).most_common(1)[0]
        top_queue = Counter(queue_labels[mask]).most_common(1)[0]
        top_tag = Counter(tag1_labels[mask]).most_common(1)[0]
        lang_dist = Counter(train_df["language"].values[mask])
        lang_str = ", ".join(f"{l}:{c}" for l, c in lang_dist.most_common())
        print(f"  {cid:8d}  {size:6,}  {top_type[0]:>14s}({top_type[1]/size*100:3.0f}%)  "
              f"{top_queue[0]:>19s}({top_queue[1]/size*100:3.0f}%)  "
              f"{top_tag[0]:>14s}({top_tag[1]/size*100:3.0f}%)  {lang_str}")


# ── Summary / Decision Gate ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 1 SUMMARY — DECISION GATE")
print("=" * 60)

print(f"\n  Dataset: {len(df):,} tickets ({len(train_df):,} train / {len(test_df):,} test)")

print(f"\n  HDBSCAN (dim={best_hdb_key[0]}, min_cluster_size={best_hdb_key[1]}):")
print(f"    Clusters: {n_hdb_clusters}")
print(f"    Coverage: {hdb_coverage*100:.1f}% of tickets assigned to a cluster")
print(f"    Outliers: {n_hdb_noise:,} ({(1-hdb_coverage)*100:.1f}%)")
if clusters_for_70:
    print(f"    Clusters for 70% coverage: {clusters_for_70}")
else:
    if len(hdb_cumulative) > 0:
        print(f"    Max coverage with all {n_hdb_clusters} clusters: {hdb_cumulative[-1]:.1f}%")

print(f"\n  K-Means (best k={best_k}, silhouette={kmeans_results[best_k]['silhouette']:.3f}):")
km_best_sizes = sorted(Counter(kmeans_results[best_k]["labels"]).values(), reverse=True)
km_cumulative = np.cumsum(km_best_sizes) / len(train_df) * 100
for i, cum in enumerate(km_cumulative):
    if cum >= 70:
        print(f"    Clusters for 70% coverage: {i+1} out of {best_k}")
        break

# Overall concentration assessment
print(f"\n  Concentration assessment:")
# For k-means k=20, how many clusters for 70%?
km20_sizes = sorted(Counter(kmeans_results[20]["labels"]).values(), reverse=True)
km20_cum = np.cumsum(km20_sizes) / len(train_df) * 100
km20_for_70 = next((i+1 for i, c in enumerate(km20_cum) if c >= 70), 20)
print(f"    K-Means k=20: {km20_for_70} clusters cover 70%")
print(f"    K-Means k=50: top cluster covers {sorted(Counter(kmeans_results[50]['labels']).values(), reverse=True)[0]/len(train_df)*100:.1f}%")

print(f"\n  Verdict: ", end="")
# The real test: can a small number of clusters cover most traffic?
# With k-means, if top ~30% of clusters cover 70%, that's decent concentration
if hdb_coverage >= 0.7 and n_hdb_clusters >= 5:
    print("PASS — Strong concentration via HDBSCAN. Proceed to Phase 2.")
elif km20_for_70 <= 8:
    print("PASS — Strong concentration via K-Means. Proceed to Phase 2.")
elif km20_for_70 <= 12:
    print("MARGINAL — Moderate concentration. Blueprint approach may work with tuning.")
else:
    print("FAIL — Weak concentration. Blueprint premise does not hold for this data.")

# Save summary
summary = {
    "total_tickets": len(df),
    "train_size": len(train_df),
    "test_size": len(test_df),
    "language_split": dict(Counter(df["language"])),
    "hdbscan": {
        "best_config": {"umap_dim": best_hdb_key[0], "min_cluster_size": best_hdb_key[1]},
        "n_clusters": n_hdb_clusters,
        "coverage_pct": round(hdb_coverage * 100, 2),
        "n_outliers": int(n_hdb_noise),
        "top_15_cluster_sizes": hdb_sizes_sorted[:15],
        "clusters_for_70pct": clusters_for_70,
    },
    "hdbscan_sweep": {
        f"dim{k[0]}_mincs{k[1]}": {"n_clusters": v["n_clusters"], "coverage": round(v["coverage"] * 100, 2)}
        for k, v in hdb_results.items()
    },
    "kmeans": {
        "best_k": best_k,
        "best_silhouette": round(kmeans_results[best_k]["silhouette"], 4),
        "all_silhouettes": {k: round(v["silhouette"], 4) for k, v in kmeans_results.items()},
    },
}
with open(OUTPUT_DIR / "phase1_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary saved to output/phase1_summary.json")
print(f"  Plots saved to output/concentration_curves.png")
print(f"  Train/test splits saved to output/")
