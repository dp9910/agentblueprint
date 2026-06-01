"""
Phase 1 Re-cluster: English + Translated German Sample
=======================================================
Uses the 1,500 translated German tickets from the validation run
plus all English training tickets. Re-embeds and re-clusters from scratch
to check whether the language blob dissolves.

Compares:
  - Original clustering (language blob present)
  - New clustering (translated German mixed in)
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import hdbscan
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTPUT_DIR = Path("output")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ── Load data ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("RE-CLUSTER: English + Translated German")
print("=" * 60)

print("\n[1/6] Loading data...")
train_df = pd.read_parquet(OUTPUT_DIR / "train_split.parquet")

# Load translation cache
with open(OUTPUT_DIR / "translations_cache.json") as f:
    cache = json.load(f)
print(f"  Translation cache: {len(cache):,} entries")

# Split into English and German
en_mask = train_df["language"] == "en"
de_mask = train_df["language"] == "de"
en_df = train_df[en_mask].copy().reset_index(drop=True)
print(f"  English training tickets: {len(en_df):,}")

# For German, only keep tickets where BOTH subject and body are in cache
de_df = train_df[de_mask].copy()
de_df["subj_translated"] = de_df["subject"].apply(
    lambda x: str(x) in cache if pd.notna(x) else True)
de_df["body_translated"] = de_df["body"].apply(
    lambda x: str(x) in cache if pd.notna(x) else True)
de_translated = de_df[de_df["subj_translated"] & de_df["body_translated"]].copy()
de_translated = de_translated.reset_index(drop=True)
print(f"  German tickets with translations available: {len(de_translated):,}")

# Apply translations
de_translated["subject"] = de_translated["subject"].apply(
    lambda x: cache.get(str(x), str(x)) if pd.notna(x) else "")
de_translated["body"] = de_translated["body"].apply(
    lambda x: cache.get(str(x), str(x)) if pd.notna(x) else "")

# Build text column
en_df["text"] = en_df["subject"].fillna("") + " " + en_df["body"].fillna("")
en_df["text"] = en_df["text"].str.strip()
en_df["translated"] = False

de_translated["text"] = de_translated["subject"].fillna("") + " " + de_translated["body"].fillna("")
de_translated["text"] = de_translated["text"].str.strip()
de_translated["translated"] = True

# Combine
combined_df = pd.concat([en_df, de_translated], ignore_index=True)
# Fill NaN in categorical columns
combined_df["type"] = combined_df["type"].fillna("Unknown")
combined_df["queue"] = combined_df["queue"].fillna("Unknown")
combined_df["tag_1"] = combined_df["tag_1"].fillna("none")

print(f"  Combined corpus: {len(combined_df):,} tickets")
print(f"    English (native): {(~combined_df['translated']).sum():,}")
print(f"    German (translated): {combined_df['translated'].sum():,}")


# ── Embed ─────────────────────────────────────────────────────────────────────

print("\n[2/6] Embedding combined corpus...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
t0 = time.time()
combined_embeddings = model.encode(
    combined_df["text"].tolist(),
    batch_size=256, show_progress_bar=True, normalize_embeddings=True,
)
elapsed = time.time() - t0
print(f"  Embedded {len(combined_embeddings):,} tickets in {elapsed:.1f}s")
print(f"  Shape: {combined_embeddings.shape}")

np.save(OUTPUT_DIR / "combined_embeddings.npy", combined_embeddings)


# ── UMAP reduce ──────────────────────────────────────────────────────────────

print("\n[3/6] UMAP reduction...")
t0 = time.time()
reducer = umap.UMAP(
    n_components=30, n_neighbors=30, min_dist=0.0,
    metric="cosine", random_state=RANDOM_SEED, low_memory=True,
)
umap_30 = reducer.fit_transform(combined_embeddings)
print(f"  dim=30: {umap_30.shape} ({time.time()-t0:.1f}s)")

# 2D for visualization
t0 = time.time()
reducer_2d = umap.UMAP(
    n_components=2, n_neighbors=30, min_dist=0.1,
    metric="cosine", random_state=RANDOM_SEED, low_memory=True,
)
umap_2d = reducer_2d.fit_transform(combined_embeddings)
print(f"  dim=2:  {umap_2d.shape} ({time.time()-t0:.1f}s)")


# ── Cluster ───────────────────────────────────────────────────────────────────

print("\n[4/6] Clustering (HDBSCAN)...")
hdb = hdbscan.HDBSCAN(
    min_cluster_size=200, min_samples=10,
    metric="euclidean", cluster_selection_method="eom", core_dist_n_jobs=-1,
)
labels = hdb.fit_predict(umap_30)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
coverage = 1 - (n_noise / len(labels))

print(f"  Clusters: {n_clusters}")
print(f"  Coverage: {coverage*100:.1f}%")
print(f"  Noise: {n_noise:,} ({n_noise/len(labels)*100:.1f}%)")

cluster_sizes = Counter(labels)
if -1 in cluster_sizes:
    del cluster_sizes[-1]
sizes_sorted = sorted(cluster_sizes.items(), key=lambda x: -x[1])
print(f"  Top 15 cluster sizes: {[s for _, s in sizes_sorted[:15]]}")


# ── Analyze: did the language blob dissolve? ──────────────────────────────────

print("\n[5/6] Analyzing cluster composition...")

translated_mask = combined_df["translated"].values
type_labels = combined_df["type"].values
tag1_labels = combined_df["tag_1"].values
lang_labels = combined_df["language"].values

print(f"\n  {'Rank':>4s}  {'CID':>4s}  {'Size':>6s}  {'%En':>5s}  {'%De(tr)':>7s}  "
      f"{'Type Purity':>12s}  {'Tag1 Purity':>12s}  Dominant Labels")
print(f"  {'-' * 110}")

cluster_data = []
for rank, (cid, size) in enumerate(sizes_sorted[:20], 1):
    mask = labels == cid

    # Language/translation composition
    n_en = (~translated_mask[mask]).sum()
    n_de_tr = translated_mask[mask].sum()
    en_pct = n_en / size * 100
    de_pct = n_de_tr / size * 100

    # Type purity
    type_counts = Counter(type_labels[mask])
    top_type, top_type_n = type_counts.most_common(1)[0]
    type_purity = top_type_n / size

    # Tag1 purity
    tag_counts = Counter(tag1_labels[mask])
    top_tag, top_tag_n = tag_counts.most_common(1)[0]
    tag_purity = top_tag_n / size

    is_language_blob = de_pct > 80 and top_type == "Unknown"
    marker = " ← LANGUAGE BLOB?" if is_language_blob else ""

    print(f"  {rank:4d}  {cid:4d}  {size:6,}  {en_pct:4.0f}%  {de_pct:5.0f}%    "
          f"{type_purity:8.1%}      {tag_purity:8.1%}      "
          f"type={top_type}, tag1={top_tag}{marker}")

    cluster_data.append({
        "rank": rank, "cluster_id": int(cid), "size": int(size),
        "pct_english": round(en_pct, 1), "pct_german_translated": round(de_pct, 1),
        "type_purity": round(type_purity, 3), "tag1_purity": round(tag_purity, 3),
        "dominant_type": top_type, "dominant_tag1": top_tag,
        "is_language_blob": is_language_blob,
    })

# Check: are there any large clusters that are >80% one language with Unknown type?
language_blobs = [c for c in cluster_data if c["is_language_blob"]]
print(f"\n  Language blobs detected: {len(language_blobs)}")
if language_blobs:
    for lb in language_blobs:
        print(f"    Cluster {lb['cluster_id']} (rank #{lb['rank']}, n={lb['size']:,})")

# Concentration curve for new clustering
cumulative = np.cumsum([s for _, s in sizes_sorted]) / len(combined_df) * 100
clusters_for_70 = next((i+1 for i, c in enumerate(cumulative) if c >= 70), None)
print(f"\n  Concentration: {clusters_for_70} clusters cover 70% of traffic")

# Coherence metrics
print(f"\n  Cluster coherence vs labels (non-noise):")
non_noise = labels != -1
for label_name, label_arr in [("type", type_labels), ("tag_1", tag1_labels)]:
    ari = adjusted_rand_score(label_arr[non_noise], labels[non_noise])
    nmi = normalized_mutual_info_score(label_arr[non_noise], labels[non_noise])
    print(f"    vs {label_name:6s}: ARI={ari:.3f}  NMI={nmi:.3f}")


# ── Plots ─────────────────────────────────────────────────────────────────────

print("\n[6/6] Generating plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Concentration curve
ax = axes[0, 0]
ax.plot(range(1, len(cumulative) + 1), cumulative, "b-o", markersize=3)
ax.axhline(y=70, color="r", linestyle="--", alpha=0.7, label="70% target")
if clusters_for_70:
    ax.axvline(x=clusters_for_70, color="g", linestyle=":", alpha=0.7)
    ax.annotate(f"{clusters_for_70} clusters", xy=(clusters_for_70, 70),
                xytext=(clusters_for_70 + 3, 55), arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("Number of clusters (sorted by size)")
ax.set_ylabel("Cumulative % of tickets")
ax.set_title(f"Concentration Curve (EN + translated DE)\n{n_clusters} clusters, {coverage*100:.1f}% coverage")
ax.legend()
ax.grid(True, alpha=0.3)

# UMAP colored by cluster
ax = axes[0, 1]
noise_m = labels == -1
ax.scatter(umap_2d[noise_m, 0], umap_2d[noise_m, 1],
           c="lightgray", s=1, alpha=0.2, label=f"Noise ({noise_m.sum():,})")
ax.scatter(umap_2d[~noise_m, 0], umap_2d[~noise_m, 1],
           c=labels[~noise_m], cmap="tab20", s=1, alpha=0.5)
ax.set_title("UMAP 2D — clusters")
ax.legend(markerscale=5)

# UMAP colored by source (native EN vs translated DE)
ax = axes[1, 0]
en_m = ~translated_mask
de_m = translated_mask
ax.scatter(umap_2d[en_m, 0], umap_2d[en_m, 1], c="steelblue", s=1, alpha=0.3, label=f"English ({en_m.sum():,})")
ax.scatter(umap_2d[de_m, 0], umap_2d[de_m, 1], c="crimson", s=1, alpha=0.5, label=f"German translated ({de_m.sum():,})")
ax.set_title("UMAP 2D — native EN vs translated DE")
ax.legend(markerscale=5)

# UMAP colored by type
ax = axes[1, 1]
type_map = {t: i for i, t in enumerate(sorted(combined_df["type"].unique()))}
type_colors = np.array([type_map[t] for t in type_labels])
ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=type_colors, cmap="Set1", s=1, alpha=0.4)
ax.set_title("UMAP 2D — by ticket type")
handles = [plt.scatter([], [], c=[plt.cm.Set1(type_map[t] / max(len(type_map), 1))], s=20, label=t)
           for t in sorted(type_map.keys())]
ax.legend(handles=handles, markerscale=2, fontsize=7)

plt.suptitle("Re-clustering: English + Translated German (1,500 sample)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "recluster_translated.png", dpi=150, bbox_inches="tight")
print(f"  Saved recluster_translated.png")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"\n  Corpus: {len(en_df):,} English + {len(de_translated):,} translated German = {len(combined_df):,}")
print(f"  Clusters: {n_clusters}")
print(f"  Coverage: {coverage*100:.1f}%")
print(f"  Clusters for 70%: {clusters_for_70}")
print(f"  Language blobs: {len(language_blobs)}")

if len(language_blobs) == 0:
    print(f"\n  RESULT: Language blob DISSOLVED. Translation fixes the clustering.")
    print(f"  German tickets now mix into intent-based clusters alongside English.")
else:
    blob_total = sum(lb["size"] for lb in language_blobs)
    print(f"\n  RESULT: {len(language_blobs)} language blob(s) remain ({blob_total:,} tickets).")
    print(f"  Translation partially fixes the issue.")

# Save
summary = {
    "corpus": {"english": len(en_df), "german_translated": len(de_translated), "total": len(combined_df)},
    "clustering": {
        "n_clusters": n_clusters, "coverage_pct": round(coverage * 100, 2),
        "noise": int(n_noise), "clusters_for_70pct": clusters_for_70,
    },
    "language_blobs": language_blobs,
    "cluster_details": cluster_data,
}
with open(OUTPUT_DIR / "recluster_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"  Saved recluster_summary.json")

# Save combined dataframe and labels for next phase
combined_df["cluster_label"] = labels
combined_df.to_parquet(OUTPUT_DIR / "combined_train_translated.parquet")
np.save(OUTPUT_DIR / "combined_umap30.npy", umap_30)
np.save(OUTPUT_DIR / "combined_umap2d.npy", umap_2d)
print(f"  Saved combined dataset + embeddings for Phase 2")
