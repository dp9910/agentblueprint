"""
Phase 1 Sanity Checks
=====================
Three checks before proceeding to Phase 2:

1. Escalation ceiling — exact % that must go to full-agent path
2. Type-purity per cluster — misroute early warning
3. Eyeball test — 10 random tickets from each top-9 cluster
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("output")

# ── Load data ─────────────────────────────────────────────────────────────────

train_df = pd.read_parquet(OUTPUT_DIR / "train_split.parquet")

# Recreate best HDBSCAN labels (dim=30, min_cs=200)
import hdbscan

umap_data = np.load(OUTPUT_DIR / "train_umap_embeddings.npz")
umap_30 = umap_data["dim_30"]

hdb = hdbscan.HDBSCAN(
    min_cluster_size=200,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    core_dist_n_jobs=-1,
)
hdb_labels = hdb.fit_predict(umap_30)
hdb_probabilities = hdb.probabilities_

n_total = len(train_df)
cluster_sizes = Counter(hdb_labels)
n_noise = cluster_sizes.pop(-1, 0)
n_clustered = n_total - n_noise

# Sort clusters by size descending
sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1: Escalation Ceiling
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("CHECK 1: ESCALATION CEILING")
print("=" * 70)

noise_pct = n_noise / n_total * 100
clustered_pct = n_clustered / n_total * 100
top9_total = sum(size for _, size in sorted_clusters[:9])
top9_pct = top9_total / n_total * 100
tail_clusters_total = sum(size for _, size in sorted_clusters[9:])
tail_pct = tail_clusters_total / n_total * 100

print(f"\n  Total training tickets:     {n_total:,}")
print(f"\n  Tier breakdown:")
print(f"    Top 9 clusters (blueprint):  {top9_total:,}  ({top9_pct:.1f}%)")
print(f"    Remaining {len(sorted_clusters)-9} clusters:       {tail_clusters_total:,}  ({tail_pct:.1f}%)")
print(f"    HDBSCAN noise (outliers):    {n_noise:,}  ({noise_pct:.1f}%)")
print(f"\n  Maximum possible savings ceiling:")
print(f"    Best case: {top9_pct:.1f}% handled by blueprints (cheap path)")
print(f"    Minimum full-agent spend: {noise_pct:.1f}% (noise) + potentially some from tail clusters")
print(f"    If tail clusters also get blueprints: {noise_pct:.1f}% escalation floor")
print(f"    If only top-9 get blueprints: {100 - top9_pct:.1f}% escalation floor")

# Also look at confidence distribution — low-confidence assignments are risky
print(f"\n  HDBSCAN membership probabilities (non-noise):")
non_noise_probs = hdb_probabilities[hdb_labels != -1]
for threshold in [0.5, 0.7, 0.8, 0.9, 0.95]:
    count_above = (non_noise_probs >= threshold).sum()
    print(f"    prob >= {threshold}: {count_above:,} ({count_above/n_total*100:.1f}% of total)")

# Low-confidence clustered points are effectively soft outliers
soft_outlier_mask = (hdb_labels != -1) & (hdb_probabilities < 0.5)
n_soft_outliers = soft_outlier_mask.sum()
print(f"\n  Soft outliers (clustered but prob < 0.5): {n_soft_outliers:,} ({n_soft_outliers/n_total*100:.1f}%)")
print(f"  Effective escalation floor (noise + soft outliers): "
      f"{(n_noise + n_soft_outliers):,} ({(n_noise + n_soft_outliers)/n_total*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2: TYPE-PURITY PER CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CHECK 2: TYPE-PURITY PER CLUSTER (top 9)")
print("=" * 70)

type_labels = train_df["type"].values
queue_labels = train_df["queue"].values
tag1_labels = train_df["tag_1"].values
lang_labels = train_df["language"].values

print(f"\n  {'#':>2s}  {'Cluster':>7s}  {'Size':>6s}  {'Type Purity':>12s}  {'Queue Purity':>13s}  "
      f"{'Tag1 Purity':>12s}  {'Lang Split':>15s}  Dominant Labels")
print(f"  {'-' * 130}")

purity_data = []
for rank, (cid, size) in enumerate(sorted_clusters[:9], 1):
    mask = hdb_labels == cid

    # Type distribution
    type_counts = Counter(type_labels[mask])
    top_type, top_type_count = type_counts.most_common(1)[0]
    type_purity = top_type_count / size

    # Queue distribution
    queue_counts = Counter(queue_labels[mask])
    top_queue, top_queue_count = queue_counts.most_common(1)[0]
    queue_purity = top_queue_count / size

    # Tag1 distribution
    tag_counts = Counter(tag1_labels[mask])
    top_tag, top_tag_count = tag_counts.most_common(1)[0]
    tag_purity = top_tag_count / size

    # Language split
    lang_counts = Counter(lang_labels[mask])
    en_pct = lang_counts.get("en", 0) / size * 100
    de_pct = lang_counts.get("de", 0) / size * 100

    # Full type distribution for this cluster
    type_dist = ", ".join(f"{t}:{c}" for t, c in type_counts.most_common(3))

    safety = "SAFE" if type_purity >= 0.7 else ("WATCH" if type_purity >= 0.5 else "SPLIT")

    print(f"  {rank:2d}  {cid:7d}  {size:6,}  {type_purity:10.1%} [{safety:5s}]  "
          f"{queue_purity:11.1%}    {tag_purity:10.1%}    "
          f"en:{en_pct:4.0f}% de:{de_pct:4.0f}%  "
          f"type=[{type_dist}]")

    purity_data.append({
        "rank": rank,
        "cluster_id": int(cid),
        "size": int(size),
        "type_purity": round(type_purity, 3),
        "queue_purity": round(queue_purity, 3),
        "tag1_purity": round(tag_purity, 3),
        "dominant_type": top_type,
        "dominant_queue": top_queue,
        "dominant_tag1": top_tag,
        "type_distribution": dict(type_counts),
        "queue_distribution": {k: v for k, v in queue_counts.most_common(5)},
        "safety": safety,
    })

print(f"\n  Legend: SAFE = type purity >= 70% | WATCH = 50-70% | SPLIT = < 50%")
n_safe = sum(1 for p in purity_data if p["safety"] == "SAFE")
n_watch = sum(1 for p in purity_data if p["safety"] == "WATCH")
n_split = sum(1 for p in purity_data if p["safety"] == "SPLIT")
print(f"  Summary: {n_safe} SAFE, {n_watch} WATCH, {n_split} SPLIT out of 9 clusters")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3: EYEBALL TEST — 10 RANDOM TICKETS PER TOP-9 CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CHECK 3: EYEBALL TEST — 10 RANDOM TICKETS PER TOP-9 CLUSTER")
print("=" * 70)

np.random.seed(42)

eyeball_data = {}
for rank, (cid, size) in enumerate(sorted_clusters[:9], 1):
    mask = hdb_labels == cid
    indices = np.where(mask)[0]
    sample_idx = np.random.choice(indices, size=min(10, len(indices)), replace=False)

    print(f"\n{'─' * 70}")
    print(f"CLUSTER {cid} (rank #{rank}, n={size:,})")
    print(f"  Dominant: type={purity_data[rank-1]['dominant_type']} | "
          f"queue={purity_data[rank-1]['dominant_queue']} | "
          f"tag1={purity_data[rank-1]['dominant_tag1']}")
    print(f"{'─' * 70}")

    cluster_samples = []
    for i, idx in enumerate(sample_idx, 1):
        row = train_df.iloc[idx]
        subject = str(row["subject"])[:80]
        body = str(row["body"])[:200].replace("\n", " ")
        lang = row["language"]
        ticket_type = row["type"]
        queue = row["queue"]
        tag1 = row["tag_1"]
        prob = hdb_probabilities[idx]

        print(f"\n  [{i:2d}] lang={lang} | type={ticket_type} | queue={queue} | tag1={tag1} | prob={prob:.2f}")
        print(f"       Subject: {subject}")
        print(f"       Body:    {body}...")

        cluster_samples.append({
            "index": int(idx),
            "language": lang,
            "type": ticket_type,
            "queue": str(queue),
            "tag1": str(tag1),
            "subject": str(row["subject"]),
            "body_preview": str(row["body"])[:300],
            "probability": round(float(prob), 3),
        })

    eyeball_data[f"cluster_{cid}_rank{rank}"] = cluster_samples


# ── Save results ──────────────────────────────────────────────────────────────

results = {
    "escalation_ceiling": {
        "noise_count": int(n_noise),
        "noise_pct": round(noise_pct, 2),
        "soft_outlier_count": int(n_soft_outliers),
        "soft_outlier_pct": round(n_soft_outliers / n_total * 100, 2),
        "effective_escalation_pct": round((n_noise + n_soft_outliers) / n_total * 100, 2),
        "top9_coverage_pct": round(top9_pct, 2),
    },
    "purity": purity_data,
    "eyeball_samples": eyeball_data,
}

with open(OUTPUT_DIR / "phase1_sanity_checks.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n\n{'=' * 70}")
print("SANITY CHECK SUMMARY")
print(f"{'=' * 70}")
print(f"\n  1. Escalation ceiling: {noise_pct:.1f}% hard noise + {n_soft_outliers/n_total*100:.1f}% soft outliers "
      f"= {(n_noise + n_soft_outliers)/n_total*100:.1f}% effective floor")
print(f"  2. Type purity: {n_safe}/9 clusters are SAFE (>=70% purity), {n_watch} WATCH, {n_split} SPLIT")
print(f"  3. Eyeball samples printed above — review for coherence")
print(f"\n  Full results saved to output/phase1_sanity_checks.json")
