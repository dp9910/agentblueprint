"""
Phase 3 — Step 1: Prepare Test Set + Build Router
===================================================
1. Translate German test tickets (using existing cache)
2. Embed test tickets
3. Build embedding-based router (cosine similarity to cluster centroids)
4. Route all test tickets and save routing decisions

NO API calls needed — runs entirely on local embedding model + cached translations.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = Path("output")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("PHASE 3 — STEP 1: PREPARE TEST SET + BUILD ROUTER")
print("=" * 60)

# ── Load training data (for centroids) ───────────────────────────────────────

print("\n[1/5] Loading training data and blueprints...")
train_df = pd.read_parquet(OUTPUT_DIR / "combined_train_translated.parquet")
train_embeddings = np.load(OUTPUT_DIR / "combined_embeddings.npy")
labels = train_df["cluster_label"].values

with open(OUTPUT_DIR / "blueprints.json") as f:
    blueprints = json.load(f)

blueprint_cluster_ids = [int(cid) for cid in blueprints.keys() if "steps" in blueprints[cid]]
print(f"  Training set: {len(train_df):,} tickets")
print(f"  Blueprint clusters: {blueprint_cluster_ids}")

# ── Compute cluster centroids (raw 384-dim space) ────────────────────────────

print("\n[2/5] Computing cluster centroids...")
centroids = {}
for cid in blueprint_cluster_ids:
    mask = labels == cid
    centroid = train_embeddings[mask].mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)  # normalize
    centroids[cid] = centroid
    print(f"  Cluster {cid}: centroid from {mask.sum():,} tickets")

# ── Prepare test set ─────────────────────────────────────────────────────────

print("\n[3/5] Preparing test set...")
test_df = pd.read_parquet(OUTPUT_DIR / "test_split.parquet")
print(f"  Test set: {len(test_df):,} tickets")

# Load translation cache
with open(OUTPUT_DIR / "translations_cache.json") as f:
    cache = json.load(f)
print(f"  Translation cache: {len(cache):,} entries")

# Classify test tickets by language
test_df["type"] = test_df["type"].fillna("Unknown")
test_df["queue"] = test_df["queue"].fillna("Unknown")
test_df["tag_1"] = test_df["tag_1"].fillna("none")

en_mask = test_df["language"] == "en"
de_mask = test_df["language"] == "de"
print(f"  English test tickets: {en_mask.sum():,}")
print(f"  German test tickets: {de_mask.sum():,}")

# Translate German test tickets that have cached translations
de_test = test_df[de_mask].copy()
de_test["subj_in_cache"] = de_test["subject"].apply(lambda x: str(x) in cache if pd.notna(x) else True)
de_test["body_in_cache"] = de_test["body"].apply(lambda x: str(x) in cache if pd.notna(x) else True)
de_translatable = de_test[de_test["subj_in_cache"] & de_test["body_in_cache"]].copy()
print(f"  German test tickets with cached translations: {len(de_translatable):,}")

# Apply translations
de_translatable["subject"] = de_translatable["subject"].apply(
    lambda x: cache.get(str(x), str(x)) if pd.notna(x) else "")
de_translatable["body"] = de_translatable["body"].apply(
    lambda x: cache.get(str(x), str(x)) if pd.notna(x) else "")
de_translatable["translated"] = True

# English test tickets
en_test = test_df[en_mask].copy()
en_test["translated"] = False

# Combine
test_combined = pd.concat([en_test, de_translatable], ignore_index=True)
test_combined["text"] = test_combined["subject"].fillna("") + " " + test_combined["body"].fillna("")
test_combined["text"] = test_combined["text"].str.strip()
print(f"  Combined test set: {len(test_combined):,} ({(~test_combined['translated']).sum():,} EN + "
      f"{test_combined['translated'].sum():,} DE translated)")

# ── Embed test set ───────────────────────────────────────────────────────────

print("\n[4/5] Embedding test set...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
t0 = time.time()
test_embeddings = model.encode(
    test_combined["text"].tolist(),
    batch_size=256, show_progress_bar=True, normalize_embeddings=True,
)
print(f"  Embedded {len(test_embeddings):,} tickets in {time.time()-t0:.1f}s")
print(f"  Shape: {test_embeddings.shape}")

# ── Route test tickets ───────────────────────────────────────────────────────

print("\n[5/5] Routing test tickets to blueprints...")

# Compute cosine similarity to each centroid
centroid_ids = sorted(centroids.keys())
centroid_matrix = np.array([centroids[cid] for cid in centroid_ids])
similarities = cosine_similarity(test_embeddings, centroid_matrix)

# For each ticket: best cluster, confidence, 2nd best
best_idx = similarities.argmax(axis=1)
best_sim = similarities.max(axis=1)

# Second best for confidence gap
sorted_sims = np.sort(similarities, axis=1)
second_best_sim = sorted_sims[:, -2]
confidence_gap = best_sim - second_best_sim

# Assign routes
routes = []
for i in range(len(test_combined)):
    assigned_cluster = centroid_ids[best_idx[i]]
    sim = best_sim[i]
    gap = confidence_gap[i]

    routes.append({
        "assigned_cluster": int(assigned_cluster),
        "similarity": round(float(sim), 4),
        "confidence_gap": round(float(gap), 4),
    })

test_combined["routed_cluster"] = [r["assigned_cluster"] for r in routes]
test_combined["route_similarity"] = [r["similarity"] for r in routes]
test_combined["route_confidence"] = [r["confidence_gap"] for r in routes]

# Route distribution
route_dist = Counter(test_combined["routed_cluster"])
print(f"\n  Route distribution:")
for cid in centroid_ids:
    n = route_dist.get(cid, 0)
    print(f"    Cluster {cid} ({blueprints[str(cid)]['intent'][:50]}): {n:,} tickets ({n/len(test_combined)*100:.1f}%)")

# Confidence stats
print(f"\n  Similarity stats: mean={best_sim.mean():.3f}, min={best_sim.min():.3f}, "
      f"median={np.median(best_sim):.3f}, max={best_sim.max():.3f}")
print(f"  Confidence gap: mean={confidence_gap.mean():.3f}, min={confidence_gap.min():.3f}")

# Breakdown by language
for lang, lang_mask in [("English (native)", ~test_combined["translated"].values),
                         ("German (translated)", test_combined["translated"].values)]:
    if lang_mask.sum() == 0:
        continue
    print(f"\n  {lang} ({lang_mask.sum():,} tickets):")
    print(f"    Similarity: mean={best_sim[lang_mask].mean():.3f}, "
          f"median={np.median(best_sim[lang_mask]):.3f}")
    lang_routes = Counter(test_combined.loc[lang_mask, "routed_cluster"])
    for cid in centroid_ids:
        n = lang_routes.get(cid, 0)
        if n > 0:
            print(f"    → Cluster {cid}: {n}")

# ── Sample test tickets for experiment ───────────────────────────────────────

# For the experiment, we need a balanced sample:
# - Stratified by routed cluster
# - Mix of native EN and translated DE
# - Small enough to fit within API rate limits

EXPERIMENT_SIZE = 30  # 30 tickets = manageable within free-tier limits

# Stratified sample: proportional to route distribution
experiment_indices = []
for cid in centroid_ids:
    cid_mask = test_combined["routed_cluster"] == cid
    cid_indices = np.where(cid_mask)[0]
    n_sample = max(2, int(EXPERIMENT_SIZE * cid_mask.sum() / len(test_combined)))
    if len(cid_indices) > 0:
        chosen = np.random.choice(cid_indices, size=min(n_sample, len(cid_indices)), replace=False)
        experiment_indices.extend(chosen)

# Ensure we have exactly EXPERIMENT_SIZE (add random extras or trim)
if len(experiment_indices) > EXPERIMENT_SIZE:
    experiment_indices = list(np.random.choice(experiment_indices, EXPERIMENT_SIZE, replace=False))
elif len(experiment_indices) < EXPERIMENT_SIZE:
    remaining = set(range(len(test_combined))) - set(experiment_indices)
    extra = np.random.choice(list(remaining), EXPERIMENT_SIZE - len(experiment_indices), replace=False)
    experiment_indices.extend(extra)

experiment_df = test_combined.iloc[experiment_indices].copy().reset_index(drop=True)
experiment_embeds = test_embeddings[experiment_indices]

print(f"\n  Experiment sample: {len(experiment_df)} tickets")
print(f"    English: {(~experiment_df['translated']).sum()}")
print(f"    German (translated): {experiment_df['translated'].sum()}")
print(f"    Route distribution: {dict(Counter(experiment_df['routed_cluster']))}")

# ── Save everything ──────────────────────────────────────────────────────────

test_combined.to_parquet(OUTPUT_DIR / "test_routed.parquet")
np.save(OUTPUT_DIR / "test_embeddings.npy", test_embeddings)
np.save(OUTPUT_DIR / "cluster_centroids.npy", centroid_matrix)

experiment_df.to_parquet(OUTPUT_DIR / "experiment_sample.parquet")
np.save(OUTPUT_DIR / "experiment_embeddings.npy", experiment_embeds)

with open(OUTPUT_DIR / "routing_summary.json", "w") as f:
    json.dump({
        "test_set_size": len(test_combined),
        "english_count": int((~test_combined["translated"]).sum()),
        "german_translated_count": int(test_combined["translated"].sum()),
        "blueprint_clusters": blueprint_cluster_ids,
        "route_distribution": {str(k): int(v) for k, v in route_dist.items()},
        "similarity_stats": {
            "mean": round(float(best_sim.mean()), 4),
            "median": round(float(np.median(best_sim)), 4),
            "min": round(float(best_sim.min()), 4),
            "max": round(float(best_sim.max()), 4),
        },
        "experiment_size": len(experiment_df),
    }, f, indent=2)

print(f"\n{'=' * 60}")
print("STEP 1 COMPLETE")
print(f"{'=' * 60}")
print(f"  Saved:")
print(f"    output/test_routed.parquet — full test set with routes")
print(f"    output/test_embeddings.npy — test embeddings")
print(f"    output/cluster_centroids.npy — centroid vectors")
print(f"    output/experiment_sample.parquet — {len(experiment_df)} tickets for experiment")
print(f"    output/routing_summary.json — routing stats")
print(f"\n  NEXT: Run phase3_experiment.py (needs Gemini API)")
