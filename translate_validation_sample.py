"""
Translation Validation: Quick sample test
==========================================
1. Pull 1,500 random German tickets from training set
2. Translate subject + body via local LLM
3. Re-embed the translated text
4. Check whether they land near the right English intent clusters

If translated billing complaints land near cluster 3 (English billing),
and translated outages near cluster 1, translation fixes the language blob.
"""

import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import hdbscan
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("output")
LOCAL_LLM_URL = "http://127.0.0.1:8033/v1/chat/completions"
SAMPLE_SIZE = 1500
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ── Load existing data ────────────────────────────────────────────────────────

print("Loading data...")
train_df = pd.read_parquet(OUTPUT_DIR / "train_split.parquet")
embeddings = np.load(OUTPUT_DIR / "train_embeddings.npy")
umap_data = np.load(OUTPUT_DIR / "train_umap_embeddings.npz")
umap_30 = umap_data["dim_30"]

# Recreate HDBSCAN labels (best config: dim=30, min_cs=200)
hdb = hdbscan.HDBSCAN(
    min_cluster_size=200, min_samples=10,
    metric="euclidean", cluster_selection_method="eom", core_dist_n_jobs=-1,
)
hdb_labels = hdb.fit_predict(umap_30)

# Identify German tickets
de_mask = train_df["language"] == "de"
de_indices = np.where(de_mask)[0]
print(f"Total German training tickets: {len(de_indices):,}")


# ── Step 1: Sample 1,500 German tickets ──────────────────────────────────────

sample_indices = np.random.choice(de_indices, size=SAMPLE_SIZE, replace=False)
sample_df = train_df.iloc[sample_indices].copy()
sample_clusters = hdb_labels[sample_indices]

print(f"\nSampled {SAMPLE_SIZE} German tickets")
print(f"  Cluster distribution of sample:")
cluster_dist = Counter(sample_clusters)
for cid, count in sorted(cluster_dist.items(), key=lambda x: -x[1]):
    label = "NOISE" if cid == -1 else f"cluster {cid}"
    print(f"    {label}: {count}")


# ── Step 2: Translate ─────────────────────────────────────────────────────────

print(f"\nTranslating {SAMPLE_SIZE} tickets (subject + body)...")

cache = {}


def translate_one(text):
    if not text or text == "nan" or pd.isna(text):
        return ""
    text = str(text).strip()
    if not text:
        return ""
    if text in cache:
        return cache[text]

    prompt = f"Translate the following German text to English. Output only the translation, nothing else.\n\n{text}"
    try:
        resp = requests.post(
            LOCAL_LLM_URL,
            json={"messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 1024},
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"]
        result = result.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        cache[text] = result
        return result
    except Exception as e:
        return text  # fallback


# Collect unique texts to translate
unique_subjects = list(set(sample_df["subject"].dropna().astype(str).unique()))
unique_bodies = list(set(sample_df["body"].dropna().astype(str).unique()))
all_texts = unique_subjects + unique_bodies
print(f"  Unique texts to translate: {len(unique_subjects)} subjects + {len(unique_bodies)} bodies = {len(all_texts)}")

t0 = time.time()
n_done = 0

# Use 4 workers for parallel translation
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(translate_one, t): t for t in all_texts}
    for future in as_completed(futures):
        n_done += 1
        if n_done % 200 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (len(all_texts) - n_done) / rate
            print(f"    [{n_done:,}/{len(all_texts):,}] {rate:.1f}/s, ETA {eta:.0f}s")

elapsed = time.time() - t0
print(f"  Translated {len(all_texts):,} texts in {elapsed:.1f}s ({len(all_texts)/elapsed:.1f}/s)")

# Apply translations to sample
sample_df["subject_en"] = sample_df["subject"].apply(
    lambda x: cache.get(str(x), str(x)) if pd.notna(x) else "")
sample_df["body_en"] = sample_df["body"].apply(
    lambda x: cache.get(str(x), str(x)) if pd.notna(x) else "")
sample_df["text_en"] = (sample_df["subject_en"] + " " + sample_df["body_en"]).str.strip()


# ── Step 3: Re-embed translated text ─────────────────────────────────────────

print("\nEmbedding translated sample...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
translated_embeddings = model.encode(
    sample_df["text_en"].tolist(),
    batch_size=256, show_progress_bar=False, normalize_embeddings=True,
)
print(f"  Shape: {translated_embeddings.shape}")


# ── Step 4: Check where translated tickets land ──────────────────────────────

print("\nChecking cluster assignment of translated tickets...")

# Get cluster centroids from the original embeddings (in raw 384-dim space)
# For each cluster, compute the mean embedding of its members
cluster_ids = sorted(set(hdb_labels) - {-1})
centroids = {}
for cid in cluster_ids:
    mask = hdb_labels == cid
    centroids[cid] = embeddings[mask].mean(axis=0)

centroid_matrix = np.array([centroids[cid] for cid in cluster_ids])
# Normalize centroids
centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
centroid_matrix = centroid_matrix / centroid_norms

# For each translated ticket, find nearest centroid (cosine similarity)
similarities = translated_embeddings @ centroid_matrix.T  # (1500, n_clusters)
nearest_cluster_idx = similarities.argmax(axis=1)
nearest_cluster_ids = np.array([cluster_ids[i] for i in nearest_cluster_idx])
nearest_cluster_sims = similarities.max(axis=1)

# Also get their ORIGINAL cluster assignment (before translation)
original_clusters = sample_clusters

# Key question: do translated tickets from the German blob (cluster 9)
# now disperse into meaningful English clusters?
print("\n" + "=" * 70)
print("RESULTS: Where do translated German tickets land?")
print("=" * 70)

# Overall dispersion
print(f"\n  All {SAMPLE_SIZE} translated German tickets → nearest cluster:")
reassigned = Counter(nearest_cluster_ids)
for cid, count in sorted(reassigned.items(), key=lambda x: -x[1]):
    pct = count / SAMPLE_SIZE * 100
    # Get dominant label for this cluster
    cmask = hdb_labels == cid
    if cmask.sum() > 0:
        top_tag = Counter(train_df.loc[cmask, "tag_1"].values).most_common(1)[0][0]
        top_type = Counter(train_df.loc[cmask, "type"].values).most_common(1)[0][0]
    else:
        top_tag = top_type = "?"
    print(f"    cluster {cid:3d}: {count:4d} ({pct:5.1f}%) [type={top_type}, tag1={top_tag}]")

# Focus on cluster 9 members (the German blob)
c9_mask = original_clusters == 9
n_c9 = c9_mask.sum()
if n_c9 > 0:
    print(f"\n  Cluster 9 members specifically ({n_c9} tickets) → after translation:")
    c9_reassigned = Counter(nearest_cluster_ids[c9_mask])
    for cid, count in sorted(c9_reassigned.items(), key=lambda x: -x[1]):
        pct = count / n_c9 * 100
        cmask = hdb_labels == cid
        if cmask.sum() > 0:
            top_tag = Counter(train_df.loc[cmask, "tag_1"].values).most_common(1)[0][0]
            top_type = Counter(train_df.loc[cmask, "type"].values).most_common(1)[0][0]
        else:
            top_tag = top_type = "?"
        print(f"    → cluster {cid:3d}: {count:4d} ({pct:5.1f}%) [type={top_type}, tag1={top_tag}]")

    stayed_in_9 = c9_reassigned.get(9, 0)
    dispersed = n_c9 - stayed_in_9
    print(f"\n    Dispersed out of cluster 9: {dispersed}/{n_c9} ({dispersed/n_c9*100:.1f}%)")

# Similarity stats
print(f"\n  Cosine similarity to nearest centroid:")
print(f"    Mean: {nearest_cluster_sims.mean():.3f}")
print(f"    Median: {np.median(nearest_cluster_sims):.3f}")
print(f"    Min: {nearest_cluster_sims.min():.3f}")
print(f"    >0.7: {(nearest_cluster_sims > 0.7).sum()} ({(nearest_cluster_sims > 0.7).sum()/SAMPLE_SIZE*100:.1f}%)")
print(f"    >0.8: {(nearest_cluster_sims > 0.8).sum()} ({(nearest_cluster_sims > 0.8).sum()/SAMPLE_SIZE*100:.1f}%)")

# Verdict
print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")
if n_c9 > 0:
    dispersal_rate = dispersed / n_c9
    if dispersal_rate > 0.7:
        print(f"\n  PASS: {dispersal_rate*100:.0f}% of cluster-9 tickets dispersed into intent clusters.")
        print("  Translation fixes the language blob. Proceed with full German translation.")
    elif dispersal_rate > 0.4:
        print(f"\n  PARTIAL: {dispersal_rate*100:.0f}% dispersed. Translation helps but doesn't fully fix it.")
        print("  May need a different embedding model or additional preprocessing.")
    else:
        print(f"\n  FAIL: Only {dispersal_rate*100:.0f}% dispersed. Problem is deeper than language.")
        print("  Consider a different embedding model or cluster-per-language approach.")

# Save results
results = {
    "sample_size": SAMPLE_SIZE,
    "n_cluster9_in_sample": int(n_c9),
    "dispersal_rate": round(dispersed / n_c9, 3) if n_c9 > 0 else None,
    "cluster_reassignment": {str(k): int(v) for k, v in reassigned.items()},
    "cluster9_reassignment": {str(k): int(v) for k, v in c9_reassigned.items()} if n_c9 > 0 else {},
    "similarity_stats": {
        "mean": round(float(nearest_cluster_sims.mean()), 3),
        "median": round(float(np.median(nearest_cluster_sims)), 3),
    },
    "translation_time_sec": round(elapsed, 1),
}
with open(OUTPUT_DIR / "translation_validation.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to output/translation_validation.json")
