"""
Translate 1,500 German sample with persistent cache saves.
Saves cache every 200 translations so it's resumable if interrupted.
"""

import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = Path("output")
CACHE_PATH = OUTPUT_DIR / "translations_cache.json"
LOCAL_LLM_URL = "http://127.0.0.1:8033/v1/chat/completions"
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Load train data
train_df = pd.read_parquet(OUTPUT_DIR / "train_split.parquet")
de_mask = train_df["language"] == "de"
de_indices = np.where(de_mask)[0]

# Same sample as validation script
sample_indices = np.random.choice(de_indices, size=1500, replace=False)
sample_df = train_df.iloc[sample_indices].copy()

# Collect unique texts
unique_subjects = list(set(sample_df["subject"].dropna().astype(str).unique()))
unique_bodies = list(set(sample_df["body"].dropna().astype(str).unique()))
all_texts = unique_subjects + unique_bodies

# Load existing cache
if CACHE_PATH.exists():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    print(f"Loaded {len(cache):,} cached translations")
else:
    cache = {}

# Filter to untranslated
to_translate = [t for t in all_texts if t not in cache and t != "nan"]
print(f"Total unique texts: {len(all_texts):,}")
print(f"Already cached: {len(all_texts) - len(to_translate):,}")
print(f"Need to translate: {len(to_translate):,}")

if not to_translate:
    print("Nothing to translate — cache is complete!")
else:
    def translate_one(text):
        if not text or text == "nan":
            return text, ""
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
            return text, result
        except Exception as e:
            return text, text  # fallback

    t0 = time.time()
    n_done = 0
    n_since_save = 0

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(translate_one, t): t for t in to_translate}
        for future in as_completed(futures):
            orig, translated = future.result()
            cache[orig] = translated
            n_done += 1
            n_since_save += 1

            if n_since_save >= 200:
                # Save checkpoint
                with open(CACHE_PATH, "w") as f:
                    json.dump(cache, f, ensure_ascii=False)
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = len(to_translate) - n_done
                eta = remaining / rate if rate > 0 else 0
                print(f"  [{n_done:,}/{len(to_translate):,}] {rate:.1f}/s, "
                      f"ETA {eta:.0f}s ({eta/60:.1f}min) — cache saved ({len(cache):,})")
                n_since_save = 0

    # Final save
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False)
    elapsed = time.time() - t0
    print(f"\nDone: {len(to_translate):,} translations in {elapsed:.1f}s ({len(to_translate)/elapsed:.1f}/s)")

print(f"Cache total: {len(cache):,} entries saved to {CACHE_PATH}")
