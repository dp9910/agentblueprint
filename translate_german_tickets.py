"""
Translate German tickets to English using local LLM (translategemma on port 8033).
Translates subject and body separately, saves results for re-embedding.
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = Path("output")
TRANSLATE_CACHE = OUTPUT_DIR / "translations_cache.json"
LOCAL_LLM_URL = "http://127.0.0.1:8033/v1/chat/completions"

# Load train split
train_df = pd.read_parquet(OUTPUT_DIR / "train_split.parquet")
test_df = pd.read_parquet(OUTPUT_DIR / "test_split.parquet")

# Identify German rows
de_train_mask = train_df["language"] == "de"
de_test_mask = test_df["language"] == "de"
n_de_train = de_train_mask.sum()
n_de_test = de_test_mask.sum()
print(f"German tickets to translate: {n_de_train:,} train + {n_de_test:,} test = {n_de_train + n_de_test:,} total")

# Load existing cache if present
if TRANSLATE_CACHE.exists():
    with open(TRANSLATE_CACHE) as f:
        cache = json.load(f)
    print(f"Loaded {len(cache):,} cached translations")
else:
    cache = {}


def translate(text, max_retries=3):
    """Translate German text to English via local LLM."""
    if not text or text == "nan" or pd.isna(text):
        return ""

    text = str(text).strip()
    if not text:
        return ""

    # Check cache
    if text in cache:
        return cache[text]

    prompt = f"Translate the following German text to English. Output only the translation, nothing else.\n\n{text}"

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                LOCAL_LLM_URL,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 1024,
                },
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"]
            # Strip common artifacts
            result = result.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            cache[text] = result
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"  FAILED after {max_retries} attempts: {str(e)[:80]}")
                return text  # fallback to original


def translate_batch_sequential(texts, label=""):
    """Translate a list of texts sequentially (local LLM is likely single-threaded)."""
    results = []
    n_cached = 0
    n_translated = 0
    t0 = time.time()

    for i, text in enumerate(texts):
        if str(text) in cache:
            n_cached += 1
        else:
            n_translated += 1

        result = translate(text)
        results.append(result)

        # Progress every 500
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(texts) - i - 1) / rate
            print(f"  {label} [{i+1:,}/{len(texts):,}] "
                  f"{rate:.1f} texts/s, ETA {eta/60:.1f}min "
                  f"(cached={n_cached}, translated={n_translated})")

    elapsed = time.time() - t0
    print(f"  {label} Done: {len(texts):,} texts in {elapsed:.1f}s "
          f"({n_cached} cached, {n_translated} new)")
    return results


# ── Collect unique German texts to translate ──────────────────────────────────
# Translate unique texts only, then map back — avoids duplicate work

all_de = pd.concat([
    train_df.loc[de_train_mask, ["subject", "body"]],
    test_df.loc[de_test_mask, ["subject", "body"]],
])

unique_subjects = set(all_de["subject"].dropna().unique())
unique_bodies = set(all_de["body"].dropna().unique())

# Filter out already cached
new_subjects = [s for s in unique_subjects if str(s) not in cache]
new_bodies = [b for b in unique_bodies if str(b) not in cache]
print(f"\nUnique German texts: {len(unique_subjects):,} subjects, {len(unique_bodies):,} bodies")
print(f"Already cached: {len(unique_subjects) - len(new_subjects):,} subjects, "
      f"{len(unique_bodies) - len(new_bodies):,} bodies")
print(f"Need to translate: {len(new_subjects):,} subjects, {len(new_bodies):,} bodies")

# Translate subjects first (shorter, faster)
print(f"\n--- Translating subjects ---")
translate_batch_sequential(new_subjects, label="subjects")

# Save cache checkpoint
with open(TRANSLATE_CACHE, "w") as f:
    json.dump(cache, f, ensure_ascii=False)
print(f"  Cache checkpoint saved ({len(cache):,} entries)")

# Translate bodies
print(f"\n--- Translating bodies ---")
translate_batch_sequential(new_bodies, label="bodies")

# Save final cache
with open(TRANSLATE_CACHE, "w") as f:
    json.dump(cache, f, ensure_ascii=False)
print(f"  Final cache saved ({len(cache):,} entries)")


# ── Apply translations to dataframes ─────────────────────────────────────────

def apply_translations(df, mask):
    """Replace German subject/body with English translations."""
    df = df.copy()
    for col in ["subject", "body"]:
        translated = df.loc[mask, col].apply(lambda x: cache.get(str(x), str(x)) if pd.notna(x) else "")
        df.loc[mask, col] = translated
    # Rebuild text column
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df["text"] = df["text"].str.strip()
    return df


print("\n--- Applying translations ---")
train_translated = apply_translations(train_df, de_train_mask)
test_translated = apply_translations(test_df, de_test_mask)

# Verify
sample_idx = train_df[de_train_mask].index[0]
print(f"\n  Sample verification (train idx={sample_idx}):")
print(f"    Original subject: {str(train_df.loc[sample_idx, 'subject'])[:80]}")
print(f"    Translated:       {str(train_translated.loc[sample_idx, 'subject'])[:80]}")
print(f"    Original body:    {str(train_df.loc[sample_idx, 'body'])[:80]}")
print(f"    Translated:       {str(train_translated.loc[sample_idx, 'body'])[:80]}")

# Save translated splits
train_translated.to_parquet(OUTPUT_DIR / "train_split_translated.parquet")
test_translated.to_parquet(OUTPUT_DIR / "test_split_translated.parquet")
print(f"\n  Saved translated splits to output/")
print(f"  Total translations in cache: {len(cache):,}")
