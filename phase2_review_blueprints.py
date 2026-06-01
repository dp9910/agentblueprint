"""
Phase 2 — Blueprint Review Gate (Batched)
==========================================
For each blueprint, test it against 10 in-cluster + 5 out-of-cluster tickets.
Uses BATCHED prompts: 2 API calls per blueprint (in-cluster batch + out-of-cluster batch).
Total: 10 API calls instead of 75.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from google import genai

load_dotenv()

OUTPUT_DIR = Path("output")
RANDOM_SEED = 99
np.random.seed(RANDOM_SEED)

client = genai.Client(api_key=os.getenv("gemini_api_key"))

# Alternate between models to spread rate limits
MODELS = ["gemini-3.5-flash", "gemini-2.5-flash"]
model_idx = [0]

def call_gemini(prompt, max_retries=10):
    """Call Gemini with rate-limit retry, alternating models."""
    for attempt in range(max_retries):
        model = MODELS[model_idx[0] % len(MODELS)]
        try:
            response = client.models.generate_content(
                model=model, contents=prompt,
            )
            model_idx[0] += 1  # rotate for next call
            return response, model
        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "overloaded"]):
                # Try the other model first
                model_idx[0] += 1
                other_model = MODELS[model_idx[0] % len(MODELS)]
                try:
                    response = client.models.generate_content(
                        model=other_model, contents=prompt,
                    )
                    model_idx[0] += 1
                    return response, other_model
                except Exception:
                    pass
                wait = 30 * (attempt + 1)
                print(f"      Both models busy, waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")

# Load data
train_df = pd.read_parquet(OUTPUT_DIR / "combined_train_translated.parquet")
labels = train_df["cluster_label"].values

with open(OUTPUT_DIR / "blueprints.json") as f:
    blueprints = json.load(f)

print("=" * 60)
print("PHASE 2: BLUEPRINT REVIEW GATE (BATCHED)")
print("=" * 60)

total_input_tokens = 0
total_output_tokens = 0
total_thinking_tokens = 0

review_results = {}

for cid_str, blueprint in blueprints.items():
    cid = int(cid_str)
    if "steps" not in blueprint:
        print(f"\n  Skipping cluster {cid} — parse error")
        continue

    mask = labels == cid
    cluster_indices = np.where(mask)[0]

    # Sample tickets
    in_sample_idx = np.random.choice(cluster_indices, size=min(10, len(cluster_indices)), replace=False)
    other_mask = (labels != cid) & (labels != -1)
    other_indices = np.where(other_mask)[0]
    out_sample_idx = np.random.choice(other_indices, size=min(5, len(other_indices)), replace=False)

    # Format blueprint
    bp_text = json.dumps({
        "intent": blueprint["intent"],
        "scope": blueprint["scope"],
        "steps": blueprint["steps"],
        "minimal_context": blueprint["minimal_context"],
        "exit_conditions": blueprint["exit_conditions"],
    }, indent=2)

    print(f"\n{'─' * 60}")
    print(f"CLUSTER {cid} — {blueprint['intent'][:60]}")
    print(f"  {len(blueprint['steps'])} steps, {len(blueprint['exit_conditions'])} exit conditions")
    print(f"{'─' * 60}")

    cluster_results = {"in_cluster": [], "out_of_cluster": []}

    # ── Batch in-cluster tickets ──
    print(f"\n  Evaluating {len(in_sample_idx)} in-cluster tickets (batched)...")
    tickets_block = ""
    ticket_meta = []
    for i, idx in enumerate(in_sample_idx):
        row = train_df.iloc[idx]
        tickets_block += f"\n--- TICKET {i+1} ---\n"
        tickets_block += f"Subject: {row.get('subject', 'N/A')}\n"
        tickets_block += f"Body: {str(row.get('body', ''))[:400]}\n"
        ticket_meta.append({"type": str(row.get("type", "")), "tag1": str(row.get("tag_1", ""))})

    prompt = f"""You are reviewing whether a customer support blueprint is appropriate for specific tickets.

BLUEPRINT:
{bp_text}

TICKETS TO EVALUATE:
{tickets_block}

For EACH ticket (1 through {len(in_sample_idx)}), answer:
1. FITS: Does this ticket fall within the blueprint's stated intent and scope? (yes/no)
2. STEPS_WORK: Could the blueprint's steps produce a reasonable resolution? (yes/no)
3. EXIT_TRIGGERED: Does any exit condition apply (should escalate instead)? (yes/no)

Respond with ONLY a valid JSON array — one object per ticket:
[{{"ticket_num": 1, "fits": true/false, "steps_work": true/false, "exit_triggered": true/false, "exit_reason": "which exit condition or null", "brief_reason": "one sentence"}}]"""

    response, used_model = call_gemini(prompt)
    usage = response.usage_metadata
    total_input_tokens += usage.prompt_token_count or 0
    total_output_tokens += usage.candidates_token_count or 0
    total_thinking_tokens += getattr(usage, 'thoughts_token_count', 0) or 0
    print(f"    (used {used_model})")

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        in_results = json.loads(raw)
        if not isinstance(in_results, list):
            in_results = [in_results]
    except json.JSONDecodeError:
        print(f"    PARSE ERROR on in-cluster batch: {raw[:200]}")
        in_results = [{"fits": None, "steps_work": None, "exit_triggered": None, "parse_error": raw[:100]}
                      for _ in in_sample_idx]

    for i, result in enumerate(in_results):
        if i < len(ticket_meta):
            result["ticket_type"] = ticket_meta[i]["type"]
            result["ticket_tag1"] = ticket_meta[i]["tag1"]
        cluster_results["in_cluster"].append(result)
        status = "OK" if result.get("fits") and result.get("steps_work") else "MISS"
        print(f"    [{i+1:2d}] {status} | fits={result.get('fits')} steps_work={result.get('steps_work')} "
              f"exit={result.get('exit_triggered')} | {ticket_meta[i]['type']}/{ticket_meta[i]['tag1']}")

    time.sleep(5)  # brief pause between calls

    # ── Batch out-of-cluster tickets ──
    print(f"\n  Evaluating {len(out_sample_idx)} out-of-cluster tickets (batched)...")
    tickets_block = ""
    ticket_meta_out = []
    for i, idx in enumerate(out_sample_idx):
        row = train_df.iloc[idx]
        tickets_block += f"\n--- TICKET {i+1} ---\n"
        tickets_block += f"Subject: {row.get('subject', 'N/A')}\n"
        tickets_block += f"Body: {str(row.get('body', ''))[:400]}\n"
        ticket_meta_out.append({
            "type": str(row.get("type", "")),
            "tag1": str(row.get("tag_1", "")),
            "source_cluster": int(labels[idx]),
        })

    prompt = f"""You are reviewing whether a customer support blueprint is appropriate for specific tickets.
These tickets are from DIFFERENT clusters — they should ideally NOT fit this blueprint.

BLUEPRINT:
{bp_text}

TICKETS TO EVALUATE:
{tickets_block}

For EACH ticket (1 through {len(out_sample_idx)}), answer:
1. FITS: Does this ticket fall within the blueprint's stated intent and scope? (yes/no)
2. STEPS_WORK: Could the blueprint's steps produce a reasonable resolution? (yes/no)
3. EXIT_TRIGGERED: Does any exit condition apply (should escalate instead)? (yes/no)

Respond with ONLY a valid JSON array — one object per ticket:
[{{"ticket_num": 1, "fits": true/false, "steps_work": true/false, "exit_triggered": true/false, "exit_reason": "which exit condition or null", "brief_reason": "one sentence"}}]"""

    response, used_model = call_gemini(prompt)
    usage = response.usage_metadata
    total_input_tokens += usage.prompt_token_count or 0
    total_output_tokens += usage.candidates_token_count or 0
    total_thinking_tokens += getattr(usage, 'thoughts_token_count', 0) or 0
    print(f"    (used {used_model})")

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        out_results = json.loads(raw)
        if not isinstance(out_results, list):
            out_results = [out_results]
    except json.JSONDecodeError:
        print(f"    PARSE ERROR on out-of-cluster batch: {raw[:200]}")
        out_results = [{"fits": None, "steps_work": None, "exit_triggered": None, "parse_error": raw[:100]}
                       for _ in out_sample_idx]

    for i, result in enumerate(out_results):
        if i < len(ticket_meta_out):
            result["ticket_type"] = ticket_meta_out[i]["type"]
            result["ticket_tag1"] = ticket_meta_out[i]["tag1"]
            result["source_cluster"] = ticket_meta_out[i]["source_cluster"]
        cluster_results["out_of_cluster"].append(result)
        status = "GOOD" if not result.get("fits") or result.get("exit_triggered") else "LEAK"
        src = ticket_meta_out[i]["source_cluster"] if i < len(ticket_meta_out) else "?"
        print(f"    [{i+1:2d}] {status} | fits={result.get('fits')} exit={result.get('exit_triggered')} "
              f"| from cluster {src} | {ticket_meta_out[i]['type']}/{ticket_meta_out[i]['tag1']}")

    # Summarize
    in_fits = sum(1 for r in cluster_results["in_cluster"] if r.get("fits"))
    in_works = sum(1 for r in cluster_results["in_cluster"] if r.get("steps_work"))
    n_in = len(cluster_results["in_cluster"])
    out_rejected = sum(1 for r in cluster_results["out_of_cluster"]
                       if not r.get("fits") or r.get("exit_triggered"))
    n_out = len(cluster_results["out_of_cluster"])

    cluster_results["summary"] = {
        "in_cluster_fits": f"{in_fits}/{n_in}",
        "in_cluster_steps_work": f"{in_works}/{n_in}",
        "out_of_cluster_rejected": f"{out_rejected}/{n_out}",
    }
    review_results[cid_str] = cluster_results

    print(f"\n  Summary: in-cluster fits={in_fits}/{n_in}, steps_work={in_works}/{n_in} | "
          f"out-of-cluster rejected={out_rejected}/{n_out}")

    # Save intermediate
    with open(OUTPUT_DIR / "blueprint_review_results.json", "w") as f:
        json.dump({"reviews": review_results, "review_cost": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "thinking_tokens": total_thinking_tokens,
        }, "status": "in_progress"}, f, indent=2, default=str)
    print(f"  (intermediate results saved)")

    time.sleep(5)  # pause between blueprints


# ── Final summary ────────────────────────────────────────────────────────────

review_cost = {
    "input_tokens": total_input_tokens,
    "output_tokens": total_output_tokens,
    "thinking_tokens": total_thinking_tokens,
    "cost_usd": round(
        total_input_tokens / 1e6 * 0.15 +
        total_output_tokens / 1e6 * 0.60 +
        total_thinking_tokens / 1e6 * 0.35, 4
    ),
}

with open(OUTPUT_DIR / "blueprint_review_results.json", "w") as f:
    json.dump({"reviews": review_results, "review_cost": review_cost, "status": "complete"}, f, indent=2, default=str)

print(f"\n{'=' * 60}")
print("BLUEPRINT REVIEW SUMMARY")
print(f"{'=' * 60}")

for cid_str, results in review_results.items():
    s = results["summary"]
    print(f"  Cluster {cid_str}: in-cluster {s['in_cluster_fits']} fit, "
          f"{s['in_cluster_steps_work']} steps work | "
          f"out-of-cluster {s['out_of_cluster_rejected']} rejected")

print(f"\n  Review cost: {total_input_tokens + total_output_tokens + total_thinking_tokens:,} tokens "
      f"(${review_cost['cost_usd']:.4f})")
print(f"  Saved to output/blueprint_review_results.json")
