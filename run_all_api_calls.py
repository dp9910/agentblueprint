"""
Master API Script — Review Gate + Two-Arm Experiment
=====================================================
Handles all Gemini API calls with:
  - Adaptive rate limiting (respects retry-after)
  - Model rotation (gemini-2.5-flash / gemini-3.5-flash)
  - Progress saving after every successful call
  - Resume support (skips already-completed steps)

Run this script and let it work through rate limits. It will complete
all API calls needed for the review gate and Phase 3 experiment.
"""

import os
import json
import time
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from google import genai

load_dotenv()

OUTPUT_DIR = Path("output")
PROGRESS_FILE = OUTPUT_DIR / "api_progress.json"
RANDOM_SEED = 99
np.random.seed(RANDOM_SEED)

client = genai.Client(api_key=os.getenv("gemini_api_key"))

MODELS = ["gemini-2.5-flash", "gemini-3.5-flash"]

# ── Progress tracking ────────────────────────────────────────────────────────

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_steps": [], "results": {}, "total_tokens": {"input": 0, "output": 0, "thinking": 0}}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2, default=str)

# ── Adaptive API caller ──────────────────────────────────────────────────────

call_count = [0]
last_success_time = [0]

def call_gemini_adaptive(prompt, step_name, max_retries=200):
    """Call Gemini with patient backoff and model rotation.
    Will retry for up to ~3 hours total, checking every 60-300s."""
    retry_after = 60
    for attempt in range(max_retries):
        for model in MODELS:
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt,
                )
                call_count[0] += 1
                last_success_time[0] = time.time()
                print(f"    [{step_name}] OK via {model} (call #{call_count[0]})")
                return response, model
            except Exception as e:
                err_str = str(e)
                if any(k in err_str for k in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                    m = re.search(r'retry in ([0-9.]+)s', err_str)
                    retry_after = float(m.group(1)) + 5 if m else 60
                    continue
                else:
                    raise

        # Both models failed — wait with patience
        wait = min(max(retry_after, 60), 300)  # between 60s and 5min
        if attempt % 10 == 0:
            print(f"    [{step_name}] Both models busy, waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})")
        time.sleep(wait)

    raise RuntimeError(f"[{step_name}] Failed after {max_retries} retries")

def parse_json_response(raw):
    """Parse JSON from Gemini response, stripping markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())

# ── Load data ────────────────────────────────────────────────────────────────

print("=" * 60)
print("MASTER API SCRIPT: REVIEW + EXPERIMENT")
print("=" * 60)

progress = load_progress()
print(f"  Completed steps: {len(progress['completed_steps'])}")

train_df = pd.read_parquet(OUTPUT_DIR / "combined_train_translated.parquet")
labels = train_df["cluster_label"].values

with open(OUTPUT_DIR / "blueprints.json") as f:
    blueprints = json.load(f)

experiment_df = pd.read_parquet(OUTPUT_DIR / "experiment_sample.parquet")

print(f"  Experiment tickets: {len(experiment_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: BLUEPRINT REVIEW GATE (batched — 10 API calls)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print("PART 1: BLUEPRINT REVIEW GATE")
print(f"{'═' * 60}")

# Reset seed for consistent sampling
np.random.seed(RANDOM_SEED)

review_results = progress["results"].get("review", {})

for cid_str, blueprint in blueprints.items():
    cid = int(cid_str)
    if "steps" not in blueprint:
        continue

    # ── In-cluster review ──
    step_name = f"review_{cid}_in"
    if step_name not in progress["completed_steps"]:
        mask = labels == cid
        cluster_indices = np.where(mask)[0]
        in_sample_idx = np.random.choice(cluster_indices, size=min(10, len(cluster_indices)), replace=False)

        bp_text = json.dumps({
            "intent": blueprint["intent"], "scope": blueprint["scope"],
            "steps": blueprint["steps"], "minimal_context": blueprint["minimal_context"],
            "exit_conditions": blueprint["exit_conditions"],
        }, indent=2)

        tickets_block = ""
        ticket_meta = []
        for i, idx in enumerate(in_sample_idx):
            row = train_df.iloc[idx]
            tickets_block += f"\n--- TICKET {i+1} ---\nSubject: {row.get('subject', 'N/A')}\nBody: {str(row.get('body', ''))[:400]}\n"
            ticket_meta.append({"type": str(row.get("type", "")), "tag1": str(row.get("tag_1", ""))})

        prompt = f"""You are reviewing whether a customer support blueprint is appropriate for specific tickets.

BLUEPRINT:
{bp_text}

TICKETS TO EVALUATE:
{tickets_block}

For EACH ticket, answer: FITS (yes/no), STEPS_WORK (yes/no), EXIT_TRIGGERED (yes/no).
Respond with ONLY a valid JSON array:
[{{"ticket_num": 1, "fits": true/false, "steps_work": true/false, "exit_triggered": true/false, "exit_reason": "null or reason", "brief_reason": "one sentence"}}]"""

        response, model = call_gemini_adaptive(prompt, step_name)
        usage = response.usage_metadata
        progress["total_tokens"]["input"] += usage.prompt_token_count or 0
        progress["total_tokens"]["output"] += usage.candidates_token_count or 0
        progress["total_tokens"]["thinking"] += getattr(usage, 'thoughts_token_count', 0) or 0

        try:
            results = parse_json_response(response.text)
            if not isinstance(results, list):
                results = [results]
        except json.JSONDecodeError:
            results = [{"fits": None, "steps_work": None, "exit_triggered": None, "parse_error": True}
                       for _ in in_sample_idx]

        for i, r in enumerate(results):
            if i < len(ticket_meta):
                r["ticket_type"] = ticket_meta[i]["type"]
                r["ticket_tag1"] = ticket_meta[i]["tag1"]

        if cid_str not in review_results:
            review_results[cid_str] = {}
        review_results[cid_str]["in_cluster"] = results

        progress["completed_steps"].append(step_name)
        progress["results"]["review"] = review_results
        save_progress(progress)
        print(f"  Cluster {cid} in-cluster: {sum(1 for r in results if r.get('fits'))}/{len(results)} fit")
    else:
        # Still need to advance the RNG for consistent sampling
        mask = labels == cid
        cluster_indices = np.where(mask)[0]
        np.random.choice(cluster_indices, size=min(10, len(cluster_indices)), replace=False)
        print(f"  Cluster {cid} in-cluster: already done (skipping)")

    # ── Out-of-cluster review ──
    step_name = f"review_{cid}_out"
    if step_name not in progress["completed_steps"]:
        other_mask = (labels != cid) & (labels != -1)
        other_indices = np.where(other_mask)[0]
        out_sample_idx = np.random.choice(other_indices, size=min(5, len(other_indices)), replace=False)

        bp_text = json.dumps({
            "intent": blueprint["intent"], "scope": blueprint["scope"],
            "steps": blueprint["steps"], "minimal_context": blueprint["minimal_context"],
            "exit_conditions": blueprint["exit_conditions"],
        }, indent=2)

        tickets_block = ""
        ticket_meta_out = []
        for i, idx in enumerate(out_sample_idx):
            row = train_df.iloc[idx]
            tickets_block += f"\n--- TICKET {i+1} ---\nSubject: {row.get('subject', 'N/A')}\nBody: {str(row.get('body', ''))[:400]}\n"
            ticket_meta_out.append({"type": str(row.get("type", "")), "tag1": str(row.get("tag_1", "")), "source_cluster": int(labels[idx])})

        prompt = f"""You are reviewing whether a customer support blueprint is appropriate for specific tickets.
These tickets are from DIFFERENT clusters — they should ideally NOT fit this blueprint.

BLUEPRINT:
{bp_text}

TICKETS:
{tickets_block}

For EACH ticket: FITS (yes/no), STEPS_WORK (yes/no), EXIT_TRIGGERED (yes/no).
Respond with ONLY a valid JSON array:
[{{"ticket_num": 1, "fits": true/false, "steps_work": true/false, "exit_triggered": true/false, "exit_reason": "null or reason", "brief_reason": "one sentence"}}]"""

        response, model = call_gemini_adaptive(prompt, step_name)
        usage = response.usage_metadata
        progress["total_tokens"]["input"] += usage.prompt_token_count or 0
        progress["total_tokens"]["output"] += usage.candidates_token_count or 0
        progress["total_tokens"]["thinking"] += getattr(usage, 'thoughts_token_count', 0) or 0

        try:
            results = parse_json_response(response.text)
            if not isinstance(results, list):
                results = [results]
        except json.JSONDecodeError:
            results = [{"fits": None, "steps_work": None, "exit_triggered": None, "parse_error": True}
                       for _ in out_sample_idx]

        for i, r in enumerate(results):
            if i < len(ticket_meta_out):
                r.update(ticket_meta_out[i])

        review_results[cid_str]["out_of_cluster"] = results

        # Summary
        in_res = review_results[cid_str].get("in_cluster", [])
        in_fits = sum(1 for r in in_res if r.get("fits"))
        in_works = sum(1 for r in in_res if r.get("steps_work"))
        out_rejected = sum(1 for r in results if not r.get("fits") or r.get("exit_triggered"))
        review_results[cid_str]["summary"] = {
            "in_cluster_fits": f"{in_fits}/{len(in_res)}",
            "in_cluster_steps_work": f"{in_works}/{len(in_res)}",
            "out_of_cluster_rejected": f"{out_rejected}/{len(results)}",
        }

        progress["completed_steps"].append(step_name)
        progress["results"]["review"] = review_results
        save_progress(progress)
        print(f"  Cluster {cid} out-of-cluster: {out_rejected}/{len(results)} rejected")
    else:
        other_mask = (labels != cid) & (labels != -1)
        other_indices = np.where(other_mask)[0]
        np.random.choice(other_indices, size=min(5, len(other_indices)), replace=False)
        print(f"  Cluster {cid} out-of-cluster: already done (skipping)")

# Save review results in standard format
with open(OUTPUT_DIR / "blueprint_review_results.json", "w") as f:
    json.dump({"reviews": review_results, "status": "complete"}, f, indent=2, default=str)
print("\n  Review gate complete — saved blueprint_review_results.json")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: ARM A — Full Agent (no blueprint)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print("PART 2: ARM A — FULL AGENT")
print(f"{'═' * 60}")

BATCH_SIZE = 5
arm_a_results = progress["results"].get("arm_a", [])
arm_a_tokens = progress["results"].get("arm_a_tokens", {"input": 0, "output": 0, "thinking": 0})

for batch_start in range(0, len(experiment_df), BATCH_SIZE):
    step_name = f"arm_a_batch_{batch_start}"
    if step_name in progress["completed_steps"]:
        print(f"  Batch {batch_start//BATCH_SIZE+1}: already done (skipping)")
        continue

    batch = experiment_df.iloc[batch_start:batch_start + BATCH_SIZE]

    tickets_text = ""
    for i, (_, row) in enumerate(batch.iterrows(), 1):
        tickets_text += f"\n=== TICKET {i} ===\nSubject: {row.get('subject', 'N/A')}\nBody: {str(row.get('body', ''))[:600]}\n"

    prompt = f"""You are a senior customer support agent. For each ticket below, provide a professional response.

For EACH ticket, provide:
1. Classification: What type of issue is this?
2. Severity: low / medium / high / critical
3. Recommended action: resolve directly, escalate, or request more info
4. Draft response: Write a brief professional response to the customer (2-4 sentences).

{tickets_text}

Respond with ONLY a valid JSON array:
[{{"ticket_num": 1, "classification": "...", "severity": "...", "action": "...", "draft_response": "..."}}]"""

    response, model = call_gemini_adaptive(prompt, step_name)
    usage = response.usage_metadata
    arm_a_tokens["input"] += usage.prompt_token_count or 0
    arm_a_tokens["output"] += usage.candidates_token_count or 0
    arm_a_tokens["thinking"] += getattr(usage, 'thoughts_token_count', 0) or 0

    try:
        batch_results = parse_json_response(response.text)
        if not isinstance(batch_results, list):
            batch_results = [batch_results]
    except json.JSONDecodeError:
        batch_results = [{"classification": "parse_error", "severity": "unknown",
                          "action": "unknown", "draft_response": "Error parsing response"}
                         for _ in range(len(batch))]

    arm_a_results.extend(batch_results)
    progress["completed_steps"].append(step_name)
    progress["results"]["arm_a"] = arm_a_results
    progress["results"]["arm_a_tokens"] = arm_a_tokens
    save_progress(progress)
    print(f"  Batch {batch_start//BATCH_SIZE+1}: {len(batch_results)} results ({model})")

print(f"\n  Arm A complete: {len(arm_a_results)} responses")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: ARM B — Blueprint Agent
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print("PART 3: ARM B — BLUEPRINT AGENT")
print(f"{'═' * 60}")

arm_b_results = progress["results"].get("arm_b", {})  # keyed by experiment index
arm_b_tokens = progress["results"].get("arm_b_tokens", {"input": 0, "output": 0, "thinking": 0})

for cid in sorted(experiment_df["routed_cluster"].unique()):
    cluster_mask = experiment_df["routed_cluster"] == cid
    cluster_tickets = experiment_df[cluster_mask]
    bp = blueprints[str(cid)]

    bp_text = json.dumps({
        "intent": bp["intent"], "scope": bp["scope"], "steps": bp["steps"],
        "minimal_context": bp["minimal_context"], "exit_conditions": bp["exit_conditions"],
    }, indent=2)

    cluster_indices = cluster_tickets.index.tolist()

    for batch_start in range(0, len(cluster_tickets), BATCH_SIZE):
        batch = cluster_tickets.iloc[batch_start:batch_start + BATCH_SIZE]
        batch_idx = cluster_indices[batch_start:batch_start + BATCH_SIZE]
        step_name = f"arm_b_c{cid}_b{batch_start}"

        if step_name in progress["completed_steps"]:
            print(f"  Cluster {cid} batch {batch_start//BATCH_SIZE+1}: already done (skipping)")
            continue

        tickets_text = ""
        for i, (_, row) in enumerate(batch.iterrows(), 1):
            tickets_text += f"\n=== TICKET {i} ===\nSubject: {row.get('subject', 'N/A')}\nBody: {str(row.get('body', ''))[:600]}\n"

        prompt = f"""You are a customer support agent. Follow the BLUEPRINT below step-by-step for each ticket.

BLUEPRINT:
{bp_text}

IMPORTANT: Work through EVERY blueprint step. If any exit condition triggers, recommend escalation.

TICKETS:
{tickets_text}

For EACH ticket, respond with:
[{{"ticket_num": 1, "exit_triggered": false, "exit_reason": null, "classification": "...", "severity": "...", "action": "resolve/escalate/request_info", "draft_response": "..."}}]

Respond with ONLY valid JSON."""

        response, model = call_gemini_adaptive(prompt, step_name)
        usage = response.usage_metadata
        arm_b_tokens["input"] += usage.prompt_token_count or 0
        arm_b_tokens["output"] += usage.candidates_token_count or 0
        arm_b_tokens["thinking"] += getattr(usage, 'thoughts_token_count', 0) or 0

        try:
            batch_results = parse_json_response(response.text)
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
        except json.JSONDecodeError:
            batch_results = [{"classification": "parse_error", "exit_triggered": False,
                              "action": "unknown", "draft_response": "Error parsing response"}
                             for _ in range(len(batch))]

        for i, r in enumerate(batch_results):
            r["routed_cluster"] = int(cid)
            if i < len(batch_idx):
                arm_b_results[str(batch_idx[i])] = r

        progress["completed_steps"].append(step_name)
        progress["results"]["arm_b"] = arm_b_results
        progress["results"]["arm_b_tokens"] = arm_b_tokens
        save_progress(progress)
        print(f"  Cluster {cid} batch: {len(batch_results)} results ({model})")

# Convert to ordered list
arm_b_results_list = [arm_b_results.get(str(i), {"draft_response": "No response", "action": "unknown"})
                       for i in range(len(experiment_df))]
print(f"\n  Arm B complete: {len(arm_b_results)} responses")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: BLIND JUDGE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print("PART 4: BLIND JUDGE")
print(f"{'═' * 60}")

# Use a fixed seed for judge ordering
np.random.seed(42)

judge_results = progress["results"].get("judge", [])
judge_tokens = progress["results"].get("judge_tokens", {"input": 0, "output": 0, "thinking": 0})
orderings_all = progress["results"].get("judge_orderings", [])

for batch_start in range(0, len(experiment_df), BATCH_SIZE):
    step_name = f"judge_batch_{batch_start}"
    if step_name in progress["completed_steps"]:
        print(f"  Judge batch {batch_start//BATCH_SIZE+1}: already done (skipping)")
        # Still advance RNG
        for _ in range(min(BATCH_SIZE, len(experiment_df) - batch_start)):
            np.random.random()
        continue

    batch = experiment_df.iloc[batch_start:batch_start + BATCH_SIZE]

    comparisons = ""
    orderings = []
    for i, (idx, row) in enumerate(batch.iterrows(), 1):
        a_idx = batch_start + i - 1
        a_result = arm_a_results[a_idx] if a_idx < len(arm_a_results) else {}
        b_result = arm_b_results_list[a_idx] if a_idx < len(arm_b_results_list) else {}

        a_resp = a_result.get("draft_response", "No response")
        b_resp = b_result.get("draft_response", "No response")
        a_act = a_result.get("action", "unknown")
        b_act = b_result.get("action", "unknown")

        if np.random.random() > 0.5:
            rx, ry, ax, ay = a_resp, b_resp, a_act, b_act
            orderings.append("A_first")
        else:
            rx, ry, ax, ay = b_resp, a_resp, b_act, a_act
            orderings.append("B_first")

        comparisons += f"\n=== COMPARISON {i} ===\n"
        comparisons += f"TICKET:\n  Subject: {row.get('subject', 'N/A')}\n  Body: {str(row.get('body', ''))[:300]}\n"
        comparisons += f"\nRESPONSE X:\n  Action: {ax}\n  Response: {rx}\n"
        comparisons += f"\nRESPONSE Y:\n  Action: {ay}\n  Response: {ry}\n"

    prompt = f"""You are a quality auditor for customer support. For each comparison, evaluate two responses (X and Y) to the same ticket.

{comparisons}

For EACH comparison:
1. Which response better addresses the customer's needs? (X, Y, or tie)
2. Rate each response's quality (1-5: 1=terrible, 3=acceptable, 5=excellent)
3. Is the recommended action appropriate? (yes/no for each)

Respond with ONLY a valid JSON array:
[{{"comparison_num": 1, "winner": "X"/"Y"/"tie", "x_quality": 3, "y_quality": 4, "x_action_appropriate": true, "y_action_appropriate": true, "reason": "brief explanation"}}]"""

    response, model = call_gemini_adaptive(prompt, step_name)
    usage = response.usage_metadata
    judge_tokens["input"] += usage.prompt_token_count or 0
    judge_tokens["output"] += usage.candidates_token_count or 0
    judge_tokens["thinking"] += getattr(usage, 'thoughts_token_count', 0) or 0

    try:
        batch_judgments = parse_json_response(response.text)
        if not isinstance(batch_judgments, list):
            batch_judgments = [batch_judgments]
    except json.JSONDecodeError:
        batch_judgments = [{"winner": "tie", "x_quality": 3, "y_quality": 3,
                            "x_action_appropriate": True, "y_action_appropriate": True,
                            "reason": "parse_error"} for _ in range(len(batch))]

    # De-anonymize
    for i, j in enumerate(batch_judgments):
        ordering = orderings[i] if i < len(orderings) else "A_first"
        if ordering == "A_first":
            j["arm_a_quality"] = j.get("x_quality", 3)
            j["arm_b_quality"] = j.get("y_quality", 3)
            j["arm_a_action_ok"] = j.get("x_action_appropriate", True)
            j["arm_b_action_ok"] = j.get("y_action_appropriate", True)
            w = j.get("winner", "tie")
            j["actual_winner"] = "arm_a" if w == "X" else ("arm_b" if w == "Y" else "tie")
        else:
            j["arm_a_quality"] = j.get("y_quality", 3)
            j["arm_b_quality"] = j.get("x_quality", 3)
            j["arm_a_action_ok"] = j.get("y_action_appropriate", True)
            j["arm_b_action_ok"] = j.get("x_action_appropriate", True)
            w = j.get("winner", "tie")
            j["actual_winner"] = "arm_b" if w == "X" else ("arm_a" if w == "Y" else "tie")

        judge_results.append(j)
    orderings_all.extend(orderings)

    progress["completed_steps"].append(step_name)
    progress["results"]["judge"] = judge_results
    progress["results"]["judge_tokens"] = judge_tokens
    progress["results"]["judge_orderings"] = orderings_all
    save_progress(progress)
    print(f"  Judge batch {batch_start//BATCH_SIZE+1}: done ({model})")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print("FINAL ANALYSIS")
print(f"{'═' * 60}")

n = len(judge_results)
winners = Counter(j.get("actual_winner", "tie") for j in judge_results)
a_qualities = [j.get("arm_a_quality", 3) for j in judge_results]
b_qualities = [j.get("arm_b_quality", 3) for j in judge_results]

# Token costs
with open(OUTPUT_DIR / "blueprint_generation_cost.json") as f:
    gen_cost = json.load(f)
bp_gen_cost = gen_cost["cost_usd"]["total"]

arm_a_total = arm_a_tokens["input"] + arm_a_tokens["output"] + arm_a_tokens["thinking"]
arm_b_total = arm_b_tokens["input"] + arm_b_tokens["output"] + arm_b_tokens["thinking"]

arm_a_cost = (arm_a_tokens["input"] / 1e6 * 0.15 +
              arm_a_tokens["output"] / 1e6 * 0.60 +
              arm_a_tokens["thinking"] / 1e6 * 0.35)
arm_b_cost = (arm_b_tokens["input"] / 1e6 * 0.15 +
              arm_b_tokens["output"] / 1e6 * 0.60 +
              arm_b_tokens["thinking"] / 1e6 * 0.35)

a_per = arm_a_total / max(n, 1)
b_per = arm_b_total / max(n, 1)
token_save = (1 - b_per / max(a_per, 1)) * 100

if a_per > b_per:
    save_per_ticket_cost = (a_per - b_per) / 1e6 * 0.30
    breakeven = int(bp_gen_cost / max(save_per_ticket_cost, 1e-10))
else:
    breakeven = "never (blueprint uses more tokens)"

# Language split
lang_analysis = {}
for lang, mask_fn in [("native_english", lambda: ~experiment_df["translated"].values),
                       ("translated_german", lambda: experiment_df["translated"].values)]:
    lang_idx = np.where(mask_fn())[0]
    if len(lang_idx) == 0:
        continue
    lang_j = [judge_results[i] for i in lang_idx if i < len(judge_results)]
    lang_w = Counter(j.get("actual_winner", "tie") for j in lang_j)
    lang_analysis[lang] = {
        "count": len(lang_j),
        "arm_a_wins": lang_w.get("arm_a", 0),
        "arm_b_wins": lang_w.get("arm_b", 0),
        "ties": lang_w.get("tie", 0),
        "arm_a_avg_quality": round(np.mean([j.get("arm_a_quality", 3) for j in lang_j]), 2),
        "arm_b_avg_quality": round(np.mean([j.get("arm_b_quality", 3) for j in lang_j]), 2),
    }

# ── Print report ──
print(f"\n  QUALITY (blind judge, n={n}):")
print(f"    Arm A wins (full agent):     {winners.get('arm_a', 0)} ({winners.get('arm_a', 0)/max(n,1)*100:.0f}%)")
print(f"    Arm B wins (blueprint agent): {winners.get('arm_b', 0)} ({winners.get('arm_b', 0)/max(n,1)*100:.0f}%)")
print(f"    Ties:                         {winners.get('tie', 0)} ({winners.get('tie', 0)/max(n,1)*100:.0f}%)")
print(f"    Arm A avg quality: {np.mean(a_qualities):.2f}/5")
print(f"    Arm B avg quality: {np.mean(b_qualities):.2f}/5")

print(f"\n  TOKENS PER TICKET:")
print(f"    Arm A (full agent):      {a_per:,.0f} tokens/ticket")
print(f"    Arm B (blueprint agent): {b_per:,.0f} tokens/ticket")
print(f"    Token saving: {token_save:+.1f}%")

print(f"\n  COST:")
print(f"    Arm A: ${arm_a_cost:.4f} for {n} tickets (${arm_a_cost/max(n,1)*1000:.2f}/1k tickets)")
print(f"    Arm B: ${arm_b_cost:.4f} for {n} tickets (${arm_b_cost/max(n,1)*1000:.2f}/1k tickets)")
print(f"    Blueprint generation: ${bp_gen_cost:.4f} (one-time)")
print(f"    Break-even: {breakeven} tickets")

print(f"\n  BY LANGUAGE:")
for lang, stats in lang_analysis.items():
    print(f"    {lang} (n={stats['count']}):")
    print(f"      A wins={stats['arm_a_wins']}, B wins={stats['arm_b_wins']}, ties={stats['ties']}")
    print(f"      Quality: A={stats['arm_a_avg_quality']:.2f}, B={stats['arm_b_avg_quality']:.2f}")

# Review summary
print(f"\n  REVIEW GATE:")
for cid_str, rr in review_results.items():
    if "summary" in rr:
        s = rr["summary"]
        print(f"    Cluster {cid_str}: in={s['in_cluster_fits']} fit, "
              f"{s['in_cluster_steps_work']} steps_work | out={s['out_of_cluster_rejected']} rejected")

# ── Save final report ──
report = {
    "experiment": {
        "total_tickets": n,
        "english": int((~experiment_df["translated"]).sum()),
        "german_translated": int(experiment_df["translated"].sum()),
    },
    "quality": {
        "arm_a_wins": winners.get("arm_a", 0),
        "arm_b_wins": winners.get("arm_b", 0),
        "ties": winners.get("tie", 0),
        "arm_a_avg_quality": round(float(np.mean(a_qualities)), 2),
        "arm_b_avg_quality": round(float(np.mean(b_qualities)), 2),
    },
    "tokens": {
        "arm_a": arm_a_tokens,
        "arm_b": arm_b_tokens,
        "judge": judge_tokens,
        "arm_a_per_ticket": round(a_per),
        "arm_b_per_ticket": round(b_per),
        "token_saving_pct": round(token_save, 1),
    },
    "cost": {
        "arm_a_usd": round(arm_a_cost, 6),
        "arm_b_usd": round(arm_b_cost, 6),
        "blueprint_gen_usd": bp_gen_cost,
        "judge_usd": round(
            judge_tokens["input"] / 1e6 * 0.15 +
            judge_tokens["output"] / 1e6 * 0.60 +
            judge_tokens["thinking"] / 1e6 * 0.35, 6
        ),
        "breakeven_tickets": breakeven,
    },
    "by_language": lang_analysis,
    "review_gate": {cid: rr.get("summary", {}) for cid, rr in review_results.items()},
}

with open(OUTPUT_DIR / "experiment_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

# Also save individual result files
with open(OUTPUT_DIR / "arm_a_results.json", "w") as f:
    json.dump({"results": arm_a_results, "tokens": arm_a_tokens}, f, indent=2, default=str)
with open(OUTPUT_DIR / "arm_b_results.json", "w") as f:
    json.dump({"results": arm_b_results_list, "tokens": arm_b_tokens}, f, indent=2, default=str)
with open(OUTPUT_DIR / "judge_results.json", "w") as f:
    json.dump({"judgments": judge_results, "tokens": judge_tokens}, f, indent=2, default=str)

print(f"\n{'═' * 60}")
print("ALL DONE")
print(f"{'═' * 60}")
print(f"  Total API calls: {call_count[0]}")
print(f"  Saved: experiment_report.json, arm_a_results.json, arm_b_results.json, judge_results.json")
