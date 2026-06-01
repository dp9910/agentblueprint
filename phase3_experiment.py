"""
Phase 3 — Two-Arm Experiment
==============================
For each test ticket in the experiment sample, run both arms:
  Arm A: Full agent — Gemini sees ONLY the ticket, drafts a response from scratch
  Arm B: Blueprint agent — ticket is routed to a blueprint, Gemini follows the blueprint steps

Then a BLIND Gemini judge evaluates both responses without knowing which arm produced them.

Token costs are tracked per-arm for break-even analysis.
Results are split by native-EN vs translated-DE.
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
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

client = genai.Client(api_key=os.getenv("gemini_api_key"))

# ── Model rotation for rate limits ───────────────────────────────────────────

MODELS = ["gemini-3.5-flash", "gemini-2.5-flash"]
model_call_counts = {m: 0 for m in MODELS}

def call_gemini(prompt, max_retries=12, preferred_model=None):
    """Call Gemini with model rotation and rate-limit retry."""
    models_to_try = [preferred_model] + [m for m in MODELS if m != preferred_model] if preferred_model else MODELS[:]

    for attempt in range(max_retries):
        for model in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt,
                )
                model_call_counts[model] = model_call_counts.get(model, 0) + 1
                return response, model
            except Exception as e:
                err_str = str(e)
                if any(k in err_str for k in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                    continue  # try next model
                else:
                    raise
        # All models failed this round
        wait = 30 * (attempt + 1)
        print(f"      All models busy, waiting {wait}s... (attempt {attempt+1}/{max_retries})")
        time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries on all models")

# ── Load data ────────────────────────────────────────────────────────────────

print("=" * 60)
print("PHASE 3: TWO-ARM EXPERIMENT")
print("=" * 60)

print("\n[1/5] Loading experiment data...")
experiment_df = pd.read_parquet(OUTPUT_DIR / "experiment_sample.parquet")

with open(OUTPUT_DIR / "blueprints.json") as f:
    blueprints = json.load(f)

print(f"  Experiment tickets: {len(experiment_df)}")
print(f"  English: {(~experiment_df['translated']).sum()}")
print(f"  German (translated): {experiment_df['translated'].sum()}")
print(f"  Route distribution: {dict(Counter(experiment_df['routed_cluster']))}")

# ── Arm A: Full Agent ────────────────────────────────────────────────────────

print("\n[2/5] Running Arm A (Full Agent)...")
print("  Agent sees ONLY the ticket — no blueprint guidance")

arm_a_results = []
arm_a_tokens = {"input": 0, "output": 0, "thinking": 0}
BATCH_SIZE = 3  # 3 tickets per API call

for batch_start in range(0, len(experiment_df), BATCH_SIZE):
    batch = experiment_df.iloc[batch_start:batch_start + BATCH_SIZE]
    batch_num = batch_start // BATCH_SIZE + 1
    total_batches = (len(experiment_df) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} tickets)...")

    tickets_text = ""
    for i, (_, row) in enumerate(batch.iterrows(), 1):
        tickets_text += f"\n=== TICKET {i} ===\n"
        tickets_text += f"Subject: {row.get('subject', 'N/A')}\n"
        tickets_text += f"Body: {str(row.get('body', ''))[:600]}\n"

    prompt = f"""You are a senior customer support agent. For each ticket below, provide a professional response.

For EACH ticket, provide:
1. Classification: What type of issue is this? (e.g., technical bug, billing question, feature request, security concern, etc.)
2. Severity: low / medium / high / critical
3. Recommended action: What should be done? (resolve directly, escalate, request more info, etc.)
4. Draft response: Write a brief professional response to the customer (2-4 sentences).

{tickets_text}

Respond with ONLY a valid JSON array — one object per ticket:
[{{"ticket_num": 1, "classification": "...", "severity": "...", "action": "...", "draft_response": "..."}}]"""

    response, used_model = call_gemini(prompt)
    usage = response.usage_metadata
    arm_a_tokens["input"] += usage.prompt_token_count or 0
    arm_a_tokens["output"] += usage.candidates_token_count or 0
    arm_a_tokens["thinking"] += getattr(usage, 'thoughts_token_count', 0) or 0

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        batch_results = json.loads(raw)
        if not isinstance(batch_results, list):
            batch_results = [batch_results]
    except json.JSONDecodeError:
        print(f"    PARSE ERROR: {raw[:200]}")
        batch_results = [{"classification": "parse_error", "severity": "unknown",
                          "action": "unknown", "draft_response": raw[:200]}
                         for _ in range(len(batch))]

    for i, result in enumerate(batch_results):
        arm_a_results.append(result)
    print(f"    Done ({used_model}) — {len(batch_results)} results")
    time.sleep(5)

print(f"\n  Arm A: {len(arm_a_results)} responses, "
      f"{arm_a_tokens['input'] + arm_a_tokens['output'] + arm_a_tokens['thinking']:,} tokens")

# Save intermediate
with open(OUTPUT_DIR / "arm_a_results.json", "w") as f:
    json.dump({"results": arm_a_results, "tokens": arm_a_tokens}, f, indent=2, default=str)
print("  (saved intermediate arm_a_results.json)")


# ── Arm B: Blueprint Agent ───────────────────────────────────────────────────

print("\n[3/5] Running Arm B (Blueprint Agent)...")
print("  Agent sees ticket + matched blueprint — follows blueprint steps")

arm_b_results = []
arm_b_tokens = {"input": 0, "output": 0, "thinking": 0}

# Group by routed cluster for efficient batching
for cid in sorted(experiment_df["routed_cluster"].unique()):
    cluster_mask = experiment_df["routed_cluster"] == cid
    cluster_tickets = experiment_df[cluster_mask]
    bp = blueprints[str(cid)]

    bp_text = json.dumps({
        "intent": bp["intent"],
        "scope": bp["scope"],
        "steps": bp["steps"],
        "minimal_context": bp["minimal_context"],
        "exit_conditions": bp["exit_conditions"],
    }, indent=2)

    cluster_indices = cluster_tickets.index.tolist()

    for batch_start in range(0, len(cluster_tickets), BATCH_SIZE):
        batch = cluster_tickets.iloc[batch_start:batch_start + BATCH_SIZE]
        batch_idx = cluster_indices[batch_start:batch_start + BATCH_SIZE]

        tickets_text = ""
        for i, (_, row) in enumerate(batch.iterrows(), 1):
            tickets_text += f"\n=== TICKET {i} ===\n"
            tickets_text += f"Subject: {row.get('subject', 'N/A')}\n"
            tickets_text += f"Body: {str(row.get('body', ''))[:600]}\n"

        prompt = f"""You are a customer support agent. Follow the BLUEPRINT below step-by-step to handle each ticket.

BLUEPRINT:
{bp_text}

IMPORTANT: For each ticket, work through EVERY step in the blueprint. If any exit condition is triggered, note it and recommend escalation instead.

TICKETS:
{tickets_text}

For EACH ticket, respond with the output of following the blueprint:
[{{"ticket_num": 1, "step_outputs": [{{"step": 1, "output": "..."}}], "exit_triggered": false/true, "exit_reason": "null or which condition", "classification": "...", "severity": "...", "action": "resolve/escalate/request_info", "draft_response": "..."}}]

Respond with ONLY valid JSON."""

        response, used_model = call_gemini(prompt)
        usage = response.usage_metadata
        arm_b_tokens["input"] += usage.prompt_token_count or 0
        arm_b_tokens["output"] += usage.candidates_token_count or 0
        arm_b_tokens["thinking"] += getattr(usage, 'thoughts_token_count', 0) or 0

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        try:
            batch_results = json.loads(raw)
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
        except json.JSONDecodeError:
            print(f"    PARSE ERROR (cluster {cid}): {raw[:200]}")
            batch_results = [{"classification": "parse_error", "exit_triggered": False,
                              "action": "unknown", "draft_response": raw[:200]}
                             for _ in range(len(batch))]

        for i, result in enumerate(batch_results):
            result["routed_cluster"] = int(cid)
            if i < len(batch_idx):
                result["experiment_index"] = int(batch_idx[i])
            arm_b_results.append(result)

        print(f"    Cluster {cid}: batch of {len(batch)} done ({used_model})")
        time.sleep(5)

# Sort arm_b_results to match experiment_df order
arm_b_by_idx = {r.get("experiment_index", i): r for i, r in enumerate(arm_b_results)}
arm_b_results_sorted = [arm_b_by_idx.get(i, arm_b_results[i] if i < len(arm_b_results) else {})
                         for i in range(len(experiment_df))]

print(f"\n  Arm B: {len(arm_b_results)} responses, "
      f"{arm_b_tokens['input'] + arm_b_tokens['output'] + arm_b_tokens['thinking']:,} tokens")

with open(OUTPUT_DIR / "arm_b_results.json", "w") as f:
    json.dump({"results": arm_b_results_sorted, "tokens": arm_b_tokens}, f, indent=2, default=str)
print("  (saved intermediate arm_b_results.json)")


# ── Blind Judge ──────────────────────────────────────────────────────────────

print("\n[4/5] Running Blind Judge...")
print("  Judge sees BOTH responses (randomly ordered) + ticket — picks the better one")

judge_results = []
judge_tokens = {"input": 0, "output": 0, "thinking": 0}

for batch_start in range(0, len(experiment_df), BATCH_SIZE):
    batch = experiment_df.iloc[batch_start:batch_start + BATCH_SIZE]
    batch_num = batch_start // BATCH_SIZE + 1
    total_batches = (len(experiment_df) + BATCH_SIZE - 1) // BATCH_SIZE

    comparisons = ""
    orderings = []  # track which response is A/B for each ticket

    for i, (idx, row) in enumerate(batch.iterrows(), 1):
        a_result = arm_a_results[batch_start + i - 1] if batch_start + i - 1 < len(arm_a_results) else {}
        b_result = arm_b_results_sorted[batch_start + i - 1] if batch_start + i - 1 < len(arm_b_results_sorted) else {}

        a_response = a_result.get("draft_response", "No response generated")
        b_response = b_result.get("draft_response", "No response generated")
        a_action = a_result.get("action", "unknown")
        b_action = b_result.get("action", "unknown")

        # Randomly swap order to prevent position bias
        if np.random.random() > 0.5:
            resp_x, resp_y = a_response, b_response
            act_x, act_y = a_action, b_action
            orderings.append("A_first")
        else:
            resp_x, resp_y = b_response, a_response
            act_x, act_y = b_action, a_action
            orderings.append("B_first")

        comparisons += f"\n=== COMPARISON {i} ===\n"
        comparisons += f"TICKET:\n  Subject: {row.get('subject', 'N/A')}\n  Body: {str(row.get('body', ''))[:300]}\n"
        comparisons += f"\nRESPONSE X:\n  Action: {act_x}\n  Response: {resp_x}\n"
        comparisons += f"\nRESPONSE Y:\n  Action: {act_y}\n  Response: {resp_y}\n"

    prompt = f"""You are a quality auditor evaluating customer support responses. For each comparison below, a customer ticket was handled by two different systems (X and Y). You don't know which system is which.

{comparisons}

For EACH comparison, evaluate:
1. Which response better addresses the customer's needs? (X, Y, or tie)
2. Rate each response's quality (1-5 scale: 1=terrible, 3=acceptable, 5=excellent)
3. Is the recommended action appropriate? (yes/no for each)

Respond with ONLY a valid JSON array:
[{{"comparison_num": 1, "winner": "X"/"Y"/"tie", "x_quality": 3, "y_quality": 4, "x_action_appropriate": true, "y_action_appropriate": true, "reason": "brief explanation"}}]"""

    response, used_model = call_gemini(prompt)
    usage = response.usage_metadata
    judge_tokens["input"] += usage.prompt_token_count or 0
    judge_tokens["output"] += usage.candidates_token_count or 0
    judge_tokens["thinking"] += getattr(usage, 'thoughts_token_count', 0) or 0

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        batch_judgments = json.loads(raw)
        if not isinstance(batch_judgments, list):
            batch_judgments = [batch_judgments]
    except json.JSONDecodeError:
        print(f"    PARSE ERROR (judge batch {batch_num}): {raw[:200]}")
        batch_judgments = [{"winner": "tie", "x_quality": 3, "y_quality": 3,
                            "x_action_appropriate": True, "y_action_appropriate": True,
                            "reason": "parse_error"}
                           for _ in range(len(batch))]

    # De-anonymize: map X/Y back to A/B
    for i, judgment in enumerate(batch_judgments):
        ordering = orderings[i] if i < len(orderings) else "A_first"
        if ordering == "A_first":
            judgment["arm_a_quality"] = judgment.get("x_quality", 3)
            judgment["arm_b_quality"] = judgment.get("y_quality", 3)
            judgment["arm_a_action_ok"] = judgment.get("x_action_appropriate", True)
            judgment["arm_b_action_ok"] = judgment.get("y_action_appropriate", True)
            w = judgment.get("winner", "tie")
            judgment["actual_winner"] = "arm_a" if w == "X" else ("arm_b" if w == "Y" else "tie")
        else:
            judgment["arm_a_quality"] = judgment.get("y_quality", 3)
            judgment["arm_b_quality"] = judgment.get("x_quality", 3)
            judgment["arm_a_action_ok"] = judgment.get("y_action_appropriate", True)
            judgment["arm_b_action_ok"] = judgment.get("x_action_appropriate", True)
            w = judgment.get("winner", "tie")
            judgment["actual_winner"] = "arm_b" if w == "X" else ("arm_a" if w == "Y" else "tie")

        judge_results.append(judgment)

    print(f"  Batch {batch_num}/{total_batches} done ({used_model})")
    time.sleep(5)

with open(OUTPUT_DIR / "judge_results.json", "w") as f:
    json.dump({"judgments": judge_results, "tokens": judge_tokens}, f, indent=2, default=str)


# ── Final Analysis ───────────────────────────────────────────────────────────

print("\n[5/5] Analyzing results...")

# Winner distribution
winners = Counter(j.get("actual_winner", "tie") for j in judge_results)
n = len(judge_results)

# Quality scores
a_qualities = [j.get("arm_a_quality", 3) for j in judge_results]
b_qualities = [j.get("arm_b_quality", 3) for j in judge_results]

# Token cost comparison
# Arm B: amortize blueprint generation cost
with open(OUTPUT_DIR / "blueprint_generation_cost.json") as f:
    gen_cost = json.load(f)
bp_gen_total_cost = gen_cost["cost_usd"]["total"]

arm_a_cost = (arm_a_tokens["input"] / 1e6 * 0.15 +
              arm_a_tokens["output"] / 1e6 * 0.60 +
              arm_a_tokens["thinking"] / 1e6 * 0.35)
arm_b_cost = (arm_b_tokens["input"] / 1e6 * 0.15 +
              arm_b_tokens["output"] / 1e6 * 0.60 +
              arm_b_tokens["thinking"] / 1e6 * 0.35)

arm_a_total_tokens = arm_a_tokens["input"] + arm_a_tokens["output"] + arm_a_tokens["thinking"]
arm_b_total_tokens = arm_b_tokens["input"] + arm_b_tokens["output"] + arm_b_tokens["thinking"]

# Per-ticket costs
a_per_ticket = arm_a_total_tokens / max(n, 1)
b_per_ticket = arm_b_total_tokens / max(n, 1)
token_saving_pct = (1 - b_per_ticket / max(a_per_ticket, 1)) * 100

# Break-even: how many tickets until blueprint gen cost is offset by per-ticket savings
token_saving_per_ticket = a_per_ticket - b_per_ticket
if token_saving_per_ticket > 0:
    cost_saving_per_ticket = token_saving_per_ticket / 1e6 * 0.30  # blended rate
    breakeven_tickets = int(bp_gen_total_cost / max(cost_saving_per_ticket, 1e-10))
else:
    breakeven_tickets = float("inf")

# ── Split by language ────────────────────────────────────────────────────────

lang_analysis = {}
for lang_name, lang_mask_vals in [("native_english", ~experiment_df["translated"].values),
                                    ("translated_german", experiment_df["translated"].values)]:
    lang_idx = np.where(lang_mask_vals)[0]
    if len(lang_idx) == 0:
        continue
    lang_judgments = [judge_results[i] for i in lang_idx if i < len(judge_results)]
    lang_winners = Counter(j.get("actual_winner", "tie") for j in lang_judgments)
    lang_a_q = [j.get("arm_a_quality", 3) for j in lang_judgments]
    lang_b_q = [j.get("arm_b_quality", 3) for j in lang_judgments]

    lang_analysis[lang_name] = {
        "count": len(lang_judgments),
        "arm_a_wins": lang_winners.get("arm_a", 0),
        "arm_b_wins": lang_winners.get("arm_b", 0),
        "ties": lang_winners.get("tie", 0),
        "arm_a_avg_quality": round(np.mean(lang_a_q), 2) if lang_a_q else 0,
        "arm_b_avg_quality": round(np.mean(lang_b_q), 2) if lang_b_q else 0,
    }


# ── Print & save final report ────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("PHASE 3: EXPERIMENT RESULTS")
print(f"{'=' * 60}")

print(f"\n  QUALITY (blind judge, n={n}):")
print(f"    Arm A wins:  {winners.get('arm_a', 0)} ({winners.get('arm_a', 0)/max(n,1)*100:.0f}%)")
print(f"    Arm B wins:  {winners.get('arm_b', 0)} ({winners.get('arm_b', 0)/max(n,1)*100:.0f}%)")
print(f"    Ties:        {winners.get('tie', 0)} ({winners.get('tie', 0)/max(n,1)*100:.0f}%)")
print(f"    Arm A avg quality: {np.mean(a_qualities):.2f}/5")
print(f"    Arm B avg quality: {np.mean(b_qualities):.2f}/5")

print(f"\n  TOKENS:")
print(f"    Arm A (full agent):     {arm_a_total_tokens:,} total ({a_per_ticket:,.0f}/ticket)")
print(f"    Arm B (blueprint agent): {arm_b_total_tokens:,} total ({b_per_ticket:,.0f}/ticket)")
print(f"    Token saving: {token_saving_pct:.1f}% per ticket")

print(f"\n  COST:")
print(f"    Arm A cost: ${arm_a_cost:.4f} ({n} tickets)")
print(f"    Arm B cost: ${arm_b_cost:.4f} ({n} tickets) + ${bp_gen_total_cost:.4f} (blueprint gen)")
print(f"    Break-even: {breakeven_tickets:,} tickets")

print(f"\n  BY LANGUAGE:")
for lang, stats in lang_analysis.items():
    print(f"    {lang} (n={stats['count']}):")
    print(f"      A wins={stats['arm_a_wins']}, B wins={stats['arm_b_wins']}, ties={stats['ties']}")
    print(f"      Quality: A={stats['arm_a_avg_quality']:.2f}, B={stats['arm_b_avg_quality']:.2f}")

print(f"\n  MODEL USAGE: {dict(model_call_counts)}")

# Save complete report
report = {
    "experiment": {
        "total_tickets": n,
        "english_count": int((~experiment_df["translated"]).sum()),
        "german_translated_count": int(experiment_df["translated"].sum()),
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
        "arm_a_per_ticket": round(a_per_ticket),
        "arm_b_per_ticket": round(b_per_ticket),
        "token_saving_pct": round(token_saving_pct, 1),
    },
    "cost": {
        "arm_a_usd": round(arm_a_cost, 6),
        "arm_b_usd": round(arm_b_cost, 6),
        "blueprint_generation_usd": bp_gen_total_cost,
        "judge_usd": round(
            judge_tokens["input"] / 1e6 * 0.15 +
            judge_tokens["output"] / 1e6 * 0.60 +
            judge_tokens["thinking"] / 1e6 * 0.35, 6
        ),
        "breakeven_tickets": breakeven_tickets if breakeven_tickets != float("inf") else "never",
    },
    "by_language": lang_analysis,
    "model_usage": model_call_counts,
}

with open(OUTPUT_DIR / "experiment_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n  Saved to output/experiment_report.json")
print(f"  Saved arm results to output/arm_a_results.json, output/arm_b_results.json")
print(f"  Saved judge results to output/judge_results.json")
