"""
Phase 3 — Real Experiment: Cloud vs Local+Blueprint
=====================================================
The actual test of the blueprint hypothesis:

  Arm A: Gemini Flash (cloud) — full agent, no blueprint
  Arm B: Gemma 4B (local) + blueprint — small model guided by blueprint

Question: Can a small local model + blueprint match a cloud model's quality?

Judge: Gemini Flash (blind, randomized order)
Split: native English vs translated German
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
import requests

load_dotenv(override=True)

OUTPUT_DIR = Path("output")
np.random.seed(42)

# ── API clients ──────────────────────────────────────────────────────────────

gemini_client = genai.Client(api_key=os.getenv("gemini_api_key"))
LOCAL_URL = "http://127.0.0.1:8080/v1/chat/completions"

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash"]

def call_gemini(prompt, max_retries=12):
    """Call Gemini Flash with model rotation."""
    for attempt in range(max_retries):
        for model in GEMINI_MODELS:
            try:
                response = gemini_client.models.generate_content(
                    model=model, contents=prompt,
                )
                return response, model
            except Exception as e:
                if any(k in str(e) for k in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                    continue
                raise
        wait = min(60 * (attempt + 1), 300)
        print(f"      Gemini busy, waiting {wait}s (attempt {attempt+1})")
        time.sleep(wait)
    raise RuntimeError("Gemini failed after all retries")


def call_local(prompt, max_tokens=4000):
    """Call local Gemma 4B via OpenAI-compatible API.
    Gemma 4B is a thinking model — it uses reasoning tokens before output.
    Needs high max_tokens to allow for both reasoning + response."""
    response = requests.post(LOCAL_URL, json={
        "model": "gemma-4-E4B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }, timeout=180)
    response.raise_for_status()
    data = response.json()
    msg = data["choices"][0]["message"]
    text = msg.get("content", "")
    # If content is empty but reasoning exists, the model ran out of tokens
    if not text.strip() and msg.get("reasoning_content"):
        text = msg["reasoning_content"]
    usage = data.get("usage", {})
    return text, usage


def parse_json(raw):
    """Parse JSON from LLM response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()
    return json.loads(raw)


# ── Load data ────────────────────────────────────────────────────────────────

print("=" * 60)
print("PHASE 3: REAL EXPERIMENT")
print("Cloud (Gemini Flash) vs Local (Gemma 4B + Blueprint)")
print("=" * 60)

experiment_df = pd.read_parquet(OUTPUT_DIR / "experiment_sample.parquet")
with open(OUTPUT_DIR / "blueprints.json") as f:
    blueprints = json.load(f)

n = len(experiment_df)
print(f"\n  Experiment tickets: {n}")
print(f"  English: {(~experiment_df['translated']).sum()}")
print(f"  German (translated): {experiment_df['translated'].sum()}")
print(f"  Route distribution: {dict(Counter(experiment_df['routed_cluster']))}")

# ── Progress tracking ────────────────────────────────────────────────────────

PROGRESS_FILE = OUTPUT_DIR / "real_experiment_progress.json"

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"arm_a": {}, "arm_b": {}, "judge": {},
            "arm_a_tokens": {"input": 0, "output": 0, "thinking": 0},
            "arm_b_tokens": {"prompt": 0, "completion": 0},
            "judge_tokens": {"input": 0, "output": 0, "thinking": 0}}

def save_progress(prog):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(prog, f, indent=2, default=str)

progress = load_progress()
print(f"  Existing progress: {len(progress['arm_a'])} arm_a, {len(progress['arm_b'])} arm_b, {len(progress['judge'])} judge")


# ══════════════════════════════════════════════════════════════════════════════
# ARM A: Gemini Flash (full agent, no blueprint)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print("ARM A: GEMINI FLASH (full agent)")
print(f"{'═' * 60}")

for i, (_, row) in enumerate(experiment_df.iterrows()):
    if str(i) in progress["arm_a"]:
        continue

    ticket_text = f"Subject: {row.get('subject', 'N/A')}\nBody: {str(row.get('body', ''))[:600]}"

    prompt = f"""You are a senior customer support agent. Handle this ticket professionally.

TICKET:
{ticket_text}

Provide:
1. Classification: What type of issue is this?
2. Severity: low / medium / high / critical
3. Action: resolve directly, escalate, or request more info
4. Draft response: A professional response to the customer (2-4 sentences).

Respond with ONLY valid JSON:
{{"classification": "...", "severity": "...", "action": "...", "draft_response": "..."}}"""

    response, model = call_gemini(prompt)
    usage = response.usage_metadata
    progress["arm_a_tokens"]["input"] += usage.prompt_token_count or 0
    progress["arm_a_tokens"]["output"] += usage.candidates_token_count or 0
    progress["arm_a_tokens"]["thinking"] += getattr(usage, 'thoughts_token_count', 0) or 0

    try:
        result = parse_json(response.text)
    except (json.JSONDecodeError, Exception):
        result = {"classification": "unknown", "severity": "medium",
                  "action": "escalate", "draft_response": response.text.strip()[:300]}

    progress["arm_a"][str(i)] = result
    save_progress(progress)
    print(f"  [{i+1:2d}/{n}] {model} | {result.get('classification', '?')[:25]} | {result.get('action', '?')}")
    time.sleep(3)

print(f"\n  Arm A done: {len(progress['arm_a'])} responses")


# ══════════════════════════════════════════════════════════════════════════════
# ARM B: Gemma 4B Local + Blueprint
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print("ARM B: GEMMA 4B LOCAL + BLUEPRINT")
print(f"{'═' * 60}")

for i, (_, row) in enumerate(experiment_df.iterrows()):
    if str(i) in progress["arm_b"]:
        continue

    cid = int(row["routed_cluster"])
    bp = blueprints[str(cid)]
    bp_text = json.dumps({
        "intent": bp["intent"], "scope": bp["scope"], "steps": bp["steps"],
        "minimal_context": bp["minimal_context"], "exit_conditions": bp["exit_conditions"],
    }, indent=2)

    ticket_text = f"Subject: {row.get('subject', 'N/A')}\nBody: {str(row.get('body', ''))[:600]}"

    prompt = f"""You are a customer support agent. Follow the BLUEPRINT below step-by-step to handle this ticket.

BLUEPRINT:
{bp_text}

TICKET:
{ticket_text}

Work through EVERY step in the blueprint. If any exit condition triggers, recommend escalation.

Respond with ONLY valid JSON:
{{"classification": "...", "severity": "...", "action": "resolve/escalate/request_info", "exit_triggered": false, "draft_response": "..."}}"""

    try:
        text, usage = call_local(prompt, max_tokens=4000)
        progress["arm_b_tokens"]["prompt"] += usage.get("prompt_tokens", 0)
        progress["arm_b_tokens"]["completion"] += usage.get("completion_tokens", 0)
    except Exception as e:
        print(f"  [{i+1:2d}/{n}] LOCAL ERROR: {str(e)[:100]}")
        text = '{"classification": "error", "severity": "medium", "action": "escalate", "draft_response": "Unable to process"}'

    try:
        result = parse_json(text)
    except (json.JSONDecodeError, Exception):
        result = {"classification": "unknown", "severity": "medium",
                  "action": "escalate", "exit_triggered": False, "draft_response": text.strip()[:300]}

    result["routed_cluster"] = cid
    progress["arm_b"][str(i)] = result
    save_progress(progress)
    print(f"  [{i+1:2d}/{n}] cluster {cid} | {result.get('classification', '?')[:25]} | {result.get('action', '?')}")

print(f"\n  Arm B done: {len(progress['arm_b'])} responses")


# ══════════════════════════════════════════════════════════════════════════════
# BLIND JUDGE: Gemini Flash
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print("BLIND JUDGE: GEMINI FLASH")
print(f"{'═' * 60}")

np.random.seed(123)  # fixed seed for judge ordering

for i, (_, row) in enumerate(experiment_df.iterrows()):
    if str(i) in progress["judge"]:
        continue

    a_result = progress["arm_a"].get(str(i), {})
    b_result = progress["arm_b"].get(str(i), {})

    a_resp = a_result.get("draft_response", "No response")
    b_resp = b_result.get("draft_response", "No response")
    a_act = a_result.get("action", "unknown")
    b_act = b_result.get("action", "unknown")

    # Randomly swap to prevent position bias
    if np.random.random() > 0.5:
        rx, ry = a_resp, b_resp
        ax, ay = a_act, b_act
        ordering = "A_first"
    else:
        rx, ry = b_resp, a_resp
        ax, ay = b_act, a_act
        ordering = "B_first"

    ticket_text = f"Subject: {row.get('subject', 'N/A')}\nBody: {str(row.get('body', ''))[:300]}"

    prompt = f"""You are a quality auditor for customer support. Compare these two responses to the same ticket. You don't know which system produced which.

TICKET:
{ticket_text}

RESPONSE X:
  Action: {ax}
  Response: {rx}

RESPONSE Y:
  Action: {ay}
  Response: {ry}

Evaluate:
1. Which better addresses the customer's needs? (X, Y, or tie)
2. Rate each quality (1-5: 1=terrible, 3=acceptable, 5=excellent)
3. Is each action appropriate? (yes/no)

Respond with ONLY valid JSON:
{{"winner": "X"/"Y"/"tie", "x_quality": 3, "y_quality": 4, "x_action_ok": true, "y_action_ok": true, "reason": "brief explanation"}}"""

    response, model = call_gemini(prompt)
    usage = response.usage_metadata
    progress["judge_tokens"]["input"] += usage.prompt_token_count or 0
    progress["judge_tokens"]["output"] += usage.candidates_token_count or 0
    progress["judge_tokens"]["thinking"] += getattr(usage, 'thoughts_token_count', 0) or 0

    try:
        judgment = parse_json(response.text)
    except (json.JSONDecodeError, Exception):
        judgment = {"winner": "tie", "x_quality": 3, "y_quality": 3,
                    "x_action_ok": True, "y_action_ok": True, "reason": "parse_error"}

    # De-anonymize
    if ordering == "A_first":
        judgment["arm_a_quality"] = judgment.get("x_quality", 3)
        judgment["arm_b_quality"] = judgment.get("y_quality", 3)
        w = judgment.get("winner", "tie")
        judgment["actual_winner"] = "arm_a" if w == "X" else ("arm_b" if w == "Y" else "tie")
    else:
        judgment["arm_a_quality"] = judgment.get("y_quality", 3)
        judgment["arm_b_quality"] = judgment.get("x_quality", 3)
        w = judgment.get("winner", "tie")
        judgment["actual_winner"] = "arm_b" if w == "X" else ("arm_a" if w == "Y" else "tie")

    judgment["ordering"] = ordering
    progress["judge"][str(i)] = judgment
    save_progress(progress)

    winner = judgment["actual_winner"]
    print(f"  [{i+1:2d}/{n}] winner={winner} | A={judgment['arm_a_quality']}/5 B={judgment['arm_b_quality']}/5 | {judgment.get('reason', '')[:50]}")
    time.sleep(3)


# ══════════════════════════════════════════════════════════════════════════════
# FINAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print("EXPERIMENT RESULTS: Gemini Flash vs Gemma 4B + Blueprint")
print(f"{'═' * 60}")

judgments = [progress["judge"][str(i)] for i in range(n) if str(i) in progress["judge"]]
n_judged = len(judgments)

winners = Counter(j["actual_winner"] for j in judgments)
a_quals = [j["arm_a_quality"] for j in judgments]
b_quals = [j["arm_b_quality"] for j in judgments]

# Token costs
a_tok = progress["arm_a_tokens"]
b_tok = progress["arm_b_tokens"]
j_tok = progress["judge_tokens"]

a_total = a_tok["input"] + a_tok["output"] + a_tok["thinking"]
b_total = b_tok["prompt"] + b_tok["completion"]

# Gemini cost (paid rate)
a_cost = (a_tok["input"] / 1e6 * 0.15 + a_tok["output"] / 1e6 * 0.60 + a_tok["thinking"] / 1e6 * 0.35)
# Local cost = $0 (runs on your machine)
b_cost = 0.0

with open(OUTPUT_DIR / "blueprint_generation_cost.json") as f:
    bp_gen_cost = json.load(f)["cost_usd"]["total"]

# Per-ticket
a_per = a_total / max(n_judged, 1)
b_per = b_total / max(n_judged, 1)

# Break-even: blueprint gen cost / per-ticket cloud cost saved
if a_cost > 0:
    cloud_cost_per_ticket = a_cost / n_judged
    breakeven = int(bp_gen_cost / cloud_cost_per_ticket) if cloud_cost_per_ticket > 0 else 0
else:
    breakeven = 0

# Language split
lang_analysis = {}
for lang_name, mask_vals in [("native_english", ~experiment_df["translated"].values),
                               ("translated_german", experiment_df["translated"].values)]:
    lang_idx = [i for i in range(n) if mask_vals[i] and str(i) in progress["judge"]]
    if not lang_idx:
        continue
    lang_j = [progress["judge"][str(i)] for i in lang_idx]
    lang_w = Counter(j["actual_winner"] for j in lang_j)
    lang_analysis[lang_name] = {
        "count": len(lang_j),
        "arm_a_wins": lang_w.get("arm_a", 0),
        "arm_b_wins": lang_w.get("arm_b", 0),
        "ties": lang_w.get("tie", 0),
        "arm_a_avg_quality": round(np.mean([j["arm_a_quality"] for j in lang_j]), 2),
        "arm_b_avg_quality": round(np.mean([j["arm_b_quality"] for j in lang_j]), 2),
    }

# Print
print(f"\n  QUALITY (blind judge, n={n_judged}):")
print(f"    Gemini Flash wins:        {winners.get('arm_a', 0)} ({winners.get('arm_a', 0)/max(n_judged,1)*100:.0f}%)")
print(f"    Gemma 4B+Blueprint wins:  {winners.get('arm_b', 0)} ({winners.get('arm_b', 0)/max(n_judged,1)*100:.0f}%)")
print(f"    Ties:                     {winners.get('tie', 0)} ({winners.get('tie', 0)/max(n_judged,1)*100:.0f}%)")
print(f"    Gemini Flash avg quality: {np.mean(a_quals):.2f}/5")
print(f"    Gemma 4B+BP avg quality:  {np.mean(b_quals):.2f}/5")

print(f"\n  TOKENS:")
print(f"    Gemini Flash:       {a_total:,} total ({a_per:,.0f}/ticket)")
print(f"    Gemma 4B+Blueprint: {b_total:,} total ({b_per:,.0f}/ticket)")

print(f"\n  COST:")
print(f"    Gemini Flash: ${a_cost:.4f} for {n_judged} tickets (${a_cost/max(n_judged,1)*1000:.2f}/1k tickets)")
print(f"    Gemma 4B:     $0.0000 for {n_judged} tickets (local, free)")
print(f"    Blueprint gen: ${bp_gen_cost:.4f} (one-time)")
print(f"    Break-even: {breakeven} tickets (then local is free)")

print(f"\n  BY LANGUAGE:")
for lang, stats in lang_analysis.items():
    print(f"    {lang} (n={stats['count']}):")
    print(f"      Gemini wins={stats['arm_a_wins']}, Gemma+BP wins={stats['arm_b_wins']}, ties={stats['ties']}")
    print(f"      Quality: Gemini={stats['arm_a_avg_quality']:.2f}, Gemma+BP={stats['arm_b_avg_quality']:.2f}")

# Save report
report = {
    "experiment": {
        "arm_a": "Gemini Flash (cloud, full agent)",
        "arm_b": "Gemma 4B Q8_0 (local) + blueprint",
        "judge": "Gemini Flash (blind)",
        "total_tickets": n_judged,
        "english": int((~experiment_df["translated"]).sum()),
        "german_translated": int(experiment_df["translated"].sum()),
    },
    "quality": {
        "arm_a_wins": winners.get("arm_a", 0),
        "arm_b_wins": winners.get("arm_b", 0),
        "ties": winners.get("tie", 0),
        "arm_a_avg_quality": round(float(np.mean(a_quals)), 2),
        "arm_b_avg_quality": round(float(np.mean(b_quals)), 2),
    },
    "tokens": {
        "arm_a_gemini": a_tok,
        "arm_b_local": b_tok,
        "judge": j_tok,
        "arm_a_per_ticket": round(a_per),
        "arm_b_per_ticket": round(b_per),
    },
    "cost": {
        "arm_a_cloud_usd": round(a_cost, 6),
        "arm_b_local_usd": 0.0,
        "blueprint_gen_usd": bp_gen_cost,
        "breakeven_tickets": breakeven,
    },
    "by_language": lang_analysis,
}

with open(OUTPUT_DIR / "real_experiment_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n  Saved to output/real_experiment_report.json")
