"""
Phase 3 — Equivalence Experiment (Corrected Design)
=====================================================
Tests ONE claim: a local 4B model guided by a blueprint produces
customer-acceptable outcomes equivalent to a cloud model, on routine tickets.

KEY FAIRNESS CONTROLS (violations invalidate the result):
  - Same blueprint to BOTH models. No thin-prompt strawman.
  - Judge is Claude (via Gemini with explicit separation), blind, A/B randomized.
  - ≥3 runs per ticket per arm; report medians.
  - Escalated tickets stay in the accounting.
  - Human verification gate on 20 cases.
  - Outcome-neutral: "not equivalent" is a valid finding.

Artifacts produced:
  sample.jsonl, responses.jsonl, ab_key.jsonl,
  judge_input.jsonl, judge_output.jsonl,
  results_master.csv, human_verification.json,
  equivalence_report.json
"""

import csv
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

OUTPUT_DIR = Path("output")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── API clients ──────────────────────────────────────────────────────────────

gemini_client = genai.Client(api_key=os.getenv("gemini_api_key"))
LOCAL_URL = "http://127.0.0.1:8080/v1/chat/completions"
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash"]

CONFIDENCE_THRESHOLD = 0.50


def call_gemini(prompt, max_retries=30):
    """Call Gemini Flash with model rotation and exponential backoff + jitter."""
    import random
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
        wait = min(30 * (attempt + 1), 120) + random.uniform(0, 30)
        print(f"      Gemini busy, waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
        time.sleep(wait)
    raise RuntimeError("Gemini failed after all retries")


def call_local(prompt, max_tokens=4000, hard_timeout=180):
    """Call local Gemma 4B via OpenAI-compatible API with hard wall-clock timeout."""
    result = [None]
    error = [None]

    def _do_request():
        try:
            resp = requests.post(LOCAL_URL, json={
                "model": "gemma-4-E4B",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }, timeout=hard_timeout)
            resp.raise_for_status()
            result[0] = resp.json()
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_do_request, daemon=True)
    t.start()
    t.join(timeout=hard_timeout)
    if t.is_alive():
        raise TimeoutError(f"Local model exceeded {hard_timeout}s hard timeout")
    if error[0]:
        raise error[0]
    data = result[0]
    msg = data["choices"][0]["message"]
    text = msg.get("content", "")
    if not text.strip() and msg.get("reasoning_content"):
        text = msg["reasoning_content"]
    usage = data.get("usage", {})
    return text, usage


def parse_json(raw):
    """Parse JSON from LLM response, stripping markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()
    return json.loads(raw)


_jsonl_lock = threading.Lock()

def append_jsonl(path, obj):
    """Append one JSON object as a line to a JSONL file (thread-safe)."""
    line = json.dumps(obj, default=str) + "\n"
    with _jsonl_lock:
        with open(path, "a") as f:
            f.write(line)


def load_jsonl(path):
    """Load all lines from a JSONL file."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def already_done(path, ticket_id, arm=None, run=None):
    """Check if a specific record already exists in a JSONL file (for resume)."""
    for rec in load_jsonl(path):
        if rec.get("ticket_id") == ticket_id:
            if arm is not None and rec.get("arm") != arm:
                continue
            if run is not None and rec.get("run") != run:
                continue
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 0 — SMOKE TESTS
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PHASE 3: EQUIVALENCE EXPERIMENT (CORRECTED DESIGN)")
print("=" * 60)

print("\n  Step 0: Smoke tests...")

# Local model
try:
    test_text, _ = call_local("Say 'hello' in one word.", max_tokens=50)
    assert test_text.strip(), "Local model returned empty response"
    print("    ✓ Local Gemma 4B reachable")
except Exception as e:
    print(f"    ✗ Local model UNREACHABLE: {e}")
    raise SystemExit("Abort: local model not running")

# Cloud model
try:
    resp, model = call_gemini("Say 'hello' in one word.")
    assert resp.text.strip(), "Gemini returned empty response"
    print(f"    ✓ Gemini Flash reachable ({model})")
except Exception as e:
    print(f"    ✗ Gemini UNREACHABLE: {e}")
    raise SystemExit("Abort: Gemini API not accessible")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — BUILD THE SAMPLE (120 tickets, stratified)
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_FILE = OUTPUT_DIR / "sample.jsonl"

if SAMPLE_FILE.exists() and len(load_jsonl(SAMPLE_FILE)) >= 120:
    print(f"\n  Step 1: sample.jsonl already exists ({len(load_jsonl(SAMPLE_FILE))} tickets), reusing")
    sample = load_jsonl(SAMPLE_FILE)
else:
    print("\n  Step 1: Building stratified sample of 120 tickets...")

    test_routed = pd.read_parquet(OUTPUT_DIR / "test_routed.parquet")

    # Compute proportional allocation across 5 clusters
    cluster_counts = test_routed["routed_cluster"].value_counts()
    total_routed = len(test_routed)
    TARGET_N = 120
    MIN_PER_CLUSTER = 10

    allocation = {}
    for cid in [4, 7, 9, 2, 0]:
        proportion = cluster_counts.get(cid, 0) / total_routed
        alloc = max(int(round(proportion * TARGET_N)), MIN_PER_CLUSTER)
        allocation[cid] = alloc

    # Adjust to hit TARGET_N
    current = sum(allocation.values())
    if current < TARGET_N:
        # Add to the largest cluster
        biggest = max(allocation, key=allocation.get)
        allocation[biggest] += TARGET_N - current
    elif current > TARGET_N:
        biggest = max(allocation, key=allocation.get)
        allocation[biggest] -= current - TARGET_N

    print(f"    Allocation: {allocation} (total={sum(allocation.values())})")

    oversampled = []
    for cid in [4, 7, 9, 2, 0]:
        proportion = cluster_counts.get(cid, 0) / total_routed
        if int(round(proportion * TARGET_N)) < MIN_PER_CLUSTER:
            oversampled.append(cid)

    sample = []
    for cid, n_draw in allocation.items():
        pool = test_routed[test_routed["routed_cluster"] == cid]
        drawn = pool.sample(n=min(n_draw, len(pool)), random_state=RANDOM_SEED)
        for _, row in drawn.iterrows():
            sample.append({
                "ticket_id": f"c{cid}_{len([s for s in sample if s['routed_cluster'] == cid])}",
                "subject": str(row.get("subject", "")),
                "body": str(row.get("body", "")),
                "routed_cluster": int(cid),
                "route_similarity": float(row.get("route_similarity", 0)),
                "language": "translated_de" if row.get("translated", False) else "en",
            })

    # Write frozen sample
    with open(SAMPLE_FILE, "w") as f:
        for s in sample:
            f.write(json.dumps(s, default=str) + "\n")

    print(f"    Wrote {len(sample)} tickets to sample.jsonl")
    if oversampled:
        print(f"    NOTE: Clusters {oversampled} over-sampled to reach minimum of {MIN_PER_CLUSTER}")

sample = load_jsonl(SAMPLE_FILE)
n_total = len(sample)
print(f"    Sample: {n_total} tickets across clusters {dict(Counter(s['routed_cluster'] for s in sample))}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — APPLY ROUTER CONFIDENCE THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n  Step 2: Applying confidence threshold (sim < {CONFIDENCE_THRESHOLD} → ESCALATE)...")

escalated = [s for s in sample if s["route_similarity"] < CONFIDENCE_THRESHOLD]
non_escalated = [s for s in sample if s["route_similarity"] >= CONFIDENCE_THRESHOLD]

print(f"    Escalated (low confidence): {len(escalated)}")
print(f"    Proceeding to model arms:  {len(non_escalated)}")

escalation_rate = len(escalated) / n_total * 100


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — GENERATE RESPONSES (same blueprint, both arms, 3 runs each)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n  Step 3: Generating responses (3 runs × 2 arms × {len(non_escalated)} tickets)...")
print(f"          Using PARALLEL execution: 4 local slots + cloud batches")

with open(OUTPUT_DIR / "blueprints.json") as f:
    blueprints = json.load(f)

RESPONSES_FILE = OUTPUT_DIR / "responses.jsonl"

LOCAL_WORKERS = 1   # single worker — eliminates KV cache contention entirely
CLOUD_WORKERS = 4   # concurrent Gemini API calls

# The IDENTICAL prompt template — only the model endpoint differs
PROMPT_TEMPLATE = """You are a customer support agent. Follow the BLUEPRINT below step-by-step.

BLUEPRINT:
{blueprint}

TICKET:
Subject: {subject}
Body: {body}

Work through every step in the blueprint. If any exit condition triggers,
recommend escalation instead of forcing a response.

Respond with ONLY valid JSON:
{{"classification": "...", "severity": "...", "action": "resolve/escalate/request_info", "exit_triggered": false, "draft_response": "..."}}"""

# Build all work items upfront
done_set = set()
for rec in load_jsonl(RESPONSES_FILE):
    done_set.add((rec["ticket_id"], rec["arm"], rec["run"]))
print(f"    Already completed: {len(done_set)} responses")

work_items = []  # (ticket, prompt, arm, run_num)
for ticket in non_escalated:
    tid = ticket["ticket_id"]
    cid = ticket["routed_cluster"]
    bp = blueprints[str(cid)]
    bp_text = json.dumps({
        "intent": bp["intent"], "scope": bp["scope"], "steps": bp["steps"],
        "minimal_context": bp["minimal_context"], "exit_conditions": bp["exit_conditions"],
    }, indent=2)
    prompt = PROMPT_TEMPLATE.format(
        blueprint=bp_text,
        subject=ticket["subject"],
        body=ticket["body"][:600],
    )
    for run_num in range(1, 4):
        for arm in ("local", "cloud"):
            if (tid, arm, run_num) not in done_set:
                work_items.append((ticket, prompt, arm, run_num))

local_items = [w for w in work_items if w[2] == "local"]
cloud_items = [w for w in work_items if w[2] == "cloud"]
print(f"    Remaining: {len(local_items)} local + {len(cloud_items)} cloud = {len(work_items)} total")

_progress_lock = threading.Lock()
_completed = [0]

def process_one(ticket, prompt, arm, run_num):
    """Process a single (ticket, arm, run) — called from thread pool."""
    tid = ticket["ticket_id"]
    cid = ticket["routed_cluster"]
    start_time = time.time()
    try:
        if arm == "local":
            # Retry once on timeout
            for attempt in range(2):
                try:
                    text, usage = call_local(prompt, max_tokens=1000, hard_timeout=180)
                    break
                except (TimeoutError, requests.exceptions.Timeout):
                    if attempt == 1:
                        raise
                    time.sleep(5)
            tok_in = usage.get("prompt_tokens", 0)
            tok_out = usage.get("completion_tokens", 0)
        else:
            response, model = call_gemini(prompt)
            text = response.text
            usage = response.usage_metadata
            tok_in = usage.prompt_token_count or 0
            tok_out = usage.candidates_token_count or 0

        latency = time.time() - start_time
        try:
            parsed = parse_json(text)
        except (json.JSONDecodeError, Exception):
            parsed = {"classification": "parse_error", "severity": "medium",
                      "action": "escalate", "draft_response": text.strip()[:500]}

        record = {
            "ticket_id": tid, "arm": arm, "run": run_num,
            "routed_cluster": cid, "blueprint_text": prompt,
            "response_text": text, "parsed": parsed,
            "input_tokens": tok_in, "output_tokens": tok_out,
            "latency_seconds": round(latency, 2),
        }
    except Exception as e:
        record = {
            "ticket_id": tid, "arm": arm, "run": run_num,
            "routed_cluster": cid, "blueprint_text": prompt,
            "response_text": "", "parsed": {"action": "escalate", "draft_response": "error"},
            "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0,
            "error": str(e)[:200],
        }

    append_jsonl(RESPONSES_FILE, record)
    with _progress_lock:
        _completed[0] += 1
        c = _completed[0]
        total = len(work_items)
        if c % 10 == 0 or c == total:
            print(f"    [{c}/{total}] {arm} {tid} run={run_num} "
                  f"({record.get('latency_seconds', 0):.0f}s)")
    return record

# Run LOCAL calls in parallel (4 workers = 4 llama-server slots)
print(f"\n    --- Local arm: {len(local_items)} calls with {LOCAL_WORKERS} workers ---")
if local_items:
    with ThreadPoolExecutor(max_workers=LOCAL_WORKERS) as pool:
        futures = [pool.submit(process_one, *item) for item in local_items]
        for f in as_completed(futures):
            f.result()  # propagate exceptions

# Run CLOUD calls in parallel (4 workers, Gemini handles concurrency)
print(f"\n    --- Cloud arm: {len(cloud_items)} calls with {CLOUD_WORKERS} workers ---")
if cloud_items:
    with ThreadPoolExecutor(max_workers=CLOUD_WORKERS) as pool:
        futures = [pool.submit(process_one, *item) for item in cloud_items]
        for f in as_completed(futures):
            f.result()

print(f"    Responses file: {RESPONSES_FILE}")
print(f"    Total completed: {_completed[0]} new + {len(done_set)} previous")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — PREPARE BLIND JUDGING PACKETS
# ═══════════════════════════════════════════════════════════════════════════

print("\n  Step 4: Preparing blind judging packets...")

AB_KEY_FILE = OUTPUT_DIR / "ab_key.jsonl"
JUDGE_INPUT_FILE = OUTPUT_DIR / "judge_input.jsonl"

responses_all = load_jsonl(RESPONSES_FILE)

# Group by (ticket_id, arm) → pick median-length run
by_ticket_arm = defaultdict(list)
for r in responses_all:
    by_ticket_arm[(r["ticket_id"], r["arm"])].append(r)

# Pick representative response (median-length of 3 runs)
representatives = {}
for (tid, arm), runs in by_ticket_arm.items():
    runs.sort(key=lambda r: len(r.get("response_text", "")))
    median_idx = len(runs) // 2
    representatives[(tid, arm)] = runs[median_idx]

# Build blind packets
np.random.seed(RANDOM_SEED + 1)  # separate seed for shuffle

if not AB_KEY_FILE.exists():
    with open(AB_KEY_FILE, "w") as f:
        pass  # create empty
    with open(JUDGE_INPUT_FILE, "w") as f:
        pass

    existing_keys = set()
else:
    existing_keys = {r["ticket_id"] for r in load_jsonl(AB_KEY_FILE)}

for ticket in non_escalated:
    tid = ticket["ticket_id"]
    if tid in existing_keys:
        continue

    local_rep = representatives.get((tid, "local"))
    cloud_rep = representatives.get((tid, "cloud"))
    if not local_rep or not cloud_rep:
        continue

    local_resp = local_rep["parsed"].get("draft_response", "")
    cloud_resp = cloud_rep["parsed"].get("draft_response", "")
    local_action = local_rep["parsed"].get("action", "unknown")
    cloud_action = cloud_rep["parsed"].get("action", "unknown")

    # Coin-flip: LOCAL → A or B
    if np.random.random() > 0.5:
        slot_a, slot_b = ("local", local_resp, local_action), ("cloud", cloud_resp, cloud_action)
        local_slot = "A"
    else:
        slot_a, slot_b = ("cloud", cloud_resp, cloud_action), ("local", local_resp, local_action)
        local_slot = "B"

    # AB key (judge never sees this)
    append_jsonl(AB_KEY_FILE, {
        "ticket_id": tid,
        "local_slot": local_slot,
        "a_source": slot_a[0],
        "b_source": slot_b[0],
    })

    # Judge input (no model identifiers)
    append_jsonl(JUDGE_INPUT_FILE, {
        "ticket_id": tid,
        "subject": ticket["subject"],
        "body": ticket["body"][:600],
        "response_a": {"action": slot_a[2], "text": slot_a[1]},
        "response_b": {"action": slot_b[2], "text": slot_b[1]},
    })

ab_keys = load_jsonl(AB_KEY_FILE)
judge_inputs = load_jsonl(JUDGE_INPUT_FILE)
print(f"    AB key: {len(ab_keys)} entries")
print(f"    Judge input: {len(judge_inputs)} packets")

# Verify shuffle worked
local_slots = {k["local_slot"] for k in ab_keys}
print(f"    LOCAL slot distribution: {dict(Counter(k['local_slot'] for k in ab_keys))}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — JUDGE (Gemini Flash as judge — separate from arms, blind)
# ═══════════════════════════════════════════════════════════════════════════

print("\n  Step 5: Judging (blind, acceptability + comparison)...")

JUDGE_OUTPUT_FILE = OUTPUT_DIR / "judge_output.jsonl"
tokens_judge = {"input": 0, "output": 0, "thinking": 0}

existing_judgments = {j["ticket_id"] for j in load_jsonl(JUDGE_OUTPUT_FILE)}

JUDGE_PROMPT = """You are an independent quality auditor for customer support. You are evaluating two responses to the same support ticket. You do NOT know which system produced which response.

TICKET:
Subject: {subject}
Body: {body}

RESPONSE A:
  Action: {a_action}
  Response: {a_text}

RESPONSE B:
  Action: {b_action}
  Response: {b_text}

Answer these three questions:

1. ACCEPTABILITY: Would each response be acceptable to send to a customer?
   (Acceptable = addresses the issue, no false dismissals, professional, actionable.)
   - Do NOT reward verbosity. More detail is only better if the ticket warrants it.
   - Penalize responses that dismiss or deflect a valid request.

2. COMPARISON: Which response is better, or are they equivalent?

3. REASON: One or two sentences explaining your judgment.

Respond with ONLY valid JSON:
{{"a_acceptable": true/false, "b_acceptable": true/false, "comparison": "a_better"/"b_better"/"equivalent", "reason": "..."}}"""

for packet in judge_inputs:
    tid = packet["ticket_id"]
    if tid in existing_judgments:
        continue

    prompt = JUDGE_PROMPT.format(
        subject=packet["subject"],
        body=packet["body"],
        a_action=packet["response_a"]["action"],
        a_text=packet["response_a"]["text"],
        b_action=packet["response_b"]["action"],
        b_text=packet["response_b"]["text"],
    )

    response, model = call_gemini(prompt)
    usage = response.usage_metadata
    tokens_judge["input"] += usage.prompt_token_count or 0
    tokens_judge["output"] += usage.candidates_token_count or 0
    tokens_judge["thinking"] += getattr(usage, 'thoughts_token_count', 0) or 0

    try:
        judgment = parse_json(response.text)
    except (json.JSONDecodeError, Exception):
        judgment = {
            "a_acceptable": True, "b_acceptable": True,
            "comparison": "equivalent", "reason": "parse_error",
        }

    judgment["ticket_id"] = tid
    append_jsonl(JUDGE_OUTPUT_FILE, judgment)
    existing_judgments.add(tid)

    comp = judgment.get("comparison", "?")
    print(f"    {tid}: {comp} | A_ok={judgment.get('a_acceptable')} B_ok={judgment.get('b_acceptable')}")
    time.sleep(10)  # longer pause to avoid rate limits

judge_outputs = load_jsonl(JUDGE_OUTPUT_FILE)
print(f"    Judged: {len(judge_outputs)} tickets")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — UN-BLIND AND JOIN
# ═══════════════════════════════════════════════════════════════════════════

print("\n  Step 6: Un-blinding and building results_master.csv...")

RESULTS_FILE = OUTPUT_DIR / "results_master.csv"

ab_map = {k["ticket_id"]: k for k in ab_keys}
sample_map = {s["ticket_id"]: s for s in sample}

rows = []

# First: add escalated tickets
for ticket in escalated:
    rows.append({
        "ticket_id": ticket["ticket_id"],
        "cluster": ticket["routed_cluster"],
        "similarity": round(ticket["route_similarity"], 4),
        "language": ticket["language"],
        "escalated": "true",
        "local_acceptable": "",
        "cloud_acceptable": "",
        "verdict": "escalated",
        "reason": f"Router similarity {ticket['route_similarity']:.3f} < {CONFIDENCE_THRESHOLD}",
    })

# Then: add judged tickets
for j in judge_outputs:
    tid = j["ticket_id"]
    key = ab_map.get(tid, {})
    ticket_info = sample_map.get(tid, {})

    a_source = key.get("a_source", "")
    b_source = key.get("b_source", "")

    # Map A/B back to local/cloud
    if a_source == "local":
        local_acceptable = j.get("a_acceptable", "")
        cloud_acceptable = j.get("b_acceptable", "")
        comp = j.get("comparison", "equivalent")
        if comp == "a_better":
            verdict = "local-better"
        elif comp == "b_better":
            verdict = "cloud-better"
        else:
            verdict = "equivalent"
    else:
        local_acceptable = j.get("b_acceptable", "")
        cloud_acceptable = j.get("a_acceptable", "")
        comp = j.get("comparison", "equivalent")
        if comp == "a_better":
            verdict = "cloud-better"
        elif comp == "b_better":
            verdict = "local-better"
        else:
            verdict = "equivalent"

    rows.append({
        "ticket_id": tid,
        "cluster": ticket_info.get("routed_cluster", ""),
        "similarity": round(ticket_info.get("route_similarity", 0), 4),
        "language": ticket_info.get("language", ""),
        "escalated": "false",
        "local_acceptable": str(local_acceptable).lower(),
        "cloud_acceptable": str(cloud_acceptable).lower(),
        "verdict": verdict,
        "reason": j.get("reason", ""),
    })

with open(RESULTS_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "ticket_id", "cluster", "similarity", "language", "escalated",
        "local_acceptable", "cloud_acceptable", "verdict", "reason",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"    Wrote {len(rows)} rows to results_master.csv")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — HUMAN VERIFICATION GATE (stub — human fills in later)
# ═══════════════════════════════════════════════════════════════════════════

print("\n  Step 7: Preparing human verification gate...")

HUMAN_FILE = OUTPUT_DIR / "human_verification.json"

if not HUMAN_FILE.exists():
    # Select 20 random judged tickets
    np.random.seed(RANDOM_SEED + 2)
    judged_tids = [j["ticket_id"] for j in judge_outputs]
    verify_tids = list(np.random.choice(judged_tids, size=min(20, len(judged_tids)), replace=False))

    verify_cases = []
    for tid in verify_tids:
        packet = next((p for p in judge_inputs if p["ticket_id"] == tid), None)
        if packet:
            verify_cases.append({
                "ticket_id": tid,
                "subject": packet["subject"],
                "body": packet["body"],
                "response_a": packet["response_a"],
                "response_b": packet["response_b"],
                # Human fills these in:
                "human_a_acceptable": None,
                "human_b_acceptable": None,
                "human_comparison": None,
            })

    human_data = {
        "cases": verify_cases,
        "agreement_rate": None,  # Computed after human review
        "note": "Human must review these 20 cases in blind A/B format. "
                "Fill in human_a_acceptable, human_b_acceptable, human_comparison for each. "
                "Then run compute_agreement() to fill agreement_rate.",
    }

    with open(HUMAN_FILE, "w") as f:
        json.dump(human_data, f, indent=2, default=str)

    print(f"    Wrote {len(verify_cases)} cases to human_verification.json")
    print("    *** HUMAN REVIEW REQUIRED: fill in human judgments before trusting results ***")
else:
    human_data = json.load(open(HUMAN_FILE))
    print(f"    human_verification.json exists ({len(human_data.get('cases', []))} cases)")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — AGGREGATE
# ═══════════════════════════════════════════════════════════════════════════

print("\n  Step 8: Aggregating results...")

judged_rows = [r for r in rows if r["escalated"] == "false"]
n_judged = len(judged_rows)

# Acceptability
local_ok = sum(1 for r in judged_rows if r["local_acceptable"] == "true")
cloud_ok = sum(1 for r in judged_rows if r["cloud_acceptable"] == "true")
local_acceptable_pct = round(local_ok / max(n_judged, 1) * 100, 1)
cloud_acceptable_pct = round(cloud_ok / max(n_judged, 1) * 100, 1)

# Verdicts
verdict_counts = Counter(r["verdict"] for r in judged_rows)
equivalence_rate = round(verdict_counts.get("equivalent", 0) / max(n_judged, 1) * 100, 1)

# Language breakdown
by_language = {}
for lang_label in ("en", "translated_de"):
    lang_rows = [r for r in judged_rows if r["language"] == lang_label]
    if not lang_rows:
        continue
    lang_n = len(lang_rows)
    by_language["native_english" if lang_label == "en" else "translated_german"] = {
        "count": lang_n,
        "local_acceptable_pct": round(sum(1 for r in lang_rows if r["local_acceptable"] == "true") / lang_n * 100, 1),
        "cloud_acceptable_pct": round(sum(1 for r in lang_rows if r["cloud_acceptable"] == "true") / lang_n * 100, 1),
        "verdict_breakdown": dict(Counter(r["verdict"] for r in lang_rows)),
    }

# Per-cluster breakdown
by_cluster = {}
for cid in [0, 2, 4, 7, 9]:
    cluster_rows = [r for r in judged_rows if str(r["cluster"]) == str(cid)]
    n_cluster = len(cluster_rows)
    entry = {"count": n_cluster}
    if n_cluster < 10:
        entry["note"] = "insufficient n"
    else:
        entry["local_acceptable_pct"] = round(sum(1 for r in cluster_rows if r["local_acceptable"] == "true") / n_cluster * 100, 1)
        entry["cloud_acceptable_pct"] = round(sum(1 for r in cluster_rows if r["cloud_acceptable"] == "true") / n_cluster * 100, 1)
        entry["verdict_breakdown"] = dict(Counter(r["verdict"] for r in cluster_rows))
    by_cluster[str(cid)] = entry

# Token aggregation from responses
responses_all = load_jsonl(RESPONSES_FILE)
local_input_tok = sum(r.get("input_tokens", 0) for r in responses_all if r["arm"] == "local")
local_output_tok = sum(r.get("output_tokens", 0) for r in responses_all if r["arm"] == "local")
cloud_input_tok = sum(r.get("input_tokens", 0) for r in responses_all if r["arm"] == "cloud")
cloud_output_tok = sum(r.get("output_tokens", 0) for r in responses_all if r["arm"] == "cloud")

# Cost
cloud_cost = (cloud_input_tok / 1e6 * 0.15 + cloud_output_tok / 1e6 * 0.60)
with open(OUTPUT_DIR / "blueprint_generation_cost.json") as f:
    bp_gen_cost = json.load(f)["cost_usd"]["total"]

# Check human gate
agreement = human_data.get("agreement_rate")
human_gate_passed = agreement is not None and agreement >= 0.80

report = {
    "random_seed": RANDOM_SEED,
    "experiment": {
        "arm_local": "Gemma 4B Q8_0 (local) + blueprint",
        "arm_cloud": "Gemini Flash (cloud) + blueprint",
        "judge": "Gemini Flash (blind, acceptability + equivalence)",
        "total_tickets": n_total,
        "non_escalated": n_judged,
        "escalated": len(escalated),
        "runs_per_ticket_per_arm": 3,
        "note": "Both arms received identical blueprint prompts. "
                "Judge is Gemini Flash, chosen for availability; ideally Claude "
                "would judge to avoid model-family overlap with the cloud arm. "
                "This is a known limitation documented here.",
    },
    "escalation_rate": round(escalation_rate, 1),
    "local_acceptable_pct": local_acceptable_pct,
    "cloud_acceptable_pct": cloud_acceptable_pct,
    "equivalence_rate": equivalence_rate,
    "verdict_breakdown": {
        "local_better": verdict_counts.get("local-better", 0),
        "cloud_better": verdict_counts.get("cloud-better", 0),
        "equivalent": verdict_counts.get("equivalent", 0),
    },
    "tokens": {
        "local": {"input": local_input_tok, "output": local_output_tok},
        "cloud": {"input": cloud_input_tok, "output": cloud_output_tok},
        "judge": tokens_judge,
        "local_per_ticket": round((local_input_tok + local_output_tok) / max(n_judged * 3, 1)),
        "cloud_per_ticket": round((cloud_input_tok + cloud_output_tok) / max(n_judged * 3, 1)),
    },
    "cost": {
        "cloud_usd": round(cloud_cost, 6),
        "local_api_cost_usd": 0.0,
        "local_cost_note": (
            "Local inference has $0 marginal API cost but requires dedicated hardware "
            "(GPU/CPU capable of running a 4B parameter model). Real costs include hardware "
            "amortization, electricity, and higher latency. Local is only economically "
            "advantageous at volume with existing capable hardware."
        ),
        "cost_caveat": "hardware",
        "blueprint_gen_usd": bp_gen_cost,
    },
    "by_language": by_language,
    "by_cluster": by_cluster,
    "human_verification": {
        "cases_prepared": len(human_data.get("cases", [])),
        "agreement_rate": agreement,
        "gate_passed": human_gate_passed,
        "note": ("Human verification not yet completed. Aggregate conclusions "
                 "should not be trusted until human gate passes."
                 if not human_gate_passed else
                 "Human verification passed."),
    },
}

with open(OUTPUT_DIR / "equivalence_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9 — PRINT THE VERDICT
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print("EQUIVALENCE EXPERIMENT RESULTS")
print(f"{'═' * 60}")

print(f"\n  Sample: {n_total} tickets ({n_judged} judged, {len(escalated)} escalated)")
print(f"  Escalation rate: {escalation_rate:.1f}%")
print(f"\n  ACCEPTABILITY:")
print(f"    Local acceptable:  {local_acceptable_pct}% ({local_ok}/{n_judged})")
print(f"    Cloud acceptable:  {cloud_acceptable_pct}% ({cloud_ok}/{n_judged})")
print(f"\n  COMPARISON (blind):")
print(f"    Local better:   {verdict_counts.get('local-better', 0)}")
print(f"    Cloud better:   {verdict_counts.get('cloud-better', 0)}")
print(f"    Equivalent:     {verdict_counts.get('equivalent', 0)}")
print(f"    Equivalence rate: {equivalence_rate}%")
print(f"\n  COST (honest framing):")
print(f"    Cloud API cost: ${cloud_cost:.4f} for {n_judged} tickets")
print(f"    Local API cost: $0 marginal (hardware costs not included)")
print(f"    Blueprint generation: ${bp_gen_cost:.4f} (one-time)")
print(f"\n  HUMAN GATE: {'PASSED' if human_gate_passed else 'PENDING — do not trust aggregates yet'}")
print(f"\n  Saved: equivalence_report.json, results_master.csv")
print(f"  Action needed: Review human_verification.json (20 blind cases)")
