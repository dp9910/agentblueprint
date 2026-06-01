"""
Phase 2 — Blueprint Generation
================================
Generate structured blueprints for the 5 head clusters using Gemini 2.5 Flash.
Each blueprint follows a locked schema. Total token cost is recorded for break-even math.

Guardrails:
  - Blueprints derived from TRAINING data only
  - Locked schema applied to all 5
  - Start with cluster 4 (biggest, hardest) to stress-test the schema
  - Exit conditions informed by the off-type minority from purity data

Outputs:
  - output/blueprints.json — the 5 structured blueprints
  - output/blueprint_generation_cost.json — token costs
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

# ── Verify train/test split is frozen ─────────────────────────────────────────

print("=" * 60)
print("PHASE 2: BLUEPRINT GENERATION")
print("=" * 60)

print("\n[1/4] Verifying train/test split...")
train_path = OUTPUT_DIR / "combined_train_translated.parquet"
test_path = OUTPUT_DIR / "test_split.parquet"

assert train_path.exists(), "Train split not found!"
assert test_path.exists(), "Test split not found!"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)
print(f"  Train: {len(train_df):,} tickets (blueprints derived from this ONLY)")
print(f"  Test:  {len(test_df):,} tickets (held out, never seen by blueprints)")

labels = train_df["cluster_label"].values


# ── Define blueprint schema ───────────────────────────────────────────────────

BLUEPRINT_SCHEMA = {
    "intent": "What this blueprint serves — the customer need it addresses",
    "scope": "What types of tickets this blueprint handles (and what it does NOT)",
    "steps": [
        {
            "step_number": "int",
            "name": "Short name for this step",
            "input_needed": "What information the executor needs to perform this step",
            "action": "What the executor does",
            "decision_output": "What this step produces or decides",
        }
    ],
    "minimal_context": "The minimum information from the ticket needed to execute this blueprint",
    "exit_conditions": [
        "Specific conditions under which this blueprint should NOT be applied — escalate instead"
    ],
}

print("\n[2/4] Blueprint schema locked:")
print(json.dumps(BLUEPRINT_SCHEMA, indent=2))


# ── Prepare cluster data for blueprint generation ─────────────────────────────

print("\n[3/4] Sampling representative tickets per cluster...")

# Cluster ordering: start with cluster 4 (biggest/hardest), then rest by size
cluster_sizes = Counter(labels)
if -1 in cluster_sizes:
    del cluster_sizes[-1]
sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: -x[1])
head_clusters = sorted_clusters[:5]

# Reorder: cluster 4 first (rank 1), then the rest
cluster_order = [cid for cid, _ in head_clusters]
print(f"  Head clusters (generation order): {cluster_order}")

# For each cluster, gather:
# 1. 15 representative tickets (mix of types within the cluster)
# 2. Purity stats and off-type minorities (for exit conditions)
cluster_profiles = {}

for cid, size in head_clusters:
    mask = labels == cid
    cluster_df = train_df[mask].copy()

    # Type distribution
    type_dist = Counter(cluster_df["type"].fillna("Unknown"))
    tag1_dist = Counter(cluster_df["tag_1"].fillna("none"))
    queue_dist = Counter(cluster_df["queue"].fillna("Unknown"))

    # Dominant labels
    top_type = type_dist.most_common(1)[0]
    type_purity = top_type[1] / size

    # Off-type minorities (for exit conditions)
    off_types = [(t, c, c/size) for t, c in type_dist.most_common() if t != top_type[0] and c/size > 0.05]

    # Sample 15 representative tickets
    sample_idx = np.random.choice(len(cluster_df), size=min(15, len(cluster_df)), replace=False)
    samples = []
    for idx in sample_idx:
        row = cluster_df.iloc[idx]
        samples.append({
            "subject": str(row.get("subject", ""))[:100],
            "body": str(row.get("body", ""))[:400],
            "type": str(row.get("type", "")),
            "tag_1": str(row.get("tag_1", "")),
            "queue": str(row.get("queue", "")),
            "answer": str(row.get("answer", ""))[:400],
        })

    cluster_profiles[cid] = {
        "cluster_id": int(cid),
        "size": int(size),
        "type_distribution": {str(k): int(v) for k, v in type_dist.most_common()},
        "tag1_distribution": {str(k): int(v) for k, v in tag1_dist.most_common(5)},
        "queue_distribution": {str(k): int(v) for k, v in queue_dist.most_common(5)},
        "type_purity": round(type_purity, 3),
        "off_type_minorities": [(t, round(p, 3)) for t, _, p in off_types],
        "sample_tickets": samples,
    }
    print(f"  Cluster {cid} (n={size:,}): {len(samples)} samples, "
          f"purity={type_purity:.1%}, off-types={[t for t,_,_ in off_types]}")


# ── Generate blueprints with Gemini 2.5 Flash ────────────────────────────────

print("\n[4/4] Generating blueprints with Gemini 2.5 Flash...")

client = genai.Client(api_key=os.getenv("gemini_api_key"))

total_input_tokens = 0
total_output_tokens = 0
total_thinking_tokens = 0
blueprints = {}

for cid in cluster_order:
    profile = cluster_profiles[cid]

    # Build the prompt
    tickets_text = ""
    for i, t in enumerate(profile["sample_tickets"], 1):
        tickets_text += f"\n--- Ticket {i} ---\n"
        tickets_text += f"Subject: {t['subject']}\n"
        tickets_text += f"Body: {t['body']}\n"
        tickets_text += f"Type: {t['type']} | Tag: {t['tag_1']} | Queue: {t['queue']}\n"
        if t["answer"]:
            tickets_text += f"Resolution: {t['answer']}\n"

    off_type_text = ""
    if profile["off_type_minorities"]:
        off_type_text = "\n".join(
            f"  - {t} ({p:.0%} of cluster)" for t, p in profile["off_type_minorities"]
        )
    else:
        off_type_text = "  (none significant)"

    prompt = f"""You are designing a structured blueprint for a customer support automation system.

TASK: Analyze the following {len(profile['sample_tickets'])} representative tickets from a cluster of {profile['size']:,} similar tickets, and produce a single structured blueprint that could handle the majority of tickets in this cluster.

CLUSTER STATISTICS:
- Size: {profile['size']:,} tickets
- Type distribution: {json.dumps(profile['type_distribution'])}
- Top tags: {json.dumps(profile['tag1_distribution'])}
- Top queues: {json.dumps(profile['queue_distribution'])}
- Type purity: {profile['type_purity']:.1%}
- Off-type minorities (tickets that land here but may not fit):
{off_type_text}

REPRESENTATIVE TICKETS:
{tickets_text}

IMPORTANT DESIGN CONSTRAINTS:
1. This cluster has {profile['type_purity']:.0%} type purity. The blueprint must be broad enough to handle the dominant pattern AND have explicit exit conditions for the {100 - int(profile['type_purity']*100)}% that may not fit.
2. Exit conditions should specifically address the off-type minorities listed above — these are tickets that get routed here but should escalate.
3. Steps should be concrete and actionable, not vague. Each step should specify what information is needed, what action to take, and what decision/output it produces.
4. The blueprint will be executed by a small LLM with ONLY the ticket text and this blueprint as context — no access to databases, customer history, or external systems. Steps should be about analyzing the ticket, classifying severity, drafting a response, and deciding whether to escalate.

OUTPUT FORMAT — respond with ONLY valid JSON matching this exact schema:
{{
  "intent": "A clear statement of what customer need this blueprint serves",
  "scope": "What this blueprint handles and explicitly what it does NOT handle",
  "steps": [
    {{
      "step_number": 1,
      "name": "Step name",
      "input_needed": "What info is needed",
      "action": "What to do",
      "decision_output": "What this step produces"
    }}
  ],
  "minimal_context": "The minimum fields/info from the ticket needed to run this blueprint",
  "exit_conditions": [
    "Condition 1 — when to refuse this blueprint and escalate",
    "Condition 2 — etc"
  ]
}}

Respond with ONLY the JSON. No markdown, no explanation, no code fences."""

    print(f"\n  Generating blueprint for cluster {cid} (n={profile['size']:,})...")
    t0 = time.time()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    elapsed = time.time() - t0
    usage = response.usage_metadata

    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    thinking_tokens = usage.thoughts_token_count or 0

    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    total_thinking_tokens += thinking_tokens

    print(f"    Done in {elapsed:.1f}s | input={input_tokens:,} output={output_tokens:,} "
          f"thinking={thinking_tokens:,}")

    # Parse the response
    raw_text = response.text.strip()
    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
    if raw_text.endswith("```"):
        raw_text = raw_text.rsplit("```", 1)[0]
    raw_text = raw_text.strip()

    try:
        blueprint = json.loads(raw_text)
        blueprint["cluster_id"] = int(cid)
        blueprint["cluster_size"] = int(profile["size"])
        blueprint["type_purity"] = profile["type_purity"]
        blueprint["generation_tokens"] = {
            "input": input_tokens,
            "output": output_tokens,
            "thinking": thinking_tokens,
        }
        blueprints[str(cid)] = blueprint
        print(f"    Blueprint parsed: {len(blueprint['steps'])} steps, "
              f"{len(blueprint['exit_conditions'])} exit conditions")
    except json.JSONDecodeError as e:
        print(f"    ERROR parsing JSON: {e}")
        print(f"    Raw response (first 500 chars): {raw_text[:500]}")
        blueprints[str(cid)] = {"raw_text": raw_text, "parse_error": str(e)}


# ── Save results ──────────────────────────────────────────────────────────────

# Gemini 2.5 Flash pricing (as of 2025):
# Input: $0.15 per 1M tokens (under 200k context)
# Output: $0.60 per 1M tokens
# Thinking: $0.35 per 1M tokens
input_cost = total_input_tokens / 1_000_000 * 0.15
output_cost = total_output_tokens / 1_000_000 * 0.60
thinking_cost = total_thinking_tokens / 1_000_000 * 0.35
total_cost = input_cost + output_cost + thinking_cost

cost_summary = {
    "model": "gemini-2.5-flash",
    "total_input_tokens": total_input_tokens,
    "total_output_tokens": total_output_tokens,
    "total_thinking_tokens": total_thinking_tokens,
    "total_tokens": total_input_tokens + total_output_tokens + total_thinking_tokens,
    "cost_usd": {
        "input": round(input_cost, 6),
        "output": round(output_cost, 6),
        "thinking": round(thinking_cost, 6),
        "total": round(total_cost, 6),
    },
    "note": "One-time cost amortized over all requests routed through blueprints",
}

with open(OUTPUT_DIR / "blueprints.json", "w") as f:
    json.dump(blueprints, f, indent=2)

with open(OUTPUT_DIR / "blueprint_generation_cost.json", "w") as f:
    json.dump(cost_summary, f, indent=2)

print(f"\n{'=' * 60}")
print("PHASE 2 SUMMARY")
print(f"{'=' * 60}")
print(f"\n  Blueprints generated: {len(blueprints)}")
for cid, bp in blueprints.items():
    if "steps" in bp:
        print(f"    Cluster {cid} (n={bp['cluster_size']:,}): "
              f"{len(bp['steps'])} steps, {len(bp['exit_conditions'])} exit conditions")
    else:
        print(f"    Cluster {cid}: PARSE ERROR")

print(f"\n  Token cost:")
print(f"    Input:    {total_input_tokens:,} tokens (${input_cost:.4f})")
print(f"    Output:   {total_output_tokens:,} tokens (${output_cost:.4f})")
print(f"    Thinking: {total_thinking_tokens:,} tokens (${thinking_cost:.4f})")
print(f"    TOTAL:    {total_input_tokens + total_output_tokens + total_thinking_tokens:,} tokens (${total_cost:.4f})")

print(f"\n  Saved to:")
print(f"    output/blueprints.json")
print(f"    output/blueprint_generation_cost.json")
print(f"\n  NEXT: Human-review gate — eyeball each blueprint against sample tickets")
