# Phase 3 — Equivalence Experiment: Local+Blueprint vs Cloud

## Goal (read first, do not lose sight of this)
Test ONE claim: **a local 4B model guided by an expert-generated blueprint produces customer-acceptable outcomes equivalent to a cloud model, on the majority of routine tickets.**

The win condition is EQUIVALENCE, not superiority. We are NOT trying to prove Gemma beats Gemini. We are proving Gemma is *good enough* to replace it on common cases. Cost savings is a downstream consequence we report honestly, not the headline.

Do not optimize any step to make the local model "win." If the result is that they are NOT equivalent, that is a valid and valuable finding — report it plainly.

---

## Fixed inputs (already produced in earlier phases — load, do not regenerate)
- The 14-cluster solution on the normalized (English + translated-German) corpus.
- The 5 head-cluster blueprints (JSON: intent, scope, steps, minimal_context, exit_conditions).
- The frozen train/test split. **Only the TEST partition is used here.** If blueprint generation ever touched a test ticket, stop and re-split.
- The embedding router (cluster centroids from training data).

---

## Step 0 — Setup and guardrails
1. Confirm the local model server is running and reachable (Gemma 4B via llama.cpp at its local endpoint). Do a one-ticket smoke test; abort if unreachable.
2. Confirm cloud API access (Gemini Flash) with a one-ticket smoke test.
3. Set a fixed random seed and write it to the output file. Everything reproducible.
4. Create an output directory. All artifacts (raw responses, judge calls, final tables) get written to disk as they are produced — never hold results only in memory.

---

## Step 1 — Build the sample
1. From the TEST partition, draw a stratified sample of **120 tickets**, allocated across the 5 clusters **proportional to the full-test routing distribution** (so Cluster 4 ≈ 45%, Cluster 7 ≈ 18%, Cluster 9 ≈ 14%, Cluster 2 ≈ 13.5%, Cluster 0 ≈ 9%).
2. If any cluster receives fewer than 10 tickets at that proportion, bump it to 10 and note the over-sampling in the output (so small clusters are reportable, not n=2).
3. Preserve each ticket's: id, subject, body, routed cluster, router cosine similarity, language flag (en / translated-de).
4. Write the sample to `sample.jsonl`. This is frozen — do not redraw it later.

---

## Step 2 — Apply the router confidence threshold (this is part of the system, not a filter)
1. For each sampled ticket, check router similarity.
2. If similarity < 0.50 → mark outcome as **ESCALATE (low confidence)**. This ticket does NOT go to a blueprint. It counts as a *correct system behavior*, not a failure.
3. If similarity ≥ 0.50 → proceeds to the model arms below.
4. Record the escalation rate. Do NOT exclude escalated tickets from the final accounting — they are part of how the system performs.

---

## Step 3 — Generate outcomes from BOTH arms (same scaffolding — this is the fairness control)
For every non-escalated ticket, generate two responses using the **same blueprint** for both models. This is the equivalence comparison. Do not give one model a thin prompt and the other the blueprint.

Prompt (identical for both, only the model differs):
```
You are a customer support agent. Follow the BLUEPRINT below step-by-step.

BLUEPRINT:
{full blueprint JSON for the ticket's routed cluster}

TICKET:
Subject: {subject}
Body: {body}

Work through every step in the blueprint. If any exit condition triggers,
recommend escalation instead of forcing a response.
```

1. **Arm LOCAL:** Gemma 4B + blueprint → response.
2. **Arm CLOUD:** Gemini Flash + blueprint → response.
3. **Run each arm 3 times per ticket** (run-to-run variance is large; we need medians not single shots). Use the same seed policy each run.
4. For each call, record: full response text, input tokens, output tokens, wall-clock latency, which arm, which run number.
5. Write every raw response to `responses.jsonl` as it is produced (resumable — if it crashes at ticket 60, do not restart from 0).

---

## Step 4 — Prepare the BLIND judging packets
This is where bias is killed. The judge must not be able to tell which system produced which response.

1. For each ticket, pick one representative response per arm. (Use the median-length of the 3 runs, or run 1 consistently — state which in the output.)
2. Randomly assign the two responses to slots **"Response A"** and **"Response B"** — coin-flip per ticket so LOCAL is not always A. Record the true mapping in a SEPARATE file (`ab_key.jsonl`) the judge never sees.
3. Strip any identifying markers (model names, token counts, formatting tells if obvious).
4. Write judging packets to `judge_input.jsonl`: ticket subject+body, Response A, Response B. Nothing else.

---

## Step 5 — Judge (Claude Code is the judge, working ONLY from the blind packets)
For each packet, the judge answers three things — answering the EQUIVALENCE question, not a generic score:

1. **Acceptability (per response):** "Would this be acceptable to send to a customer?" → A: yes/no, B: yes/no. (Acceptable = addresses the issue, no false dismissals, professional, actionable.)
2. **Comparison:** "Which response is better, or are they equivalent?" → {A better / B better / equivalent}.
3. **Reason:** one or two sentences.

Rules for the judge:
- Judge only what is in the packet. Do not guess which model wrote which.
- Penalize a response that *dismisses or deflects a valid request* ("out of scope," "we can't help") — that is worse than a generic-but-helpful answer, even if shorter or longer.
- Do NOT reward verbosity. More questions / more detail is only better if the ticket warranted it. A response that demands five diagnostic items for a trivial issue is worse service, not better.

Write judgments to `judge_output.jsonl`.

---

## Step 6 — Un-blind and join
1. Join `judge_output.jsonl` with `ab_key.jsonl` to map A/B back to LOCAL/CLOUD.
2. Produce the master table, one row per ticket: id, cluster, similarity, language, escalated?, LOCAL acceptable?, CLOUD acceptable?, verdict (local-better / cloud-better / equivalent), judge reason.
3. Write to `results_master.csv`.

---

## Step 7 — Human verification gate (do not skip — this validates the judge itself)
1. Select 20 tickets at random from the judged set.
2. Present them to the user (the human) in the SAME blind A/B format, with the judge's verdict hidden.
3. The human records their own acceptability + comparison calls.
4. Compute agreement between human and Claude-judge.
5. **Decision:** if agreement is high (≈80%+), the judge is trustworthy — proceed. If low, the judge is unreliable — STOP, report that the automated verdicts cannot be trusted, and do not publish aggregate conclusions based on them.

---

## Step 8 — Aggregate (the numbers that answer the thesis)
Compute and report:
1. **% of cases where the LOCAL response was acceptable** ← the headline number. Thesis holds if this is high (target ~70%+ of routine, non-escalated tickets).
2. **% where both were acceptable / equivalent (blind).** The equivalence rate.
3. **Breakdown of verdicts:** local-better / cloud-better / equivalent.
4. **Escalation rate** (from Step 2) — how much of volume the system honestly punts rather than guessing.
5. **Per-cluster breakdown** — but only for clusters with ≥10 tickets; mark the rest "insufficient n."
6. **Acceptability split by language** (en vs translated-de) — does translation degrade outcomes?
7. **Tokens & latency** both arms, reported honestly: local uses MORE tokens per ticket (blueprint in every prompt) but at ~$0 marginal API cost; cloud is fewer tokens but paid. State that local only wins economically at volume and assumes capable local hardware. Do not report local cost as "$0" without the hardware/latency caveat.

---

## Step 9 — Write the verdict
State plainly, in this order:
1. Did local+blueprint reach acceptable-outcome parity on the majority of routine tickets? (yes/no + the %)
2. Where did it fail, and was the failure the model or the router?
3. The honest cost framing (per-ticket vs at-scale; hardware caveat).
4. The limitation: this is synthetic-ish support data; "acceptable" here is a softer bar than a real business's standard. Claim a proof-of-concept, not production readiness.

---

## Hard rules (violations invalidate the result)
- Same blueprint to both models. No thin-prompt strawman.
- Judging is blind and A/B randomized. Judge never learns model identity.
- Escalated and mis-routed tickets stay in the accounting.
- ≥3 runs per ticket per arm; report medians.
- Human-verify the judge on 20 cases before trusting aggregates.
- If the result is "not equivalent," report it as-is. Do not tune to win.
- Write artifacts to disk progressively; the run must be resumable.
