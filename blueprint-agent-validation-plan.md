# Blueprint-Routed Agents: A Validation Project

**Goal:** Prove (or disprove) that routing high-volume, pattern-dominated customer requests through pre-derived "blueprints" — and escalating only genuine outliers to a full agent — reduces token cost dramatically while preserving resolution correctness, compared to running a from-scratch agent on every request.

**One-line thesis:** If ~70% of incoming requests are near-duplicates of past ones, a cheap classify→template→execute path can handle them at a fraction of the token cost of a full agentic loop, at statistically indistinguishable quality, while safely escalating the long tail.

---

## 0. Outcome We Are Measuring

A single comparison table and three charts that answer:

1. **Cost:** What is the blended token cost per request *with* the blueprint system vs. *without* (full agent on everything)?
2. **Correctness:** Is the blueprint path's resolution quality statistically indistinguishable from the full agent's?
3. **Coverage & safety:** What fraction of traffic gets routed to a blueprint, and how often is a blueprint *wrongly* applied to a request that didn't fit?
4. **Break-even:** After how many requests does the one-time blueprint-generation cost pay for itself?

If correctness drops or misroute rate is high, the method fails — and that is a valid, publishable result. Do not tune the experiment to produce a favorable outcome; the deliverable is an honest verdict.

---

## 1. Guardrails (read before building)

- **Honest baseline.** The "without" arm is a real from-scratch agentic loop that reasons about each request, reads what it needs, and decides its own steps — NOT an artificially wasteful "scan everything" strawman. The blueprint system must beat *this*.
- **Held-out test set.** Blueprints are derived ONLY from training data. All cost/correctness numbers come from a test set the system has never seen. No leakage.
- **Run-to-run variance is real.** Token usage on the same task can vary widely between runs. Every measured task runs **≥3 times** per arm; report medians and spread, never single runs.
- **Cold cache.** Disable/avoid prompt caching during measurement, or the second run of a task looks artificially cheap. Document cache state explicitly.
- **Same model, same prompts, same data state** across both arms. The only variable is the routing+blueprint mechanism.
- **Amortize honestly.** Blueprint generation costs tokens up front. Report it as a one-time cost and compute the break-even point; do not hide it.
- **Separate the judge from the workers.** Correctness is scored by a different model/process than the one producing resolutions, ideally blind to which arm produced which answer.

---

## 2. Data

- **Primary corpus:** `Tobi-Bueck/customer-support-tickets` (~61.8k real helpdesk tickets, routing-tagged). License is CC-BY-NC-4.0 — fine for research, note it.
- **Secondary / ground-truth intents:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (intent-labeled across 27 intents incl. an insurance vertical) — use its labels to validate that discovered clusters correspond to real intents.
- **Fast-prototype set:** `hblim/customer-complaints` or `gorkemsevinc/customer_support_tickets` for quick iteration before scaling up.
- **Scale:** develop on ~10k, run the final evaluation on the full corpus. Do not assume more data = more insight.
- **Split:** 80/20 train/test, stratified by category. If timestamps exist, ALSO produce a time-based split (train on older, test on newer) for the decay test in §7.

---

## 3. Phase 1 — Concentration Test (make-or-break, do this first)

This phase is cheap and either validates or kills the premise before any money is spent.

1. Embed every training ticket with a local embedding model (e.g. a small sentence-transformer running on-device).
2. Cluster the embeddings with HDBSCAN (preferred — it leaves true outliers unclustered, which is exactly the escalation signal) and also k-means for comparison.
3. Plot the **concentration curve**: cumulative % of tickets covered vs. number of clusters.
4. **Decision gate:** Does a small number of clusters cover a large share (the hypothesized ~70%)? If concentration is weak, the blueprint idea does not apply to this domain — STOP and report that finding. If strong, proceed.
5. Validate clusters against the bitext intent labels: do discovered buckets map to coherent real-world intents?

**Deliverable:** concentration curve + cluster-to-intent coherence summary.

---

## 4. Phase 2 — Blueprint Generation (one-time, amortized cost)

For each major cluster from Phase 1:

1. Sample a set of representative *resolved* tickets from the cluster (use correctly-resolved cases where labels allow).
2. Have the strong model derive a **structured blueprint**, not prose. Each blueprint must specify: the intent it serves; an ordered list of steps; for each step the input needed, the action, and the decision/output; the minimal context the executor needs; and explicit **exit conditions** ("if the request mentions X / lacks Y, this blueprint does NOT apply — escalate").
3. Record the total token cost of generating all blueprints. This is the one-time investment for the break-even calculation.
4. **Human-review checkpoint:** every blueprint is reviewed before it can be used at runtime. Flag this as a required gate, not optional — in regulated domains a wrong procedure has legal consequences.

**Deliverable:** a reviewed blueprint library (one structured blueprint per major cluster) + total generation cost.

---

## 5. Phase 3 — The Two Arms (the core experiment)

Run every test-set request through both arms. Capture per-request metrics for each.

**Arm A — Baseline (no blueprints):** a from-scratch agentic loop. The agent decides its own approach, reads what it needs, and produces a resolution.

**Arm B — Blueprint system:**
1. **Classify:** embed the request, find nearest cluster centroid. If confidently inside a known bucket → assign that blueprint. If far from all centroids (outlier) → route to the Arm-A full-agent path.
2. **Execute (in-bucket):** run the cheap path — minimal context (request + blueprint only), with the LLM called only for the narrow fill-in-the-blank judgments the blueprint defines.
3. **Escalate (outlier):** hand off to the full-agent path; optionally flag for human-in-the-loop.

**Optional third tier (recommended):** a "trivial" path with NO LLM call for buckets that need none (e.g. password reset), a middle blueprint+small-LLM path, and the full-agent top tier. Three tiers is more realistic and shows larger savings; report it separately so the two-tier result stays clean.

---

## 6. Metrics (capture per request, both arms)

- Total **input** tokens and **output** tokens (input usually dominates — track it carefully, cumulatively across multi-turn loops).
- Total **cost** (USD) per request.
- **Wall-clock latency** (the blueprint path may add classification latency; the full agent adds reasoning latency — report honestly).
- Number of tool calls / loop iterations.
- **Path taken** (which bucket, or escalated).
- **Correctness / resolution quality** (see §6.1).
- **Misroute flag:** was a blueprint applied to a request its own exit conditions should have rejected? (the key safety metric)

### 6.1 Measuring correctness (do not skip — this is the trap)

Cost is trivial to measure; correctness is where bad experiments cheat. Use at least one, ideally both:
- **Label-based:** for the bitext-derived portion with known intents/reference answers, check whether the resolution matches the correct intent-appropriate action.
- **LLM-as-judge:** a separate strong model scores Arm A and Arm B resolutions **blind** (not told which arm produced which) on a fixed rubric: did it address the issue, was it factually correct, was the action appropriate.
- **Manual audit:** hand-inspect every case where the two arms *disagree* and every flagged misroute. These are the most informative rows.

---

## 7. Robustness Tests (what makes the result convincing rather than a demo)

- **Confidence-threshold sweep:** vary how confident the classifier must be before using a blueprint; plot **cost savings vs. misroute rate** across thresholds. This "money chart" shows the tunable safety/cost tradeoff a cautious owner actually needs.
- **Coverage-decay test:** using the time-based split, build blueprints on older data, test on newer, and watch whether coverage drops and outlier rate rises over time — demonstrating the maintenance problem and whether rising outlier rate is a usable early-warning signal.
- **Cold-start note:** state explicitly that the method needs existing history; a brand-new business has no blueprints to derive. Note the possible mitigation (start from generic templates, personalize as volume accrues) but do not test it unless time allows.
- **Sensitivity to cluster count:** show how results change if you use fewer/more blueprints, so the headline number isn't an artifact of one lucky configuration.

---

## 8. Final Deliverables

1. **Concentration curve** (Phase 1) — does the premise even hold for this data.
2. **Cost-per-request by arm**, broken out by path (trivial / blueprint / full agent), with medians and spread over the ≥3 runs.
3. **Blended cost comparison** across the whole test set — the headline number — with the % reduction.
4. **Correctness comparison** — Arm A vs Arm B, with the statistical test showing whether quality is indistinguishable.
5. **Threshold tradeoff curve** (cost savings vs. misroute rate).
6. **Break-even chart** — requests processed vs. cumulative cost for each arm, including the one-time blueprint-generation cost, showing where Arm B overtakes Arm A.
7. **A short written verdict:** strong result (≥30% cost cut at ≥90% of baseline quality with low misroute), mixed, or negative — stated plainly, whichever it is.

---

## 9. Suggested Build Order

1. Phase 1 concentration test on the prototype set — get the make-or-break signal first.
2. If it passes, scale Phase 1 to the full corpus and lock the train/test split.
3. Phase 2 blueprint generation + human review.
4. Build the classifier/router and the blueprint executor.
5. Phase 3 Arm A baseline run (≥3× per test item).
6. Phase 3 Arm B run (≥3× per test item).
7. Correctness judging (blind).
8. Robustness tests (§7).
9. Assemble deliverables (§8) and write the verdict.

---

## 10. Failure Modes to Watch

- Measuring tokens but not correctness → a meaningless "cheaper" result.
- Single-run measurements → swamped by variance.
- Prompt-cache contamination → fake savings.
- Strawman baseline → unconvincing to anyone skeptical.
- Blueprint exit conditions too loose → high misroute rate → unsafe in any real deployment.
- Hiding the blueprint-generation cost → dishonest economics.

If the method genuinely doesn't beat the baseline, say so. A clean negative result is worth more than a rigged positive one.
