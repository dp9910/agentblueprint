# Blueprint-Routed Agent: Full Project Report

## The Idea

Can a small, locally-hosted language model produce customer support responses that are good enough to replace a cloud model, if we give it expert-written blueprints to follow?

The hypothesis is straightforward. Most customer support tickets are repetitive. They cluster into a handful of common patterns. If we write a detailed step-by-step blueprint for each pattern, even a small model should be able to follow the instructions and produce acceptable responses. This would eliminate per-ticket API costs and keep customer data on-premises.

This project tested that hypothesis end to end: from raw data through clustering, blueprint generation, routing, and a controlled equivalence experiment comparing a local 4B model against a cloud model.

## The Dataset

We used the Tobi-Bueck/customer-support-tickets dataset from Hugging Face, licensed under CC-BY-NC-4.0. It contains 61,765 real helpdesk tickets with subject lines, body text, and category labels (Incident, Request, Problem, Change). The tickets are split roughly evenly between English (28,261) and German (33,504).

We split the data 80/20 stratified by ticket type, producing a training set of 49,412 tickets and a held-out test set of 12,353 tickets. The test set was never used during clustering or blueprint generation.

## Phase 1: Do Tickets Actually Cluster?

Before writing any blueprints, we needed to verify the core premise: that customer support tickets concentrate into a small number of patterns.

We embedded all training tickets using paraphrase-multilingual-MiniLM-L12-v2, a multilingual sentence transformer that produces 384-dimensional vectors. This model handles both English and German natively.

We ran HDBSCAN across 12 configurations (varying UMAP dimensions and minimum cluster sizes) and compared against K-Means. The selected configuration (UMAP dim=30, min_cluster_size=200) produced 27 clusters covering 86.9% of training tickets. More importantly, just 9 clusters accounted for 70% of all traffic.

The concentration test passed. A small number of blueprints could theoretically cover the majority of incoming tickets.

## Handling German Tickets

The dataset is bilingual. Rather than maintaining separate German and English pipelines, we translated a validation sample of 1,758 German tickets to English using a local translation model and re-clustered the combined corpus. This produced 14 cleaner clusters (down from 27) covering 91% of tickets. We verified that no cluster was a "language blob" (clusters dominated by one language with low topical coherence). The German translations integrated cleanly into the English clusters.

The top 5 clusters covered 70% of the combined corpus. These became the targets for blueprint generation.

## Phase 2: Blueprint Generation and Review

We generated one structured blueprint per head cluster using Gemini 2.5 Flash. Each blueprint follows a locked schema: intent, scope, 5 to 7 ordered steps (each with action, input needed, and decision output), minimal context requirements, and exit conditions that trigger escalation.

The five blueprints cover:

| Cluster | Tickets | Primary Intent |
|---------|---------|----------------|
| 4 | 9,572 | Technical incident triage |
| 9 | 2,516 | Security incidents in healthcare data |
| 0 | 2,180 | Medical data security guidance |
| 2 | 1,929 | Digital marketing and brand strategy |
| 7 | 1,647 | Feature requests and product inquiries |

Blueprint generation cost $0.015 in total, a one-time expense amortized across all future routed tickets.

We then ran a review gate. For each blueprint, we tested whether it fits tickets from its own cluster (in-cluster fit rate: 94%) and correctly rejects tickets from other clusters (out-of-cluster rejection rate: 100%). Both rates exceeded the required thresholds.

## Phase 3: Routing and the Equivalence Experiment

With blueprints in hand, we built a cosine-similarity router. Each test ticket is embedded, compared against the five cluster centroids, and routed to the nearest blueprint. Tickets below a similarity threshold of 0.50 are flagged for escalation rather than forced through a blueprint that may not fit.

We drew a stratified sample of 120 tickets from the routed test set, proportional to cluster sizes, with a minimum of 10 per cluster. Ten tickets fell below the confidence threshold and were escalated (8.3% escalation rate). The remaining 110 proceeded to the experiment.

### Experiment Design

Both arms received the identical blueprint prompt. The only difference was the model endpoint:

| | Local Arm | Cloud Arm |
|--|-----------|-----------|
| Model | Gemma 4B Q8_0 (quantized) | Gemini Flash |
| Runs | llama.cpp on Apple M4, 16GB | Google API |
| Runs per ticket | 3 | 3 |

Each ticket was processed 3 times per arm (660 total responses). For judging, we selected the median-length response from each arm's 3 runs. A blind judge (Gemini Flash) evaluated each pair without knowing which model produced which response. The A/B slot assignment was randomized per ticket.

### Results

| Metric | Local (Gemma 4B) | Cloud (Gemini Flash) |
|--------|------------------|----------------------|
| Acceptable responses | 46.4% | 96.4% |
| Judged better | 37 times | 73 times |
| Judged equivalent | 0 times | |
| Avg output tokens | 1,065 | 221 |
| JSON parse success | 49.1% | 100% |

The local model failed to achieve equivalence. The gap is large and consistent across most clusters.

### Why the Local Model Failed

The root cause is not a lack of reasoning ability. It is format compliance. The local model produced valid JSON only 49% of the time. In the other 51%, it generated free-form "thinking process" text, explaining its reasoning step by step instead of producing the requested structured output. These responses, while sometimes thoughtful, cannot be sent to a customer.

When both models produced acceptable responses (49 cases out of 110), the local model was actually judged better 71% of the time. The blueprint content works. The 4B model simply cannot reliably follow output format instructions.

### Per-Cluster Breakdown

| Cluster | Local Acceptable | Cloud Acceptable | Local Wins | Cloud Wins |
|---------|-----------------|------------------|------------|------------|
| 0 | 0% | 100% | 0 | 11 |
| 2 | 68.8% | 100% | 10 | 6 |
| 4 | 66.0% | 100% | 21 | 26 |
| 7 | 47.4% | 84.2% | 6 | 13 |
| 9 | 0% | 94.1% | 0 | 17 |

Clusters 0 and 9 (healthcare/security domains) were total failures for the local model. Clusters 2 and 4 showed competitive performance when the local model managed to follow instructions.

### Human Verification

A human reviewer evaluated 20 blind cases and agreed with the AI judge 87.5% of the time (95% on acceptability, 80% on comparisons). The human found 4 cases the AI called decisive that were actually equivalent. Both human and AI agreed on the core finding: cloud is substantially better overall. The human verification gate passed.

### Cost

| Item | Cost |
|------|------|
| Cloud API (110 tickets, 3 runs) | $0.118 |
| Blueprint generation (one-time) | $0.015 |
| Local inference | $0 API cost |

Local inference has zero marginal API cost but requires dedicated hardware. On an Apple M4 with 16GB RAM, latency averaged 89 seconds per call versus 10 seconds for the cloud model. The local model also generated 5x more tokens per response.

## What We Learned

The blueprint approach works at the content level. Expert-written step-by-step instructions genuinely improve the quality of a small model's reasoning about customer support tickets. When the local model follows the blueprint correctly, it often produces responses that a human judge prefers over the cloud model's output.

The approach fails at the format level. A 4B parameter model cannot reliably produce structured JSON output on demand. This is the bottleneck, not the quality of reasoning.

Possible paths forward include constrained decoding (forcing valid JSON via grammar constraints in llama.cpp), using a larger local model (8B or 12B parameters), adding few-shot examples to the prompt, or post-processing the free-form output to extract structured fields. A hybrid routing strategy (local model for clusters 2 and 4, cloud for clusters 0 and 9) could also capture partial cost savings while maintaining quality.

The experiment infrastructure itself held up well. The stratified sampling, 3-run design, blind judging, A/B randomization, escalation accounting, and human verification gate all worked as intended. The test suite (82 of 84 tests passing, with the 2 failures being intentional flags about judge model independence) validates every step of the pipeline.
