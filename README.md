# AgentBlueprint

**Can a small model on a laptop do the work of a cloud model, for a fraction of the cost?**

Support tickets are repetitive. Can we group customer tickets into common patterns, write a blueprint/instruction for each pattern, and then have a small local LLM follow them?

Then test it fairly against a cloud model, same instructions for both, graded blind.

We found 46% of the local model's answers to be good enough to send, vs 96% for the cloud model.

Most of the failures from the local model were related to JSON formatting issues. But when it did answer correctly, the answers were clean and filled with proper details.

---

## Local LLM Model

unsloth/gemma-4-E4B-it-GGUF

llama-server -hf unsloth/gemma-4-E4B-it-GGUF:Q8_0 --no-mmproj --temp 1.0 --top-p 0.95 --top-k 64 --port 8080

https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF

## Project description

* Both models got the exact same instructions, so the test measures the model, not a head start.
* A separate model (judge) graded the answers without being told which model wrote which.
* 20 of those grades were checked by a human and agreed 87.5% of the time, so the grading can be trusted.
* The result came out negative, and it is reported that way instead of hidden.

---

## Pipeline

```
61,765 support tickets
        |
   Phase 1   Group tickets into common patterns
        |    14 groups cover 91% of tickets, top 5 cover 70%
        |
   Phase 2   Write one guide per top group
        |    Fits its own tickets 94%, rejects outsiders 100%
        |
   Phase 3   Send the same ticket to both models
             Run 3 times each, grade blind, spot check by hand
```

---

## Results

| | Local (Gemma 4B) | Cloud (Gemini Flash) |
|--|------------------|----------------------|
| Answer good enough  | 46.4% | 96.4% |
| Answer in the required format | 49.1% | 100% |
| Graded the better answer | 37 | 73 |

Answers break down by group:

| Group | Topic | Local | Cloud |
|---------|--------|-------|-------|
| 0 | Medical data security | 0% | 100% |
| 2 | Marketing & brand | 68.8% | 100% |
| 4 | Technical incidents | 66.0% | 100% |
| 7 | Feature requests | 47.4% | 84.2% |
| 9 | Healthcare security | 0% | 94.1% |

The local model holds up on groups 2 and 4, which are most of the volume. It scores 0% on the two medical security groups. Still open: is that a limit of the small model, or the same formatting failure hitting those tickets harder?

In the 49 cases where both answers were good enough to send, the local one was graded better 71% of the time. So the guides work when the model follows them. Treat that 71% as a hint, not proof: it only counts the cases the local model already handled, and the grader was later found to over-call winners.

On the hand check: the human and the grader agreed on "good enough to send" 95% of the time, and on "which is better" 80%. The grader never called a tie; the human called 4. So the gap is real in direction but smaller than the raw numbers say. The grader also leaned slightly toward the local model, not the cloud one, so it was not biased in the cloud's favor.

---

## Cost

This run cost about $2 in API calls.

The cost only matters at volume. One full pass over all 61,765 tickets would cost a few thousand dollars on the cloud model, and a real support team runs this every day. The local model costs about $0 per call after the hardware. The question is whether its answers are good enough to make that trade, and a raw 4B model is not there yet.

Speed and hardware, for context:

| | Local (Gemma 4B) | Cloud (Gemini 2.5 Flash) |
|--|------------------|----------------------|
| Cost per call | ~$0 | paid per token |
| Time per call | 89 sec | 10 sec |
| Hardware | Apple M4, 16GB | none |

---

## What's next

The blocker is formatting, not reasoning, and that is cheap to test:

* Force valid output with grammar constrained decoding. Format success should jump from 49% toward 100%. Open question: do the answers stay good, or just get short?
* Try a bigger local model (Gemma 4 12B, same 16GB hardware) to see if groups 0 and 9 fail on size or on format.
* Split the work: local model on the groups it handles, cloud on the rest. On these results that keeps about 60% local.
* Add images: Gemma 4 12B reads images locally, so "send a screenshot, describe the problem" could run fully on device.

---

## Dataset

**Tobi-Bueck/customer-support-tickets** (Hugging Face, CC BY NC 4.0).

https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets

| | Count |
|--|-------|
| Total tickets | 61,765 |
| English | 28,261 |
| German | 33,504 |
| Training set (80%) | 49,412 |
| Test set (20%) | 12,353 |

The split is frozen before any guide sees a test ticket.

The tickets are in two languages. At first the grouping split by language, not by topic: one big cluster was just the German tickets. Translating German to English with a local model and regrouping fixed it. Every group then held both languages at the same ratio as the full set.

---

## Test suite

84 tests. Each checks a saved result without re-running the experiment or calling any API.

| Area | Tests | Status |
|--------|-------|--------|
| Data splits, leakage, cost | 6 | Pass |
| Phase 1, grouping and coverage | 12 | Pass |
| Phase 2, guides and review | 11 | Pass |
| Phase 3, routing | 9 | Pass |
| Phase 3, fair comparison | 26 | 24 pass, 2 flag a known limit |
| Metrics | 13 | Pass |
| Robustness | 7 | Pass |

The 2 flagged tests are on purpose. They mark a known limit: the grader and the cloud model come from the same family.

```bash
pytest tests/ -v --tb=short
```

---

## Reproduction

Needs Python 3.10+, a Gemini API key in `.env`, and llama.cpp serving Gemma 4B on port 8080.

```bash
pip install pytest pandas numpy scikit-learn sentence-transformers hdbscan umap-learn google-genai python-dotenv
pytest tests/ -v --tb=short
```

Results are saved in `output/`. The tests check them with no API calls.
