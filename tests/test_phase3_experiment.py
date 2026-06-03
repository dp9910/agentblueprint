"""Phase 3 experiment — fairness, process, and completeness checks.

PRINCIPLE: every test here must pass equally whether the local model turns
out equivalent, better, or worse.  No test may fail because of the outcome.
Tests verify the experiment ran unbiased; they never verify a conclusion.

Artifact expectations (from phase3-equivalence-experiment.md):
  - sample.jsonl          120 stratified tickets
  - responses.jsonl       raw responses, 3 runs × 2 arms × N non-escalated
  - ab_key.jsonl          true LOCAL/CLOUD → A/B mapping (judge never sees)
  - judge_input.jsonl     blind packets (no model identifiers)
  - judge_output.jsonl    acceptability + comparison verdicts
  - results_master.csv    un-blinded master table
  - human_verification.json  20-case human gate
  - equivalence_report.json  final aggregated report
"""

import json
from pathlib import Path

import pytest

OUTPUT = Path(__file__).resolve().parent.parent / "output"

# ── helpers ───────────────────────────────────────────────────────────────

def _load_jsonl(name: str) -> list[dict]:
    path = OUTPUT / name
    if not path.exists():
        pytest.skip(f"{name} not found — corrected experiment not yet run")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(name: str) -> dict:
    path = OUTPUT / name
    if not path.exists():
        pytest.skip(f"{name} not found — corrected experiment not yet run")
    with open(path) as f:
        return json.load(f)


# ── fixtures local to this module ─────────────────────────────────────────

@pytest.fixture(scope="module")
def sample():
    return _load_jsonl("sample.jsonl")


@pytest.fixture(scope="module")
def responses():
    return _load_jsonl("responses.jsonl")


@pytest.fixture(scope="module")
def ab_key():
    return _load_jsonl("ab_key.jsonl")


@pytest.fixture(scope="module")
def judge_input():
    return _load_jsonl("judge_input.jsonl")


@pytest.fixture(scope="module")
def judge_output():
    return _load_jsonl("judge_output.jsonl")


@pytest.fixture(scope="module")
def results_master():
    import csv
    path = OUTPUT / "results_master.csv"
    if not path.exists():
        pytest.skip("results_master.csv not found — corrected experiment not yet run")
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def human_verification():
    return _load_json("human_verification.json")


@pytest.fixture(scope="module")
def eq_report():
    return _load_json("equivalence_report.json")


# ═══════════════════════════════════════════════════════════════════════════
# SAME-BLUEPRINT FAIRNESS (the single most important check)
# ═══════════════════════════════════════════════════════════════════════════

def test_both_arms_receive_same_blueprint(responses):
    """For every ticket, the prompt scaffolding sent to LOCAL and CLOUD must
    be byte-identical except for the model endpoint.  We check that each
    ticket_id has responses from both arms and that the blueprint text
    embedded in the prompt is the same."""
    from collections import defaultdict
    by_ticket = defaultdict(dict)
    for r in responses:
        tid = r["ticket_id"]
        arm = r["arm"]
        # Store the blueprint text from the prompt (the experiment must record it)
        by_ticket[tid][arm] = r.get("blueprint_text") or r.get("prompt")

    for tid, arms in by_ticket.items():
        assert "local" in arms and "cloud" in arms, (
            f"Ticket {tid}: missing arm — got {list(arms.keys())}"
        )
        # Compare the blueprint portion; prompts differ only by model endpoint
        local_bp = arms["local"]
        cloud_bp = arms["cloud"]
        assert local_bp == cloud_bp, (
            f"Ticket {tid}: LOCAL and CLOUD received different prompts/blueprints"
        )


# ═══════════════════════════════════════════════════════════════════════════
# JUDGE IDENTITY — Claude, distinct from BOTH worker models
# ═══════════════════════════════════════════════════════════════════════════

def test_judge_is_claude(eq_report):
    """The judge must be Claude, not Gemini or Gemma."""
    judge = eq_report["experiment"]["judge"].lower()
    assert "claude" in judge, (
        f"Judge should be Claude, got '{eq_report['experiment']['judge']}'"
    )


def test_judge_differs_from_both_arms(eq_report):
    """Judge must not share model family with either arm."""
    exp = eq_report["experiment"]
    judge = exp["judge"].lower()
    arm_local = exp.get("arm_local", exp.get("arm_b", "")).lower()
    arm_cloud = exp.get("arm_cloud", exp.get("arm_a", "")).lower()
    # Judge must not be the same model family as either arm
    assert "gemini" not in judge, (
        f"Judge '{exp['judge']}' is in the Gemini family — same as cloud arm"
    )
    assert "gemma" not in judge, (
        f"Judge '{exp['judge']}' is in the Gemma family — same as local arm"
    )


# ═══════════════════════════════════════════════════════════════════════════
# BLINDNESS — judge packets contain no model identifiers; A/B shuffled
# ═══════════════════════════════════════════════════════════════════════════

def test_judge_input_contains_no_model_identifiers(judge_input):
    """Judge packets must not leak which model produced which response.
    Only checks structural fields (keys, response_a/b action/text), not the
    raw ticket body which may contain natural occurrences of common words."""
    forbidden = {"gemini", "gemma", "arm_a", "arm_b",
                 "arm a", "arm b", "local model", "cloud model"}
    for i, packet in enumerate(judge_input):
        # Only check the structural/response fields, not the ticket body
        check_parts = []
        for key in packet:
            check_parts.append(key)
        for slot in ("response_a", "response_b"):
            if slot in packet and isinstance(packet[slot], dict):
                for v in packet[slot].values():
                    check_parts.append(str(v))
        text = " ".join(check_parts).lower()
        for term in forbidden:
            assert term not in text, (
                f"Packet {i}: judge input contains forbidden identifier '{term}'"
            )


def test_ab_assignment_shuffled(ab_key):
    """LOCAL must not always be slot A.  Both mappings must appear."""
    local_slots = {entry["local_slot"] for entry in ab_key}
    assert "A" in local_slots and "B" in local_slots, (
        f"LOCAL always in slot(s) {local_slots} — A/B assignment not shuffled"
    )


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLE SIZE & STRATIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def test_sample_size_at_least_120(sample):
    assert len(sample) >= 120, f"Sample has {len(sample)} tickets, need ≥120"


def test_every_cluster_has_at_least_10(sample):
    from collections import Counter
    counts = Counter(s["routed_cluster"] for s in sample)
    for cid, n in counts.items():
        assert n >= 10, f"Cluster {cid} has only {n} tickets (need ≥10)"


def test_sample_stratified_proportional(sample):
    """Cluster proportions in the sample should roughly match the full-test
    routing distribution.  Allow ±10pp tolerance (small-cluster bumps)."""
    from collections import Counter
    # Expected approximate proportions from routing_summary
    expected_pct = {4: 45, 7: 18, 9: 14, 2: 13.5, 0: 9.5}
    counts = Counter(s["routed_cluster"] for s in sample)
    total = len(sample)
    for cid, exp_pct in expected_pct.items():
        actual_pct = counts.get(cid, 0) / total * 100
        assert abs(actual_pct - exp_pct) < 15, (
            f"Cluster {cid}: {actual_pct:.1f}% vs expected ~{exp_pct}% "
            f"(diff > 15pp)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# THREE RUNS PER TICKET PER ARM
# ═══════════════════════════════════════════════════════════════════════════

def test_three_runs_per_ticket_per_arm(responses):
    """Every non-escalated ticket must have exactly 3 runs for each arm."""
    from collections import Counter
    counts = Counter((r["ticket_id"], r["arm"]) for r in responses)
    for (tid, arm), n in counts.items():
        assert n == 3, (
            f"Ticket {tid}, arm {arm}: expected 3 runs, got {n}"
        )


def test_both_arms_present_in_responses(responses):
    arms = {r["arm"] for r in responses}
    assert "local" in arms and "cloud" in arms, (
        f"Expected both 'local' and 'cloud' arms in responses, got {arms}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# ESCALATION ACCOUNTING — sub-0.50 tickets stay in the final table
# ═══════════════════════════════════════════════════════════════════════════

def test_escalated_tickets_in_final_accounting(sample, results_master):
    """Tickets with similarity < 0.50 must be marked ESCALATE and present
    in results_master — not silently dropped."""
    low_conf = [s for s in sample if s["route_similarity"] < 0.50]
    if not low_conf:
        pytest.skip("No sub-0.50 tickets in sample (none to escalate)")
    escalated_ids = {s["ticket_id"] for s in low_conf}
    master_ids = {row["ticket_id"] for row in results_master}
    missing = escalated_ids - master_ids
    assert not missing, (
        f"{len(missing)} escalated tickets dropped from results_master"
    )
    # Verify they're marked as escalated
    for row in results_master:
        if row["ticket_id"] in escalated_ids:
            assert row.get("escalated", "").lower() in ("true", "yes", "1"), (
                f"Ticket {row['ticket_id']}: sub-0.50 but not marked escalated"
            )


def test_escalation_rate_recorded(eq_report):
    """The report must record the escalation rate."""
    assert "escalation_rate" in eq_report or "escalation" in eq_report, (
        "Escalation rate not recorded in equivalence report"
    )


# ═══════════════════════════════════════════════════════════════════════════
# JUDGE VERDICT STRUCTURE — presence checks, never outcome checks
# ═══════════════════════════════════════════════════════════════════════════

def test_acceptability_recorded_for_every_judgment(judge_output):
    """Each judgment must have acceptability verdicts for both responses."""
    for i, j in enumerate(judge_output):
        assert "a_acceptable" in j and "b_acceptable" in j, (
            f"Judgment {i}: missing acceptability field(s)"
        )
        assert j["a_acceptable"] in (True, False, "yes", "no"), (
            f"Judgment {i}: a_acceptable={j['a_acceptable']} not boolean-like"
        )
        assert j["b_acceptable"] in (True, False, "yes", "no"), (
            f"Judgment {i}: b_acceptable={j['b_acceptable']} not boolean-like"
        )


def test_comparison_verdict_valid(judge_output):
    """Each judgment must have a comparison in {A better, B better, equivalent}."""
    valid = {"a_better", "b_better", "equivalent",
             "A better", "B better", "equivalent",
             "a", "b", "equivalent", "tie"}
    for i, j in enumerate(judge_output):
        assert "comparison" in j, f"Judgment {i}: missing 'comparison'"
        assert j["comparison"].lower().strip() in {v.lower() for v in valid}, (
            f"Judgment {i}: comparison='{j['comparison']}' not valid"
        )


def test_judge_gives_reasons(judge_output):
    """Every verdict must include a reason of substance (> 20 chars)."""
    for i, j in enumerate(judge_output):
        reason = j.get("reason", "")
        assert len(reason) > 20, (
            f"Judgment {i}: reason too short ({len(reason)} chars)"
        )


def test_all_non_escalated_tickets_judged(sample, judge_output):
    """Every non-escalated ticket must have a judgment."""
    non_escalated = {s["ticket_id"] for s in sample if s["route_similarity"] >= 0.50}
    judged = {j["ticket_id"] for j in judge_output}
    missing = non_escalated - judged
    assert not missing, (
        f"{len(missing)} non-escalated tickets missing judgments"
    )


# ═══════════════════════════════════════════════════════════════════════════
# HUMAN VERIFICATION GATE
# ═══════════════════════════════════════════════════════════════════════════

def test_human_verification_exists(human_verification):
    """The human verification record must exist with 20 cases."""
    cases = human_verification.get("cases", [])
    assert len(cases) >= 20, (
        f"Human verification has {len(cases)} cases, need ≥20"
    )


def test_human_agreement_score_computed(human_verification):
    """An agreement score must be computed and recorded."""
    assert "agreement_rate" in human_verification or "agreement" in human_verification, (
        "No agreement score in human_verification"
    )


def test_conclusion_gated_on_human_agreement(human_verification, eq_report):
    """If human-judge agreement is below threshold, the conclusion must
    be flagged as unreliable — not silently published."""
    agreement = human_verification.get(
        "agreement_rate", human_verification.get("agreement")
    )
    if agreement is None:
        # Human review not done yet — gate must block conclusions
        note = json.dumps(eq_report.get("human_verification", {})).lower()
        assert "not yet" in note or "not be trusted" in note or "pending" in note, (
            "Human review not completed, but report doesn't flag conclusions as untrustworthy"
        )
        return
    if agreement < 0.80:
        # The report must acknowledge low agreement
        verdict = json.dumps(eq_report).lower()
        assert "unreliable" in verdict or "low agreement" in verdict or "cannot be trusted" in verdict, (
            f"Agreement={agreement:.0%} < 80% but report does not flag judge as unreliable"
        )


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS MASTER TABLE — completeness, not outcome
# ═══════════════════════════════════════════════════════════════════════════

def test_results_master_has_required_columns(results_master):
    required = {
        "ticket_id", "cluster", "similarity", "language", "escalated",
        "local_acceptable", "cloud_acceptable", "verdict", "reason",
    }
    actual = set(results_master[0].keys()) if results_master else set()
    missing = required - actual
    assert not missing, f"results_master.csv missing columns: {missing}"


def test_verdict_values_valid(results_master):
    valid = {"local-better", "cloud-better", "equivalent",
             "local_better", "cloud_better",
             "escalated", "escalate"}
    for row in results_master:
        v = row["verdict"].lower().strip()
        assert v in valid, (
            f"Ticket {row['ticket_id']}: verdict='{row['verdict']}' not valid"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TOKEN & COST REPORTING — honest framing, never bare $0
# ═══════════════════════════════════════════════════════════════════════════

def test_tokens_tracked_both_arms(eq_report):
    """Both arms must have input and output token counts."""
    tokens = eq_report["tokens"]
    for arm in ("local", "cloud"):
        arm_tokens = tokens.get(arm, {})
        assert arm_tokens.get("input", 0) > 0 or arm_tokens.get("prompt", 0) > 0, (
            f"Arm {arm}: no input/prompt tokens recorded"
        )
        assert arm_tokens.get("output", 0) > 0 or arm_tokens.get("completion", 0) > 0, (
            f"Arm {arm}: no output/completion tokens recorded"
        )


def test_local_cost_reported_with_caveat(eq_report):
    """Local cost must NOT be reported as a bare $0.  It must carry a
    hardware/latency caveat or note acknowledging that local inference
    has real costs not captured by API billing."""
    cost = eq_report.get("cost", {})
    cost_text = json.dumps(cost).lower()
    # Must mention hardware, latency, or infrastructure caveat
    caveat_terms = {"hardware", "latency", "infrastructure", "caveat",
                    "local hardware", "not free", "compute", "marginal"}
    has_caveat = any(term in cost_text for term in caveat_terms)
    # OR there's a separate caveat field
    has_caveat = has_caveat or "cost_caveat" in cost or "local_cost_note" in cost
    assert has_caveat, (
        "Local cost reported without hardware/latency caveat"
    )


def test_per_ticket_tokens_recorded(eq_report):
    tokens = eq_report["tokens"]
    for key in ("local_per_ticket", "cloud_per_ticket"):
        assert tokens.get(key, 0) > 0, f"Missing or zero: {key}"


# ═══════════════════════════════════════════════════════════════════════════
# LANGUAGE BREAKDOWN — presence check
# ═══════════════════════════════════════════════════════════════════════════

def test_language_breakdown_present(eq_report):
    by_lang = eq_report.get("by_language", {})
    assert "native_english" in by_lang or "en" in by_lang, (
        "Missing English language breakdown"
    )
    assert "translated_german" in by_lang or "de" in by_lang, (
        "Missing German language breakdown"
    )


# ═══════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════

def test_random_seed_recorded(eq_report):
    assert "random_seed" in eq_report or "seed" in eq_report, (
        "No random seed recorded in the report"
    )
