"""Metrics — token tracking, cost framing, language breakdown, judge reasons.

PRINCIPLE: tests verify metrics were recorded correctly and completely.
No test asserts a particular outcome value (who won, quality threshold,
cost being zero).  Tests that would fail on a "local model loses" result
do not belong here.
"""

import json
from pathlib import Path

import pytest

OUTPUT = Path(__file__).resolve().parent.parent / "output"


def _load_json(name: str) -> dict:
    path = OUTPUT / name
    if not path.exists():
        pytest.skip(f"{name} not found — corrected experiment not yet run")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def eq_report():
    return _load_json("equivalence_report.json")


@pytest.fixture(scope="module")
def judge_output():
    path = OUTPUT / "judge_output.jsonl"
    if not path.exists():
        pytest.skip("judge_output.jsonl not found")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ═══════════════════════════════════════════════════════════════════════════
# TOKEN ACCOUNTING
# ═══════════════════════════════════════════════════════════════════════════

def test_input_output_tokens_separate(eq_report):
    """Both arms must track input and output tokens independently."""
    tokens = eq_report["tokens"]
    for arm in ("local", "cloud"):
        t = tokens.get(arm, {})
        has_input = t.get("input", 0) > 0 or t.get("prompt", 0) > 0
        has_output = t.get("output", 0) > 0 or t.get("completion", 0) > 0
        assert has_input, f"{arm}: input/prompt tokens missing or zero"
        assert has_output, f"{arm}: output/completion tokens missing or zero"


def test_per_ticket_averages_present(eq_report):
    tokens = eq_report["tokens"]
    for key in ("local_per_ticket", "cloud_per_ticket"):
        assert tokens.get(key, 0) > 0, f"{key} missing or zero"


def test_judge_tokens_tracked(eq_report):
    """Judge token usage should also be recorded (it's a real cost)."""
    tokens = eq_report["tokens"]
    judge_t = tokens.get("judge", {})
    total_judge = sum(v for v in judge_t.values() if isinstance(v, (int, float)))
    assert total_judge > 0, "Judge token usage not recorded"


# ═══════════════════════════════════════════════════════════════════════════
# COST FRAMING — honest, with caveats
# ═══════════════════════════════════════════════════════════════════════════

def test_cloud_cost_recorded(eq_report):
    """Cloud arm must have a positive API cost."""
    cost = eq_report.get("cost", {})
    cloud = cost.get("cloud_usd", cost.get("arm_a_cloud_usd", 0))
    assert cloud > 0, "Cloud cost should be > 0"


def test_local_cost_not_bare_zero(eq_report):
    """Local cost must not be reported as bare $0 without acknowledging
    that local inference has hardware/compute/latency costs."""
    cost = eq_report.get("cost", {})
    cost_text = json.dumps(cost).lower()
    caveat_terms = {"hardware", "latency", "infrastructure", "caveat",
                    "compute", "marginal", "not free", "electricity"}
    has_caveat = any(term in cost_text for term in caveat_terms)
    has_caveat = has_caveat or "cost_caveat" in cost or "local_cost_note" in cost
    assert has_caveat, (
        "Local cost reported without hardware/latency caveat — "
        "bare $0 is misleading"
    )


def test_blueprint_generation_cost_included(eq_report):
    """The one-time blueprint generation cost must be reported."""
    cost = eq_report.get("cost", {})
    bp_cost = cost.get("blueprint_gen_usd", cost.get("blueprint_generation_usd", 0))
    assert bp_cost > 0, "Blueprint generation cost not recorded"


# ═══════════════════════════════════════════════════════════════════════════
# ACCEPTABILITY & EQUIVALENCE — recorded, not outcome-gated
# ═══════════════════════════════════════════════════════════════════════════

def test_acceptability_rates_recorded(eq_report):
    """The report must state local_acceptable% and cloud_acceptable%.
    We check they exist and are in [0, 100], not that they hit any target."""
    for key in ("local_acceptable_pct", "cloud_acceptable_pct"):
        val = eq_report.get(key, eq_report.get("quality", {}).get(key))
        assert val is not None, f"{key} not recorded in report"
        assert 0 <= val <= 100, f"{key}={val} outside [0,100]"


def test_equivalence_rate_recorded(eq_report):
    """The equivalence rate must be recorded — it's the headline metric."""
    rate = eq_report.get("equivalence_rate",
                         eq_report.get("quality", {}).get("equivalence_rate"))
    assert rate is not None, "equivalence_rate not found in report"
    assert 0 <= rate <= 100, f"equivalence_rate={rate} outside [0,100]"


def test_verdict_breakdown_recorded(eq_report):
    """Verdict breakdown: local-better / cloud-better / equivalent counts."""
    breakdown = eq_report.get("verdict_breakdown",
                              eq_report.get("quality", {}).get("verdict_breakdown"))
    assert breakdown is not None, "verdict_breakdown not in report"
    # All three categories must be present (even if zero)
    for cat in ("local_better", "cloud_better", "equivalent"):
        alt_cat = cat.replace("_", "-")
        assert cat in breakdown or alt_cat in breakdown, (
            f"'{cat}' missing from verdict_breakdown"
        )


# ═══════════════════════════════════════════════════════════════════════════
# LANGUAGE BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def test_language_breakdown(eq_report):
    by_lang = eq_report.get("by_language", {})
    assert "native_english" in by_lang or "en" in by_lang, (
        "Missing native_english breakdown"
    )
    assert "translated_german" in by_lang or "de" in by_lang, (
        "Missing translated_german breakdown"
    )


# ═══════════════════════════════════════════════════════════════════════════
# JUDGE QUALITY — reasons exist, not that they favor anyone
# ═══════════════════════════════════════════════════════════════════════════

def test_judge_gives_reasons(judge_output):
    for i, j in enumerate(judge_output):
        reason = j.get("reason", "")
        assert len(reason) > 20, (
            f"Judgment {i}: reason too short ({len(reason)} chars)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# ESCALATION REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def test_escalation_rate_reported(eq_report):
    """Escalation rate must be in the report (it's part of system performance)."""
    rate = eq_report.get("escalation_rate",
                         eq_report.get("escalation", {}).get("rate"))
    assert rate is not None, "Escalation rate not recorded"
    assert 0 <= rate <= 100, f"Escalation rate={rate} outside [0,100]"


# ═══════════════════════════════════════════════════════════════════════════
# PER-CLUSTER BREAKDOWN — only for clusters with sufficient n
# ═══════════════════════════════════════════════════════════════════════════

def test_per_cluster_breakdown_present(eq_report):
    """Per-cluster results must exist.  Clusters with < 10 tickets must be
    marked insufficient, not silently included with tiny n."""
    by_cluster = eq_report.get("by_cluster", {})
    assert len(by_cluster) > 0, "No per-cluster breakdown in report"
    for cid, data in by_cluster.items():
        n = data.get("n", data.get("count", 0))
        if n < 10:
            note = json.dumps(data).lower()
            assert "insufficient" in note, (
                f"Cluster {cid} has n={n} < 10 but not marked insufficient"
            )
