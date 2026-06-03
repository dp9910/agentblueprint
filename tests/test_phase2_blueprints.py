"""§4 Phase 2 — Blueprint schema, review gate pass rates."""

import pytest

EXPECTED_CLUSTERS = {"0", "2", "4", "7", "9"}
REQUIRED_FIELDS = {"intent", "scope", "steps", "minimal_context", "exit_conditions"}
STEP_FIELDS = {"step_number", "name", "input_needed", "action", "decision_output"}


# ── 1. Blueprints exist for all 5 top clusters ──────────────────────────

def test_blueprints_exist_for_top_clusters(blueprints):
    assert set(blueprints.keys()) == EXPECTED_CLUSTERS


# ── 2. Exactly 5 blueprints ──────────────────────────────────────────────

def test_blueprint_count(blueprints):
    assert len(blueprints) == 5


# ── 3. Each blueprint has required top-level fields ──────────────────────

def test_blueprint_schema_completeness(blueprints):
    for cid, bp in blueprints.items():
        missing = REQUIRED_FIELDS - set(bp.keys())
        assert not missing, f"Cluster {cid} missing fields: {missing}"


# ── 4. Each step has required sub-fields ─────────────────────────────────

def test_steps_have_required_fields(blueprints):
    for cid, bp in blueprints.items():
        for step in bp["steps"]:
            missing = STEP_FIELDS - set(step.keys())
            assert not missing, (
                f"Cluster {cid}, step {step.get('step_number')}: missing {missing}"
            )


# ── 5. Each blueprint has ≥ 1 exit condition ─────────────────────────────

def test_exit_conditions_present(blueprints):
    for cid, bp in blueprints.items():
        assert len(bp["exit_conditions"]) >= 1, (
            f"Cluster {cid} has no exit conditions"
        )


# ── 6. Each blueprint has ≥ 3 steps ──────────────────────────────────────

def test_minimum_step_count(blueprints):
    for cid, bp in blueprints.items():
        assert len(bp["steps"]) >= 3, (
            f"Cluster {cid} has only {len(bp['steps'])} steps"
        )


# ── 7. Generation cost recorded: total_tokens > 0, cost > 0 ─────────────

def test_generation_cost_recorded(blueprint_cost):
    assert blueprint_cost["total_tokens"] > 0
    assert blueprint_cost["cost_usd"]["total"] > 0


# ── 8. Generation cost < $1.00 ───────────────────────────────────────────

def test_generation_cost_reasonable(blueprint_cost):
    assert blueprint_cost["cost_usd"]["total"] < 1.0, (
        f"Blueprint generation cost ${blueprint_cost['cost_usd']['total']:.4f} >= $1.00"
    )


# ── 9. Review gate completed ─────────────────────────────────────────────

def test_review_gate_completed(blueprint_review):
    assert blueprint_review["status"] == "complete"


# ── 10. In-cluster fit rate ≥ 80% ────────────────────────────────────────

def _parse_fraction(s: str) -> float:
    num, den = s.split("/")
    return int(num) / int(den) * 100


def test_in_cluster_fit_rate(blueprint_review):
    for cid, review in blueprint_review["reviews"].items():
        rate = _parse_fraction(review["summary"]["in_cluster_fits"])
        assert rate >= 80, f"Cluster {cid}: in-cluster fit {rate}% < 80%"


# ── 11. Out-of-cluster rejection rate ≥ 90% ──────────────────────────────

def test_out_of_cluster_rejection_rate(blueprint_review):
    for cid, review in blueprint_review["reviews"].items():
        rate = _parse_fraction(review["summary"]["out_of_cluster_rejected"])
        assert rate >= 90, f"Cluster {cid}: out-of-cluster rejection {rate}% < 90%"
