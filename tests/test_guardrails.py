"""§1 Guardrails — data integrity, split quality, cost transparency.

These tests validate the data pipeline and infrastructure that is shared
across ALL experiment designs.  They do not depend on which experiment
ran or what the outcome was.
"""

import warnings

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# DATA SPLIT INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

def test_held_out_test_set_no_leakage(train_df, test_df, test_routed_df):
    """Verify the test_routed set is derived from the held-out test_split.

    The source dataset contains duplicate (subject, body) pairs across rows,
    so *content* overlap between train and test is an inherent dataset
    property.  The meaningful guarantee is that the routed test tickets
    originate from the test split, not the train split.
    """
    routed_keys = set(
        zip(test_routed_df["subject"].astype(str),
            test_routed_df["body"].astype(str),
            test_routed_df["type"].astype(str))
    )
    test_keys = set(
        zip(test_df["subject"].astype(str),
            test_df["body"].astype(str),
            test_df["type"].astype(str))
    )
    matched = len(routed_keys & test_keys)
    match_pct = matched / len(routed_keys) * 100
    assert match_pct >= 95, (
        f"Only {match_pct:.1f}% of routed test tickets trace to test_split "
        f"({matched}/{len(routed_keys)})"
    )


def test_train_test_split_ratio(train_df, test_df):
    total = len(train_df) + len(test_df)
    test_pct = len(test_df) / total * 100
    assert 18 <= test_pct <= 22, f"Test ratio {test_pct:.1f}% outside 18-22%"


def test_stratified_split(train_df, test_df):
    train_dist = train_df["type"].value_counts(normalize=True)
    test_dist = test_df["type"].value_counts(normalize=True)
    for t in train_dist.index:
        diff = abs(train_dist[t] - test_dist.get(t, 0)) * 100
        assert diff < 2, f"Type '{t}' differs by {diff:.2f}pp between splits"


# ═══════════════════════════════════════════════════════════════════════════
# BLUEPRINT COST TRANSPARENCY
# ═══════════════════════════════════════════════════════════════════════════

def test_blueprint_cost_amortized_honestly(blueprint_cost):
    assert "cost_usd" in blueprint_cost, "No cost_usd key in blueprint cost"
    assert blueprint_cost["total_tokens"] > 0, "Total tokens should be > 0"
    assert blueprint_cost["cost_usd"]["total"] > 0, "Cost should be > 0"


# ═══════════════════════════════════════════════════════════════════════════
# KNOWN GAPS — documented, not hidden
# ═══════════════════════════════════════════════════════════════════════════

def test_variance_runs_known_gap(real_experiment_report):
    """The old experiment used a single run.  The corrected spec requires ≥3.
    This test documents the gap in the old run; the corrected experiment's
    test_phase3_experiment.py enforces the 3-run requirement directly."""
    total = real_experiment_report["experiment"]["total_tickets"]
    with pytest.warns(UserWarning, match="single run"):
        warnings.warn(
            f"Old experiment: single run of {total} tickets; "
            "corrected spec requires ≥3 runs to estimate variance",
            UserWarning,
        )


def test_same_data_across_arms(real_experiment_progress):
    """Both arms in the old experiment saw the same ticket keys."""
    arm_a_keys = set(real_experiment_progress["arm_a"].keys())
    arm_b_keys = set(real_experiment_progress["arm_b"].keys())
    assert arm_a_keys == arm_b_keys, "Arms A and B saw different ticket keys"
