"""Outcome-blind input QC logic with no model or image-library dependency."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

ARTIFACT_FIELDS = (
    "visible_non_background_object",
    "donor_removal_or_fill_trace",
    "abnormal_compositing_boundary",
    "other_arm_identity_cue",
)
SEMANTIC_LABELS = {"AQUATIC_CONTEXT", "NON_AQUATIC_CONTEXT", "AMBIGUOUS"}


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Return unweighted Cohen's kappa for two equally sized non-empty label lists."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("KAPPA_REQUIRES_EQUAL_NONEMPTY_LABEL_LISTS")
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    counts_a, counts_b = Counter(labels_a), Counter(labels_b)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in set(counts_a) | set(counts_b))
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _indexed(bundle: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    reviewer = bundle.get("reviewer_id")
    if not reviewer:
        raise ValueError("REVIEWER_ID_REQUIRED")
    declaration = bundle.get("independence_declaration", {})
    required = {
        "worked_independently": True,
        "discussed_before_seal": False,
        "model_outcomes_visible": False,
        "source_or_arm_identity_visible": False,
    }
    if any(declaration.get(key) != value for key, value in required.items()):
        raise ValueError("INDEPENDENCE_DECLARATION_FAILED")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in bundle.get("reviews", []):
        card_id = row.get("card_id")
        if not card_id or card_id in rows:
            raise ValueError("CARD_IDS_MUST_BE_NONEMPTY_AND_UNIQUE")
        if row.get("semantic_label") not in SEMANTIC_LABELS:
            raise ValueError("UNKNOWN_SEMANTIC_LABEL")
        if any(not isinstance(row.get(field), bool) for field in ARTIFACT_FIELDS):
            raise ValueError("ARTIFACT_FLAGS_MUST_BE_BOOLEAN")
        rows[card_id] = row
    if not rows:
        raise ValueError("REVIEW_BUNDLE_EMPTY")
    return rows


def evaluate_blind_reviews(
    bundle_a: Mapping[str, Any],
    bundle_b: Mapping[str, Any],
    *,
    minimum_raw_agreement: float = 0.95,
    minimum_kappa: float = 0.80,
) -> dict[str, Any]:
    """Apply two-reviewer reliability and union-reject rules.

    An ambiguous label, label disagreement, or any artifact flag from either
    reviewer rejects a card. Reliability is computed before exclusions.
    """
    if bundle_a.get("reviewer_id") == bundle_b.get("reviewer_id"):
        raise ValueError("REVIEWERS_MUST_BE_DISTINCT")
    if bundle_a.get("review_card_manifest_sha256") != bundle_b.get("review_card_manifest_sha256"):
        raise ValueError("REVIEW_CARD_MANIFEST_BINDING_MISMATCH")
    rows_a, rows_b = _indexed(bundle_a), _indexed(bundle_b)
    if set(rows_a) != set(rows_b):
        raise ValueError("REVIEW_CARD_SETS_DIFFER")
    card_ids = sorted(rows_a)
    labels_a = [str(rows_a[c]["semantic_label"]) for c in card_ids]
    labels_b = [str(rows_b[c]["semantic_label"]) for c in card_ids]
    agreement = sum(a == b for a, b in zip(labels_a, labels_b)) / len(card_ids)
    kappa = cohen_kappa(labels_a, labels_b)
    accepted: list[str] = []
    rejected: list[dict[str, Any]] = []
    for card_id in card_ids:
        a, b = rows_a[card_id], rows_b[card_id]
        reasons: list[str] = []
        if a["semantic_label"] == "AMBIGUOUS" or b["semantic_label"] == "AMBIGUOUS":
            reasons.append("AMBIGUOUS_LABEL")
        if a["semantic_label"] != b["semantic_label"]:
            reasons.append("SEMANTIC_DISAGREEMENT")
        for field in ARTIFACT_FIELDS:
            if a[field] or b[field]:
                reasons.append(field.upper())
        if reasons:
            rejected.append({"card_id": card_id, "reasons": sorted(set(reasons))})
        else:
            accepted.append(card_id)
    reliability_passed = agreement >= minimum_raw_agreement and kappa >= minimum_kappa
    return {
        "reviewed_card_count": len(card_ids),
        "raw_semantic_agreement": agreement,
        "cohen_kappa": kappa,
        "minimum_raw_semantic_agreement": minimum_raw_agreement,
        "minimum_cohen_kappa": minimum_kappa,
        "reliability_passed": reliability_passed,
        "accepted_card_ids": accepted if reliability_passed else [],
        "rejected_cards": rejected,
        "status": "PASS_BLIND_REVIEW" if reliability_passed and not rejected else "FAIL_BLIND_REVIEW",
    }


def all_pairs_match(records: Iterable[Mapping[str, Any]], *, expected_count: int = 8) -> bool:
    rows = list(records)
    return len(rows) == expected_count and all(row.get("matched") is True for row in rows)
