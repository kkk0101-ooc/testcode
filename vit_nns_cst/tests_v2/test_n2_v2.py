import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nns_cst_n2_v2.input_gate import all_pairs_match, cohen_kappa, evaluate_blind_reviews


def bundle(reviewer, labels, flagged=()):
    rows = []
    for index, label in enumerate(labels):
        rows.append({
            "card_id": f"card_{index:02d}", "semantic_label": label,
            "visible_non_background_object": index in flagged,
            "donor_removal_or_fill_trace": False, "abnormal_compositing_boundary": False,
            "other_arm_identity_cue": False,
        })
    return {
        "reviewer_id": reviewer, "review_card_manifest_sha256": "a" * 64,
        "independence_declaration": {
            "worked_independently": True, "discussed_before_seal": False,
            "model_outcomes_visible": False, "source_or_arm_identity_visible": False,
        },
        "reviews": rows,
    }


class N2V2InputGateTests(unittest.TestCase):
    def test_perfect_kappa(self):
        labels = ["AQUATIC_CONTEXT", "NON_AQUATIC_CONTEXT"] * 10
        self.assertEqual(cohen_kappa(labels, labels), 1.0)

    def test_distinct_reviewers_required(self):
        labels = ["AQUATIC_CONTEXT", "NON_AQUATIC_CONTEXT"]
        with self.assertRaisesRegex(ValueError, "REVIEWERS_MUST_BE_DISTINCT"):
            evaluate_blind_reviews(bundle("same", labels), bundle("same", labels))

    def test_any_reviewer_artifact_flag_rejects(self):
        labels = ["AQUATIC_CONTEXT", "NON_AQUATIC_CONTEXT"] * 10
        result = evaluate_blind_reviews(bundle("a", labels), bundle("b", labels, flagged={3}))
        self.assertEqual(result["status"], "FAIL_BLIND_REVIEW")
        self.assertEqual(result["rejected_cards"][0]["card_id"], "card_03")

    def test_ambiguous_rejects_even_with_agreement(self):
        labels = ["AQUATIC_CONTEXT", "NON_AQUATIC_CONTEXT"] * 10 + ["AMBIGUOUS"]
        result = evaluate_blind_reviews(bundle("a", labels), bundle("b", labels))
        self.assertEqual(result["status"], "FAIL_BLIND_REVIEW")
        self.assertIn("AMBIGUOUS_LABEL", result["rejected_cards"][0]["reasons"])

    def test_reliability_threshold_fails_closed(self):
        labels_a = ["AQUATIC_CONTEXT", "NON_AQUATIC_CONTEXT"] * 10
        labels_b = list(labels_a); labels_b[0] = "NON_AQUATIC_CONTEXT"; labels_b[1] = "AQUATIC_CONTEXT"
        result = evaluate_blind_reviews(bundle("a", labels_a), bundle("b", labels_b))
        self.assertFalse(result["reliability_passed"]); self.assertEqual(result["accepted_card_ids"], [])

    def test_eight_of_eight_matching_required(self):
        self.assertTrue(all_pairs_match([{"matched": True}] * 8))
        self.assertFalse(all_pairs_match([{"matched": True}] * 7 + [{"matched": False}]))
        self.assertFalse(all_pairs_match([{"matched": True}] * 7))


if __name__ == "__main__":
    unittest.main()
