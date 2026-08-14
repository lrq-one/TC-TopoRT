#!/usr/bin/env python3
"""Regression tests for the final independent-development hard filter."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts/filtering/run_candidate_filtering.py"
SPEC = importlib.util.spec_from_file_location("run_candidate_filtering", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FinalFilteringRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            [
                {"query_id": "q1", "candidate_id": "rank4_far", "candidate_rank": 4, "experimental_rt": 100.0, "predicted_rt": 300.0, "is_true": False},
                {"query_id": "q1", "candidate_id": "rank2_true", "candidate_rank": 2, "experimental_rt": 100.0, "predicted_rt": 105.0, "is_true": True},
                {"query_id": "q1", "candidate_id": "rank1_far", "candidate_rank": 1, "experimental_rt": 100.0, "predicted_rt": 250.0, "is_true": False},
                {"query_id": "q1", "candidate_id": "rank3_missing", "candidate_rank": 3, "experimental_rt": 100.0, "predicted_rt": None, "is_true": False},
            ]
        )
        self.candidates, self.queries, self.summary = MODULE.filter_candidates(
            self.frame, threshold_seconds=10.0
        )

    def retained(self, candidate_id: str) -> bool:
        row = self.candidates[self.candidates["candidate_id"].eq(candidate_id)].iloc[0]
        return bool(row["retained"])

    def test_prediction_inside_threshold_is_retained(self) -> None:
        self.assertTrue(self.retained("rank2_true"))

    def test_prediction_exactly_at_threshold_is_retained(self) -> None:
        frame = self.frame.iloc[[1]].copy()
        frame.loc[:, "predicted_rt"] = 110.0
        candidates, _, _ = MODULE.filter_candidates(frame, threshold_seconds=10.0)
        self.assertTrue(bool(candidates.iloc[0]["retained"]))

    def test_prediction_outside_threshold_is_removed(self) -> None:
        self.assertFalse(self.retained("rank4_far"))

    def test_missing_prediction_is_retained(self) -> None:
        self.assertTrue(self.retained("rank3_missing"))

    def test_original_rank_does_not_override_threshold_failure(self) -> None:
        self.assertFalse(self.retained("rank1_far"))

    def test_survivors_preserve_original_order(self) -> None:
        survivors = self.candidates.loc[self.candidates["retained"], "candidate_id"].tolist()
        self.assertEqual(survivors, ["rank2_true", "rank3_missing"])
        self.assertEqual(int(self.queries.iloc[0]["true_rank_after_filtering"]), 1)

    def test_results_do_not_depend_on_paper_reference_counts(self) -> None:
        self.assertEqual(self.summary["queries"], 1)
        self.assertEqual(self.summary["true_retained_count"], 1)
        self.assertEqual(self.summary["false_negatives"], 0)


if __name__ == "__main__":
    unittest.main()
