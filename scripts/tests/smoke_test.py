#!/usr/bin/env python3
"""Dataset-free smoke test for the public configuration and filtering path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
FILTER = ROOT / "scripts/filtering/run_candidate_filtering.py"
SPEC = importlib.util.spec_from_file_location("run_candidate_filtering", FILTER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def main() -> int:
    with (ROOT / "configs/candidate_filtering.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    threshold = config["development"]["metabobase"]["methods"]["tc_toport"]["threshold_seconds"]
    example = pd.DataFrame(
        [
            {"query_id": "synthetic_query", "candidate_id": "benzene_like_true", "candidate_rank": 1, "experimental_rt": 500.0, "predicted_rt": 510.0, "is_true": True},
            {"query_id": "synthetic_query", "candidate_id": "distant_candidate", "candidate_rank": 2, "experimental_rt": 500.0, "predicted_rt": 900.0, "is_true": False},
            {"query_id": "synthetic_query", "candidate_id": "unpredicted_candidate", "candidate_rank": 3, "experimental_rt": 500.0, "predicted_rt": None, "is_true": False},
        ]
    )
    candidates, queries, summary = MODULE.filter_candidates(example, float(threshold))
    assert candidates["retained"].tolist() == [True, False, True]
    assert queries.iloc[0]["true_retained"]
    assert summary["queries"] == 1 and summary["candidate_records_after"] == 2
    assert (ROOT / "gwn/mp").is_dir()
    assert (ROOT / "gwn/net").is_dir()
    assert (ROOT / "gwn/train_oof_dualview_stack.py").is_file()
    print("TC-TopoRT dataset-free smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
