"""Synthetic control tests for ARCHVIZ QC v0.9.1.

Run from the repository root with the same Python environment that loads the
custom node:

    python tests/test_qc_v091_synthetic.py

These tests intentionally bypass SAM3 and feed known RA_REGIONS rectangles
directly into the diagnostic QC node.
"""
import copy
import importlib.util
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = "archviz_relateanything_qc_v091_testpkg"
spec = importlib.util.spec_from_file_location(
    PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
mod = importlib.util.module_from_spec(spec)
sys.modules[PKG] = mod
spec.loader.exec_module(mod)

Node = mod.RACompareDiagnosticQCV091


def regions(boxes, ids=None, w=1000, h=1000):
    if ids is None:
        ids = list(range(len(boxes)))
    return {
        "boxes": [list(map(float, b)) for b in boxes],
        "source_indices": [int(x) for x in ids],
        "width": int(w),
        "height": int(h),
        "count": len(boxes),
    }


IMG = torch.zeros((1, 1000, 1000, 3), dtype=torch.float32)

BASE = [
    [100, 100, 180, 220],
    [300, 100, 380, 220],
    [500, 100, 580, 220],
    [100, 400, 180, 520],
    [300, 400, 380, 520],
    [500, 400, 580, 520],
]


def run(before, after, **overrides):
    args = dict(
        before_image=IMG,
        after_image=IMG,
        before_regions=before,
        after_regions=after,
        row_tolerance=0.02,
        column_tolerance=0.02,
        weight_center=4.0,
        weight_iou=1.5,
        weight_size=1.0,
        weight_grid=1.5,
        weight_neighborhood=1.0,
        accept_match_cost=0.35,
        dummy_unmatched_cost=0.17,
        ambiguity_margin_threshold=0.03,
        warn_center_drift_pct=1.0,
        warn_size_drift_pct=8.0,
        fail_unmatched_fraction=0.20,
        min_geometry_match_fraction=0.50,
        order_tolerance=0.01,
        detection_limit=32,
        display_sample_count=4,
        display_mode="problems",
    )
    args.update(overrides)
    out = Node().compare(**args)
    overlay, report, payload = out["result"]
    return report, json.loads(payload)


def decision_view(payload):
    p = copy.deepcopy(payload)
    p.pop("display", None)
    return p


def main():
    # 1. Identical sets.
    report, p = run(regions(BASE), regions(BASE))
    assert p["matching"]["accepted_pairs"] == len(BASE)
    assert not p["matching"]["unmatched_before_candidate_indices"]
    assert not p["matching"]["unmatched_after_candidate_indices"]
    assert p["geometry"]["center_drift"]["max_pct_diag"] == 0.0
    assert p["geometry"]["size_drift"]["max_pct"] == 0.0

    # 2. List order shuffled; decision-level geometry must remain unchanged.
    order = [5, 2, 0, 4, 1, 3]
    shuffled = [BASE[i] for i in order]
    _, ps = run(regions(BASE, list(range(6))), regions(shuffled, order))
    assert ps["matching"]["accepted_pairs"] == len(BASE)
    assert ps["geometry"]["center_drift"]["max_pct_diag"] == 0.0
    assert ps["geometry"]["size_drift"]["max_pct"] == 0.0

    # 3. Remove one AFTER rectangle -> one unmatched BEFORE candidate.
    _, pr = run(regions(BASE), regions(BASE[:-1]))
    assert len(pr["matching"]["unmatched_before_candidate_indices"]) == 1
    assert len(pr["matching"]["unmatched_after_candidate_indices"]) == 0

    # 4. Add one AFTER rectangle -> one unmatched AFTER candidate.
    added = BASE + [[750, 700, 830, 820]]
    _, pa = run(regions(BASE), regions(added))
    assert len(pa["matching"]["unmatched_before_candidate_indices"]) == 0
    assert len(pa["matching"]["unmatched_after_candidate_indices"]) == 1

    # 5. Move one rectangle -> drift measured or the pair becomes unmatched.
    moved = copy.deepcopy(BASE)
    moved[0] = [150, 100, 230, 220]
    _, pm = run(regions(BASE), regions(moved))
    drift = pm["geometry"]["center_drift"]["max_pct_diag"]
    unmatched = len(pm["matching"]["unmatched_before_candidate_indices"]) + len(pm["matching"]["unmatched_after_candidate_indices"])
    assert (drift is not None and drift > 0.0) or unmatched > 0

    # 6. Empty sets -> insufficient data, geometry not evaluated.
    _, pe = run(regions([]), regions([]))
    assert pe["overall_status"] == "INSUFFICIENT_DATA"
    assert pe["data_sufficiency_status"] == "INSUFFICIENT_DATA"
    assert pe["geometry_status"] == "NOT_EVALUATED"
    assert pe["matching"]["accepted_match_fraction"] is None

    # 7. Display sampling must not change the full report or decision metrics.
    r1, p1 = run(regions(BASE), regions(BASE), display_sample_count=2, display_mode="problems")
    r2, p2 = run(regions(BASE), regions(BASE), display_sample_count=6, display_mode="overview")
    assert r1 == r2
    assert decision_view(p1) == decision_view(p2)

    print("ARCHVIZ QC v0.9.1 synthetic controls: PASS")


if __name__ == "__main__":
    main()
