# ARCHVIZ RelateAnything QC v0.9.1 Diagnostic

Full-set BEFORE vs AFTER geometry QC for AI-assisted architectural visualization in ComfyUI.

v0.9.1 fixes the main architectural limitation found during v0.9 validation: **display sampling no longer determines the completeness of QC**. Full region sets now enter matching first; sampling is display-only.

## Status

**LAB / DIAGNOSTIC**

Verified direction:
- full BEFORE and AFTER region sets enter QC directly;
- sampler is no longer on the decision path;
- empty sets report `INSUFFICIENT_DATA`, not Detection PASS;
- unmatched regions are reported as **candidates**, not confirmed missing/added architecture;
- `Reliable match fraction` is replaced by `Accepted match fraction`;
- BEFORE and AFTER coverage are shown separately;
- ambiguous accepted pairs are flagged and excluded from geometry verdicts;
- reaching the configured detection limit raises `POSSIBLE_TRUNCATION`;
- overlay preserves source IDs such as `B03 ↔ A07`;
- display sampling can show problem pairs or a spatial overview without changing the full report.

## v0.9.1 pipeline

```text
BEFORE / geometry truth
        ↓
      detector / SAM3
        ↓
RARegions full set ──────────────┐
                                 ├→ FULL-SET MATCHING
RARegions full set ──────────────┘       │
        ↑                                ├→ data sufficiency
      detector / SAM3                    ├→ accepted / rejected pairs
        ↑                                ├→ ambiguity analysis
AFTER / AI result                        ├→ geometry on unambiguous pairs
                                         └→ full JSON/report
                                                  │
                                                  └→ display sample only
                                                      problems / overview / all
```

The **full report must remain identical** when only the display sample count or display mode changes.

## Main node

### RA · DIAGNOSTIC FULL-SET QC · v0.9.1

Key outputs:
- full region counts entering QC;
- possible truncation at the configured detector/region limit;
- accepted pairs;
- rejected-by-threshold pairs;
- ambiguous accepted pairs;
- unmatched BEFORE candidates;
- unmatched AFTER candidates;
- accepted match fraction;
- separate BEFORE / AFTER coverage;
- geometry-eligible pair count and fraction;
- center / size drift;
- row / column structure changes;
- JSON diagnostics with source IDs.

### Status semantics

**Data sufficiency**
- `PASS` — both full sets are non-empty and no configured limit was reached;
- `WARN` — possible truncation because a full set reached the configured limit;
- `INSUFFICIENT_DATA` — one or both full region sets are empty.

**Matching**
- `PASS` — no unmatched candidates, rejected pairs, or ambiguity flags;
- `WARN` — unmatched candidates, threshold rejections, or ambiguous accepted pairs exist;
- `FAIL` — unmatched fraction exceeds the configured fail threshold;
- `NOT_EVALUATED` — insufficient data.

**Geometry**
- evaluated only on accepted, non-ambiguous pairs;
- `NOT_EVALUATED` when eligible coverage is insufficient or fewer than two unambiguous pairs exist.

## Unmatched means candidate, not proof

`Unmatched BEFORE` and `Unmatched AFTER` are **diagnostic candidates only**.

An unmatched region can mean:
- a real architectural deletion/addition;
- detector instability caused by light/material changes;
- split/merged detections;
- a rejected or ambiguous match.

Confirmed misses / false positives require a detector benchmark against manual GT.

## Overlay

- GREEN = accepted, unambiguous, geometry within warning limits
- YELLOW = accepted, unambiguous, geometry drift
- ORANGE = accepted but ambiguous correspondence; excluded from geometry verdict
- RED = unmatched BEFORE candidate
- BLUE = unmatched AFTER candidate

Labels preserve source IDs: `B03`, `A07`, etc.

## Diagnostic workflow

Import:

`workflows/ARCHVIZ_RELATEANYTHING_DIAGNOSTIC_QC_v0091.json`

It includes:
- SAM3 BEFORE visualization;
- SAM3 AFTER visualization;
- RARegions debug for BEFORE;
- RARegions debug for AFTER;
- full-set v0.9.1 QC;
- QC overlay;
- QC report.

The legacy v0.9 stable-sampler workflow remains in the repository for comparison only:

`workflows/ARCHVIZ_RELATEANYTHING_STABLE_SAMPLING_QC_v009.json`

## Required control tests before detector benchmark

Use synthetic / known rectangles and verify:

1. identical sets → all pairs accepted, zero geometry change;
2. list order shuffled → same QC result;
3. one rectangle removed → one unmatched BEFORE candidate;
4. one rectangle added → one unmatched AFTER candidate;
5. one rectangle moved → measured drift or explicit rejection;
6. empty sets → `INSUFFICIENT_DATA` and Geometry `NOT_EVALUATED`;
7. display mode/count changed → **full report and JSON decision metrics remain unchanged**.

## Detector benchmark protocol

Candidate detectors:
- SAM3 direct
- GroundingDINO
- Florence-2
- YOLO-World

Recommended GT rule for windows:
- one GT object = one complete architectural window opening;
- do not split panes into separate GT windows;
- define handling for partially occluded / cropped windows before evaluation.

Recommended matching protocol:
- one detection ↔ at most one GT window;
- one GT window ↔ at most one detection;
- first benchmark threshold: IoU ≥ 0.5;
- duplicates count as false positives, with duplicates also reported as a subtype.

Report:
- GT count;
- detected count;
- TP / FP / FN;
- duplicates;
- precision;
- recall;
- processing time;
- small-window performance;
- BEFORE/AFTER stability under lighting/material changes with unchanged geometry.

Do not force the same numeric confidence threshold across different detector families. Tune each detector on a small development set, then freeze its settings for the benchmark.

## Baseline v0.9.1 settings

- row_tolerance: 0.02
- column_tolerance: 0.02
- weight_center: 4.0
- weight_iou: 1.5
- weight_size: 1.0
- weight_grid: 1.5
- weight_neighborhood: 1.0
- accept_match_cost: 0.35
- dummy_unmatched_cost: 0.17
- ambiguity_margin_threshold: 0.03
- warn_center_drift_pct: 1.0
- warn_size_drift_pct: 8.0
- fail_unmatched_fraction: 0.20
- min_geometry_match_fraction: 0.50
- detection_limit: 32
- display_sample_count: 8
- display_mode: problems

## Installation

Home portable example:

```bat
git clone https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.9.git "Q:\AI_ArchViz\ComfyUI_windows_portable\ComfyUI\custom_nodes\ARCHVIZ-RelateAnything-QC-v0.9"
```

Work / Comfy Desktop example:

```bat
git clone https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.9.git "C:\ComfyUI-Installs\ARCHVIZ_LAB\ComfyUI\custom_nodes\ARCHVIZ-RelateAnything-QC-v0.9"
```

Existing installation:

```bat
cd /d "C:\ComfyUI-Installs\ARCHVIZ_LAB\ComfyUI\custom_nodes\ARCHVIZ-RelateAnything-QC-v0.9"
git pull
```

Restart ComfyUI after pulling.

Keep the existing v0.6 ONNX / RARegions package installed.

## Important limitations

- Same camera / crop / composition is required for geometry comparison.
- Detector quality is upstream of QC and remains the current primary research target.
- Full-set matching improves diagnostic honesty but does not make unmatched regions ground truth.
- Repetitive facades can produce ambiguous correspondence; ambiguous pairs are intentionally excluded from geometry verdicts.
- v0.9.1 is a diagnostic LAB tool, not an autonomous final authority.
- Do not update the main Torch / Transformers stack for this node.

## Repository history

- v0.6 — RelateAnything ONNX integration
- v0.7 — deterministic BEFORE vs AFTER bbox QC
- v0.8.2 — normalized rejection-aware Smart Matcher
- v0.9 — stable paired-region sampling experiment
- **v0.9.1 — full-set diagnostic QC; sampler removed from decision path**

## Links

### Current project
- **ARCHVIZ RelateAnything QC repository**  
  https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.9
- **v0.9.1 diagnostic workflow**  
  https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.9/blob/main/workflows/ARCHVIZ_RELATEANYTHING_DIAGNOSTIC_QC_v0091.json
- **Legacy v0.9 workflow**  
  https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.9/blob/main/workflows/ARCHVIZ_RELATEANYTHING_STABLE_SAMPLING_QC_v009.json
- **Stable sampler source (legacy diagnostic reference)**  
  https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.9/blob/main/stable_sampler_v09.py

### Required / related components
- **RelateAnything ONNX / RARegions v0.6**  
  https://github.com/lutiy-dev/ComfyUI-RelateAnything-v0.6
- **RelateAnything model: maelic/relsgg-vits16plus**  
  https://huggingface.co/maelic/relsgg-vits16plus
- **SAM 3 upstream**  
  https://github.com/facebookresearch/sam3
- **ComfyUI**  
  https://github.com/comfyanonymous/ComfyUI
- **SciPy Hungarian assignment**  
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
- **GroundingDINO**  
  https://github.com/IDEA-Research/GroundingDINO
- **Florence-2 Large FT**  
  https://huggingface.co/microsoft/Florence-2-large-ft

### Previous QC versions
- **v0.7**  
  https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.7
- **v0.8.2**  
  https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.8.2
