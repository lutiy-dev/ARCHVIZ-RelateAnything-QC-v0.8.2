# ARCHVIZ RelateAnything QC v0.8.2

Rejection-aware BEFORE vs AFTER geometry QC for AI-assisted architectural visualization in ComfyUI.

## v0.8.2
- normalized weighted cost (0..1 component penalties, divided by active weight sum)
- rejection-aware Hungarian assignment with dummy unmatched slots
- explicit `accept_match_cost` and `dummy_unmatched_cost`
- `Geometry = NOT_EVALUATED` when reliable matches are insufficient
- candidate/accepted cost diagnostics and reliable match fraction
- visual overlay remains GREEN matched, YELLOW drifted, RED missing, BLUE added
- RelateAnything semantic delta remains advisory only

## Defaults
`accept_match_cost=0.55`, `dummy_unmatched_cost=0.35`, `min_reliable_match_fraction=0.50`.

Clone directly into ComfyUI/custom_nodes and search for:
`RA · SMART BEFORE vs AFTER QC · v0.8.2`.


## v0.9 extension · Stable Paired Region Sampling

The next LAB step is now included in this repository.

New node:

`RA · STABLE PAIRED REGION SAMPLER · v0.9`

Purpose:
- let `RARegionsV06` collect up to 32 candidate regions on both BEFORE and AFTER;
- sample a stable paired subset of 16 regions instead of relying on the first 16 detections;
- prefer shared spatial grid cells;
- supplement with nearest normalized-center pairs when required;
- output BEFORE/AFTER sampled regions in deterministic top-to-bottom / left-to-right order.

New workflow:

`workflows/ARCHVIZ_RELATEANYTHING_STABLE_SAMPLING_QC_v009.json`

Baseline sampler settings:
- candidate regions from RARegions: 32
- sample_count: 16
- grid_rows: 4
- grid_cols: 4
- supplement_center_tolerance: 0.12

The sampler is a stabilization layer, not geometry truth. Final PASS/WARN/FAIL still comes from the v0.8.2 rejection-aware QC node.
