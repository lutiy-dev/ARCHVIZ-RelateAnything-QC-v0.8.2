# Changelog

## v0.9
- Added stable paired-region sampling before QC.
- Candidate pool increased to 32 detections/regions per branch.
- Stable sample defaults to 16 paired regions on a 4×4 spatial grid.
- Added nearest-center supplementation for underfilled shared grid cells.
- Retained the verified v0.8.2 rejection-aware Smart Matcher as the QC core.
- Repository renamed and cleaned as a standalone v0.9 package.

## v0.8.2
- Normalized weighted cost.
- Rejection-aware Hungarian matching.
- Dummy unmatched assignments.
- Reliable-match fraction and NOT_EVALUATED geometry state.
- Cost diagnostics and visual overlay.
