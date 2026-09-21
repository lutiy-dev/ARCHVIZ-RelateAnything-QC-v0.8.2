# ARCHVIZ RelateAnything QC v0.9

Stable paired-region geometry QC for AI-assisted architectural visualization in ComfyUI.

v0.9 builds on the verified rejection-aware Smart Matcher from v0.8.2 and adds a **stable paired sampling layer** so BEFORE and AFTER are compared on the same spatially distributed facade sample instead of simply using the first detections returned by SAM3.

## Status

**VERIFIED LAB**

Validated behavior so far:
- same image → same image: PASS, 16/16 reliable matches, zero drift;
- multiple real AI AFTER tests: meaningful WARN/FAIL results;
- rejection-aware Hungarian matching avoids forcing obviously bad pairs;
- v0.9 adds stable paired sampling to reduce detector-selection variance on dense facades.

## Pipeline

```text
BEFORE / geometry truth
        ↓
      SAM3
        ↓
RARegions candidates (up to 32)
        ↘
         Stable Paired Region Sampler v0.9
        ↗
RARegions candidates (up to 32)
        ↑
      SAM3
        ↑
AFTER / AI result

Stable paired subset
        ↓
Rejection-aware Smart QC core (v0.8.2)
        ↓
Overlay + PASS / WARN / FAIL + JSON
```

## Nodes

### RA · STABLE PAIRED REGION SAMPLER · v0.9
Purpose:
- collect a larger candidate pool on BEFORE and AFTER;
- prefer shared spatial grid cells;
- select one representative pair per shared cell;
- supplement missing sample slots with nearest normalized-center pairs;
- output deterministic top-to-bottom / left-to-right paired samples.

Default sampler settings:
- candidate detections: 32
- sample_count: 16
- grid_rows: 4
- grid_cols: 4
- supplement_center_tolerance: 0.12

### RA · SMART BEFORE vs AFTER QC · v0.8.2
The verified QC core retained inside v0.9:
- normalized weighted matching cost;
- rejection-aware Hungarian assignment with dummy unmatched slots;
- center / IoU / size / grid / neighborhood costs;
- `Geometry = NOT_EVALUATED` when reliable pairs are insufficient;
- visual overlay and detailed JSON diagnostics.

Baseline QC settings used in current benchmark:
- accept_match_cost: 0.35
- dummy_unmatched_cost: 0.17
- min_reliable_match_fraction: 0.50
- warn_center_drift_pct: 1.0
- warn_size_drift_pct: 8.0

## Overlay

- GREEN = reliable match
- YELLOW = reliable match with geometry drift
- RED = missing from AFTER
- BLUE = added in AFTER

RelateAnything semantic relations remain **advisory only**. Geometry truth is deterministic bbox/grid math.

## Installation

Clone directly into ComfyUI custom nodes:

```bat
git clone https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.9.git "Q:\AI_ArchViz\ComfyUI_windows_portable\ComfyUI\custom_nodes\ARCHVIZ-RelateAnything-QC-v0.9"
```

Keep the existing v0.6 ONNX / RARegions package installed.

Restart ComfyUI and import:

`workflows/ARCHVIZ_RELATEANYTHING_STABLE_SAMPLING_QC_v009.json`

## Important limitations

- Same camera / crop / composition is required for meaningful geometry QC.
- SAM3 detection quality still matters; always inspect the overlay.
- Dense repetitive facades can remain ambiguous.
- v0.9 is still LAB: it should support human QC, not replace it.
- Do not update the main Torch / Transformers stack for this node.

## Repository history

- v0.6 — RelateAnything ONNX integration
- v0.7 — deterministic BEFORE vs AFTER bbox QC
- v0.8.2 — normalized rejection-aware Smart Matcher
- **v0.9 — stable paired-region sampling + v0.8.2 QC core**


## Links

### Current project
- **ARCHVIZ RelateAnything QC v0.9**  
  https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.9
- **Main v0.9 workflow**  
  https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.9/blob/main/workflows/ARCHVIZ_RELATEANYTHING_STABLE_SAMPLING_QC_v009.json
- **Stable sampler source**  
  https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.9/blob/main/stable_sampler_v09.py

### Required / related project components
- **RelateAnything ONNX / RARegions v0.6**  
  https://github.com/lutiy-dev/ComfyUI-RelateAnything-v0.6
- **RelateAnything model: maelic/relsgg-vits16plus**  
  https://huggingface.co/maelic/relsgg-vits16plus
- **SAM 3 upstream project**  
  https://github.com/facebookresearch/sam3
- **ComfyUI**  
  https://github.com/comfyanonymous/ComfyUI
- **SciPy linear_sum_assignment / Hungarian assignment docs**  
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

### Previous QC versions
- **v0.7 — deterministic BEFORE vs AFTER bbox QC**  
  https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.7
- **v0.8.2 — normalized rejection-aware Smart Matcher**  
  https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.8.2

### Installation
Clone v0.9 directly into ComfyUI:

```bat
git clone https://github.com/lutiy-dev/ARCHVIZ-RelateAnything-QC-v0.9.git "Q:\AI_ArchViz\ComfyUI_windows_portable\ComfyUI\custom_nodes\ARCHVIZ-RelateAnything-QC-v0.9"
```

If your ComfyUI is installed on another drive or folder, keep the same repository URL and only change the destination path after it.
