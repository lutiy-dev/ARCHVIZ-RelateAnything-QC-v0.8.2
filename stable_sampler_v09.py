import json
import math
import numpy as np

VERSION = "0.9.0"

def _unwrap(v):
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
        v = v[0]
    if not isinstance(v, dict):
        raise ValueError("RA_REGIONS input must be a dict")
    boxes = np.asarray(v.get("boxes", []), dtype=np.float32)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("RA_REGIONS boxes must have shape [N,4]")
    h = int(v.get("height", 0))
    w = int(v.get("width", 0))
    src = v.get("source_indices", list(range(len(boxes))))
    if len(src) != len(boxes):
        src = list(range(len(boxes)))
    return v, boxes, h, w, [int(x) for x in src]

def _centers(boxes, h, w):
    if len(boxes) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x = ((boxes[:,0] + boxes[:,2]) * 0.5) / float(w)
    y = ((boxes[:,1] + boxes[:,3]) * 0.5) / float(h)
    return np.stack([x, y], axis=1)

def _cell_id(pt, rows, cols):
    x = min(cols - 1, max(0, int(math.floor(float(pt[0]) * cols))))
    y = min(rows - 1, max(0, int(math.floor(float(pt[1]) * rows))))
    return y * cols + x

def _cell_center(cell_id, rows, cols):
    y = cell_id // cols
    x = cell_id % cols
    return np.array([(x + 0.5) / cols, (y + 0.5) / rows], dtype=np.float32)

def _pick_per_shared_cell(bc, ac, rows, cols):
    bmap, amap = {}, {}
    for i, pt in enumerate(bc):
        bmap.setdefault(_cell_id(pt, rows, cols), []).append(i)
    for j, pt in enumerate(ac):
        amap.setdefault(_cell_id(pt, rows, cols), []).append(j)

    pairs = []
    for cid in sorted(set(bmap).intersection(amap)):
        anchor = _cell_center(cid, rows, cols)
        bi = min(bmap[cid], key=lambda i: float(np.linalg.norm(bc[i] - anchor)))
        aj = min(amap[cid], key=lambda j: float(np.linalg.norm(ac[j] - anchor)))
        pair_cost = float(np.linalg.norm(bc[bi] - ac[aj]))
        pairs.append((pair_cost, bi, aj, cid, "shared_cell"))
    return pairs

def _supplement_pairs(bc, ac, used_b, used_a, tolerance):
    candidates = []
    for i in range(len(bc)):
        if i in used_b:
            continue
        for j in range(len(ac)):
            if j in used_a:
                continue
            d = float(np.linalg.norm(bc[i] - ac[j]))
            if d <= tolerance:
                candidates.append((d, i, j, -1, "nearest_supplement"))
    candidates.sort(key=lambda x: x[0])
    out = []
    for item in candidates:
        _, i, j, _, _ = item
        if i in used_b or j in used_a:
            continue
        used_b.add(i)
        used_a.add(j)
        out.append(item)
    return out

def _subset_regions(base, boxes, src, indices):
    out = dict(base)
    out["boxes"] = [boxes[i].astype(float).tolist() for i in indices]
    out["source_indices"] = [int(src[i]) for i in indices]
    out["count"] = len(indices)
    return out

class RAStablePairSamplerV09:
    RETURN_TYPES = ("RA_REGIONS", "RA_REGIONS", "STRING")
    RETURN_NAMES = ("before_sampled", "after_sampled", "sampling_debug")
    FUNCTION = "sample"
    CATEGORY = "RelateAnything/QC v0.9"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before_regions": ("RA_REGIONS",),
                "after_regions": ("RA_REGIONS",),
                "sample_count": ("INT", {"default": 16, "min": 4, "max": 32, "step": 1}),
                "grid_rows": ("INT", {"default": 4, "min": 2, "max": 8, "step": 1}),
                "grid_cols": ("INT", {"default": 4, "min": 2, "max": 8, "step": 1}),
                "supplement_center_tolerance": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 0.50, "step": 0.01}),
            }
        }

    def sample(self, before_regions, after_regions, sample_count, grid_rows, grid_cols, supplement_center_tolerance):
        bbase, bb, bh, bw, bsrc = _unwrap(before_regions)
        abase, ab, ah, aw, asrc = _unwrap(after_regions)

        if bw <= 0 or bh <= 0 or aw <= 0 or ah <= 0:
            raise ValueError("Invalid RA_REGIONS image dimensions")

        bc = _centers(bb, bh, bw)
        ac = _centers(ab, ah, aw)

        pairs = _pick_per_shared_cell(bc, ac, int(grid_rows), int(grid_cols))
        pairs.sort(key=lambda x: x[0])

        selected = []
        used_b, used_a = set(), set()

        for item in pairs:
            if len(selected) >= int(sample_count):
                break
            _, bi, aj, _, _ = item
            if bi in used_b or aj in used_a:
                continue
            used_b.add(bi)
            used_a.add(aj)
            selected.append(item)

        if len(selected) < int(sample_count):
            supplements = _supplement_pairs(
                bc, ac, used_b, used_a, float(supplement_center_tolerance)
            )
            for item in supplements:
                if len(selected) >= int(sample_count):
                    break
                selected.append(item)

        # Deterministic spatial order: top-to-bottom, then left-to-right by BEFORE center.
        selected.sort(key=lambda item: (float(bc[item[1], 1]), float(bc[item[1], 0])))

        bidx = [int(item[1]) for item in selected]
        aidx = [int(item[2]) for item in selected]

        bout = _subset_regions(bbase, bb, bsrc, bidx)
        aout = _subset_regions(abase, ab, asrc, aidx)

        debug = {
            "version": VERSION,
            "before_input_count": len(bb),
            "after_input_count": len(ab),
            "sample_count_requested": int(sample_count),
            "sample_count_output": len(selected),
            "grid_rows": int(grid_rows),
            "grid_cols": int(grid_cols),
            "supplement_center_tolerance": float(supplement_center_tolerance),
            "pairs": [
                {
                    "before_idx": int(bi),
                    "after_idx": int(aj),
                    "before_source_idx": int(bsrc[bi]),
                    "after_source_idx": int(asrc[aj]),
                    "pair_center_distance": float(cost),
                    "cell_id": int(cid),
                    "method": method,
                }
                for cost, bi, aj, cid, method in selected
            ],
        }

        return bout, aout, json.dumps(debug, ensure_ascii=False, indent=2)

NODE_CLASS_MAPPINGS = {
    "RAStablePairSamplerV09": RAStablePairSamplerV09,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RAStablePairSamplerV09": "RA · STABLE PAIRED REGION SAMPLER · v0.9",
}
