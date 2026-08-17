# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""A7 test 6: count learned parameters in the SHIPPED bundle models/final-v1.

READ-ONLY on models/final-v1.  Writes only build/a7/capacity.json.

Counting convention
  LightGBM tree: L leaves -> L leaf values + (L-1) split thresholds
                 + (L-1) split feature indices  = 3L-2 learned scalars.
  XGBoost tree : leaves*size_leaf_vector leaf weights + internal
                 (split_condition + split_index) = 2 per internal node.
  IRT          : W (D x K) + a (M x K) + b (M).
  SVD          : components (K x bins) per view.
  kNN          : stored index nnz (+ indices) + stored outcome table.
  lookup       : 32-byte key + 3 scores + 3 costs per row.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "models" / "final-v1"
OUT = ROOT / "build" / "a7"
OUT.mkdir(parents=True, exist_ok=True)

N_TRAIN = 1760
N_MODELS = 3
N_TARGETS = 3   # score, in_tokens, out_tokens


def lgbm_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    leaves = [int(m) for m in re.findall(r"^num_leaves=(\d+)$", text, re.M)]
    # the header also carries a global num_leaves; per-tree blocks follow "Tree=N"
    n_trees = len(re.findall(r"^Tree=\d+$", text, re.M))
    tree_leaves = leaves[-n_trees:] if n_trees and len(leaves) >= n_trees else leaves
    total_leaves = int(sum(tree_leaves))
    internal = total_leaves - n_trees
    params = total_leaves + 2 * internal
    return {"file": path.name, "trees": n_trees, "leaves": total_leaves,
            "internal_nodes": internal, "learned_scalars": params}


def xgb_file(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    model = obj["learner"]["gradient_booster"]["model"]
    trees = model["trees"]
    slv = int(trees[0]["tree_param"].get("size_leaf_vector", 1) or 1)
    slv = max(slv, 1)
    total_leaves = 0
    total_internal = 0
    for t in trees:
        left = t["left_children"]
        leaf = sum(1 for c in left if c == -1)
        total_leaves += leaf
        total_internal += len(left) - leaf
    params = total_leaves * slv + 2 * total_internal
    return {"file": path.name, "trees": len(trees), "size_leaf_vector": slv,
            "leaves": total_leaves, "internal_nodes": total_internal,
            "learned_scalars": params}


def main() -> None:
    report = {"bundle": str(BUNDLE), "components": {}, "totals": {}}

    # ---- LightGBM
    lg = {}
    for p in sorted((BUNDLE / "lgbm").glob("*.txt")):
        lg[p.stem] = lgbm_file(p)
    report["components"]["lightgbm"] = {
        "per_file": lg,
        "n_boosters": len(lg),
        "total_trees": sum(v["trees"] for v in lg.values()),
        "total_leaves": sum(v["leaves"] for v in lg.values()),
        "learned_scalars": sum(v["learned_scalars"] for v in lg.values()),
    }

    # ---- XGBoost
    xg = {}
    for p in sorted((BUNDLE / "xgb").glob("*.json")):
        xg[p.stem] = xgb_file(p)
    keep = np.load(BUNDLE / "xgb" / "keep-cols.npy")
    report["components"]["xgboost"] = {
        "per_file": xg,
        "n_boosters": len(xg),
        "total_trees": sum(v["trees"] for v in xg.values()),
        "total_leaves": sum(v["leaves"] for v in xg.values()),
        "keep_cols_stored": int(keep.size),
        "learned_scalars": sum(v["learned_scalars"] for v in xg.values())
                           + int(keep.size),
    }

    # ---- IRT
    irt = np.load(BUNDLE / "irt.npz")
    irt_shapes = {k: list(irt[k].shape) for k in irt.files}
    irt_params = int(sum(irt[k].size for k in ("W", "a", "b")))
    irt_scaler = int(sum(irt[k].size for k in irt.files
                         if k.startswith("scaler")))
    report["components"]["irt1d"] = {
        "shapes": irt_shapes, "learned_scalars": irt_params,
        "scaler_scalars": irt_scaler}

    # ---- SVD
    svd = {}
    svd_total = 0
    for name in ("svd-word.npz", "svd-char.npz"):
        d = np.load(BUNDLE / name)
        comp = d["components"]
        svd[name] = {"shape": list(comp.shape), "params": int(comp.size)}
        svd_total += int(comp.size)
    report["components"]["svd"] = {"per_view": svd, "learned_scalars": svd_total}

    # ---- kNN
    idx = sp.load_npz(BUNDLE / "knn" / "index.npz")
    out = np.load(BUNDLE / "knn" / "outcomes.npz")
    knn_stored = int(idx.nnz) * 2 + int(sum(out[k].size for k in out.files))
    report["components"]["knn_k40"] = {
        "index_shape": list(idx.shape), "index_nnz": int(idx.nnz),
        "outcome_shapes": {k: list(out[k].shape) for k in out.files},
        "stored_scalars": knn_stored,
        "note": "non-parametric: the entire train split is memorized verbatim",
    }

    # ---- lookup
    lk = np.load(BUNDLE / "lookup.npz")
    lk_rows = int(lk["key"].shape[0])
    lk_stored = lk_rows * (32 + 3 + 3)
    report["components"]["lookup"] = {
        "rows": lk_rows,
        "shapes": {k: list(lk[k].shape) for k in lk.files},
        "stored_scalars": lk_stored,
        "note": "sha256(prompt) -> realized score/cost; pure memorization",
    }

    learned = (report["components"]["lightgbm"]["learned_scalars"]
               + report["components"]["xgboost"]["learned_scalars"]
               + report["components"]["irt1d"]["learned_scalars"]
               + report["components"]["svd"]["learned_scalars"])
    memorized = (report["components"]["knn_k40"]["stored_scalars"]
                 + report["components"]["lookup"]["stored_scalars"])
    data_cells = N_TRAIN * N_MODELS * N_TARGETS
    report["totals"] = {
        "supervised_learned_scalars": int(learned),
        "unsupervised_svd_scalars": report["components"]["svd"]["learned_scalars"],
        "memorized_scalars": int(memorized),
        "grand_total": int(learned + memorized),
        "train_episodes": N_TRAIN,
        "supervised_label_cells_1760x3x3": data_cells,
        "params_per_label_cell": round((learned) / data_cells, 2),
        "params_per_label_cell_excl_svd": round(
            (learned - report["components"]["svd"]["learned_scalars"]) / data_cells, 2),
    }
    (OUT / "capacity.json").write_text(json.dumps(report, indent=1),
                                       encoding="utf-8")
    print(json.dumps(report["totals"], indent=1))
    print("lightgbm:", json.dumps({k: v for k, v in
          report["components"]["lightgbm"].items() if k != "per_file"}))
    print("xgboost:", json.dumps({k: v for k, v in
          report["components"]["xgboost"].items() if k != "per_file"}))
    print("irt:", json.dumps(report["components"]["irt1d"]))
    print("svd:", json.dumps(report["components"]["svd"]))
    print("knn:", json.dumps(report["components"]["knn_k40"]))
    print("lookup:", json.dumps(report["components"]["lookup"]))


if __name__ == "__main__":
    main()
