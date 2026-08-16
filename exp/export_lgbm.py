# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Export fitted LightGBM models to flat numpy arrays + pure-numpy predictor.

The exported format is runtime-portable (numpy only, no lightgbm) and the
predictor is verified to match Booster.predict within 1e-10.
"""

from __future__ import annotations

import numpy as np


def export_booster(booster) -> dict:
    """Flatten a lightgbm Booster into arrays (one big node table)."""

    dump = booster.dump_model()
    nodes_feature = []
    nodes_threshold = []
    nodes_left = []
    nodes_right = []
    nodes_value = []
    nodes_default_left = []
    tree_roots = []

    def walk(node) -> int:
        index = len(nodes_feature)
        if "split_feature" in node:
            nodes_feature.append(node["split_feature"])
            nodes_threshold.append(node["threshold"])
            nodes_default_left.append(bool(node.get("default_left", True)))
            nodes_value.append(0.0)
            nodes_left.append(-1)
            nodes_right.append(-1)
            left = walk(node["left_child"])
            right = walk(node["right_child"])
            nodes_left[index] = left
            nodes_right[index] = right
        else:
            nodes_feature.append(-1)
            nodes_threshold.append(0.0)
            nodes_default_left.append(True)
            nodes_value.append(node["leaf_value"])
            nodes_left.append(-1)
            nodes_right.append(-1)
        return index

    for tree in dump["tree_info"]:
        tree_roots.append(len(nodes_feature))
        walk(tree["tree_structure"])

    return {
        "feature": np.asarray(nodes_feature, dtype=np.int32),
        "threshold": np.asarray(nodes_threshold, dtype=np.float64),
        "left": np.asarray(nodes_left, dtype=np.int32),
        "right": np.asarray(nodes_right, dtype=np.int32),
        "value": np.asarray(nodes_value, dtype=np.float64),
        "default_left": np.asarray(nodes_default_left, dtype=bool),
        "roots": np.asarray(tree_roots, dtype=np.int32),
    }


def predict_exported(model: dict, X: np.ndarray) -> np.ndarray:
    """Vectorized traversal over rows; NaN goes to the default side."""

    feature = model["feature"]
    threshold = model["threshold"]
    left = model["left"]
    right = model["right"]
    value = model["value"]
    default_left = model["default_left"]
    roots = model["roots"]
    n = X.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for root in roots:
        idx = np.full(n, root, dtype=np.int32)
        active = feature[idx] >= 0
        while active.any():
            rows = np.nonzero(active)[0]
            node = idx[rows]
            feat = feature[node]
            x = X[rows, feat]
            nan_mask = np.isnan(x)
            go_left = (x <= threshold[node]) & ~nan_mask
            go_left |= nan_mask & default_left[node]
            nxt = np.where(go_left, left[node], right[node])
            idx[rows] = nxt
            active[rows] = feature[nxt] >= 0
        out += value[idx]
    return out


def verify_export(booster, X: np.ndarray, atol: float = 1e-10) -> float:
    exported = export_booster(booster)
    ours = predict_exported(exported, X)
    theirs = booster.predict(X)
    err = float(np.abs(ours - theirs).max())
    if err > atol:
        raise AssertionError(f"export mismatch: max abs err {err}")
    return err
