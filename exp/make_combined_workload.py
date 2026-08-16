# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Write the combined public Train+Dev runtime-check workload (2,640 eps)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import ROOT  # noqa: E402  (adds src to sys.path)
from ossp_router.public_runtime import load_public_runtime_workload  # noqa: E402


def episode_dict(episode):
    if episode.prompt is not None:
        return {"episode_id": episode.episode_id, "prompt": episode.prompt}
    return {
        "episode_id": episode.episode_id,
        "messages": [
            {"role": m.role, "content": m.content} for m in episode.messages
        ],
    }


def main() -> None:
    batch = load_public_runtime_workload(
        train_path=ROOT / "data" / "materialized" / "train" / "inputs.json",
        dev_path=ROOT / "data" / "materialized" / "dev" / "inputs.json",
        registry_path=ROOT / "data" / "public-data.v1.json",
    )
    out = ROOT / "build" / "combined-workload" / "inputs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": batch.schema_version,
        "challenge_id": batch.challenge_id,
        "split": batch.split,
        "episodes": [episode_dict(e) for e in batch.episodes],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"{len(batch.episodes)} episodes -> {out}")


if __name__ == "__main__":
    main()
