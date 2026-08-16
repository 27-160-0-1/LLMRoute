# SPDX-FileCopyrightText: Copyright 2026 SKT OSSP challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Determinism/rule-compliance audit for the final blend router.

Skipped when the model bundle is absent (bundle lives in models/final-v1).
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "final-v1"

try:  # runtime deps may be absent in minimal environments
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import lightgbm  # noqa: F401
    import xgboost  # noqa: F401

    _DEPS = True
except ImportError:  # pragma: no cover
    _DEPS = False


@unittest.skipUnless(BUNDLE.is_dir() and _DEPS, "model bundle 또는 런타임 의존성 없음")
class FinalRouterDeterminismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ossp_router.final_router import FinalRouter
        from ossp_router.protocol import load_bundled_policy, load_input

        cls.router = FinalRouter(BUNDLE)
        cls.policy = load_bundled_policy()
        cls.inputs = load_input(ROOT / "data" / "toy" / "inputs.json")

    def _decide(self, inputs, tier):
        from ossp_router.final_router import make_submission

        submission = make_submission(inputs, self.policy, tier, self.router)
        return {d.episode_id: d.model_id for d in submission.decisions}

    def _remap(self, *, reverse=False, rename=False, split=None, challenge=None):
        from ossp_router.protocol import Episode, InputBatch

        episodes = list(self.inputs.episodes)
        if reverse:
            episodes = episodes[::-1]
        mapping = {}
        out = []
        for index, episode in enumerate(episodes):
            new_id = f"renamed-{index:04d}" if rename else episode.episode_id
            mapping[new_id] = episode.episode_id
            out.append(
                Episode(
                    episode_id=new_id,
                    prompt=episode.prompt,
                    messages=episode.messages,
                )
            )
        batch = InputBatch(
            schema_version=self.inputs.schema_version,
            challenge_id=challenge or self.inputs.challenge_id,
            split=split or self.inputs.split,
            episodes=tuple(out),
        )
        return batch, mapping

    def test_same_choice_under_order_shuffle(self):
        for tier in ("fast", "balanced", "premium"):
            base = self._decide(self.inputs, tier)
            shuffled, mapping = self._remap(reverse=True)
            got = self._decide(shuffled, tier)
            for new_id, old_id in mapping.items():
                self.assertEqual(got[new_id], base[old_id], f"{tier} 순서 의존")

    def test_same_choice_under_id_rename(self):
        for tier in ("fast", "balanced", "premium"):
            base = self._decide(self.inputs, tier)
            renamed, mapping = self._remap(rename=True)
            got = self._decide(renamed, tier)
            for new_id, old_id in mapping.items():
                self.assertEqual(got[new_id], base[old_id], f"{tier} ID 의존")

    def test_same_choice_under_split_change(self):
        for tier in ("fast", "balanced", "premium"):
            base = self._decide(self.inputs, tier)
            changed, mapping = self._remap(split="private-eval", challenge="other-run")
            got = self._decide(changed, tier)
            for new_id, old_id in mapping.items():
                self.assertEqual(got[new_id], base[old_id], f"{tier} split 의존")

    def test_repeat_run_identical(self):
        for tier in ("fast", "balanced", "premium"):
            self.assertEqual(self._decide(self.inputs, tier), self._decide(self.inputs, tier))

    def test_single_episode_batch(self):
        from ossp_router.protocol import InputBatch

        single = InputBatch(
            schema_version=self.inputs.schema_version,
            challenge_id=self.inputs.challenge_id,
            split=self.inputs.split,
            episodes=(self.inputs.episodes[0],),
        )
        decisions = self._decide(single, "premium")
        self.assertEqual(len(decisions), 1)
        self.assertIn(next(iter(decisions.values())), ("ax31-light", "ax31", "axk1-think"))


if __name__ == "__main__":
    unittest.main()
