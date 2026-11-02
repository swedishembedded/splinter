# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""loop.py: a candidate replaces the champion only on a paired exam gain, and
the loop resumes from what its generations recorded."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import loop  # noqa: E402


def a_candidate_is_promoted_only_on_a_positive_paired_gain():
    rule = {"min_mean_delta": 0.0}
    assert loop.decide({"mean_delta": 0.1, "interval95": [-0.05, 0.25]}, rule)["promoted"]
    assert not loop.decide({"mean_delta": 0.0, "interval95": [-0.1, 0.1]}, rule)["promoted"]
    assert not loop.decide({"mean_delta": None}, rule)["promoted"]
    strict = {"require_interval_above_zero": True}
    assert not loop.decide({"mean_delta": 0.1, "interval95": [-0.05, 0.25]}, strict)["promoted"]
    assert loop.decide({"mean_delta": 0.2, "interval95": [0.05, 0.35]}, strict)["promoted"]


def _gen(work: Path, n: int, state: dict) -> None:
    d = work / "generations" / f"{n:03d}"
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"generation": n, **state}))


def the_champion_is_the_last_promoted_adapter_with_its_exam(tmp_path):
    campaign = loop.Campaign({"paths": {"work": str(tmp_path)}})
    assert loop.champion(campaign)["adapter"] is None
    base = {"name": "base", "adapter": None, "exam_job": None}
    _gen(tmp_path, 0, {"champion_in": base, "exam_job": "j0", "candidate": {"name": "gen001", "adapter": "a1",
         "exam_job": "j1"}, "decision": {"promoted": False}})
    kept = loop.champion(campaign)
    assert kept["adapter"] is None and kept["exam_job"] == "j0", "a kept champion carries its exam"
    _gen(tmp_path, 1, {"champion_in": kept, "exam_job": "j0", "candidate": {"name": "gen002", "adapter": "a2",
         "exam_job": "j2"}, "decision": {"promoted": True}})
    assert loop.champion(campaign) == {"name": "gen002", "adapter": "a2", "exam_job": "j2"}


def a_generation_resumes_after_its_last_finished_step(tmp_path):
    campaign = loop.Campaign({"paths": {"work": str(tmp_path)}})
    gen = loop.Generation(campaign, 0)
    gen.record("explore", explore_job="j")
    again = loop.Generation(campaign, 0)
    assert again.done("explore") and not again.done("exam") and again.state["explore_job"] == "j"


def a_second_card_takes_the_exams():
    one = loop.Campaign({"serve": {"card": 0, "port": 8799}})
    assert loop.cards(one) == ((0, 8799), None)
    two = loop.Campaign({"serve": {"card": 0, "port": 8799, "exam_card": 1, "exam_port": 8800}})
    assert loop.cards(two) == ((0, 8799), (1, 8800))


TRAIN = {"roles": ["designer", "specifier"], "images": True, "base": "local:B", "base_precision": "bf16",
         "max_block": 16384, "rank": 16, "alpha": 32, "lr": 5e-5, "steps": 80, "records_per_step": 4,
         "eval_every": 10, "patience": 3, "monitor_share": 0.1, "seed": 7}  # seed: only passed through, any value
RL = {**TRAIN, "objective": "policy-gradient",
      "reward": {"metrics": {"iou": 0.5, "feature_match": 0.25, "deviation_score": 0.25}, "fail_ceiling": 0.5,
                 "advantage": "standardised"},
      "policy_gradient": {"clip_eps": 0.2, "kl_beta": 0.02, "reference": "base", "loss_agg": "seq-mean",
                          "temperature": 1.0}}


def imitation_learns_from_every_generation_from_the_base():
    campaign = loop.Campaign({"train": TRAIN})
    states = [{"explore_export": "e0"}, {"explore_export": "e1"}]
    assert loop.learning_exports(campaign, states) == ["e0", "e1"]
    view = loop.dataset_args(campaign)
    assert view[:2] == ["--view", "sft-step"] and "--reward-metric" not in view
    train = loop.train_args(campaign, {"adapter": "/a/gen001.safetensors"})
    assert train[train.index("--from") + 1] == "local:B", "imitation trains a fresh adapter on the base"
    assert "--eval-every" in train and "--clip-eps" not in train


def reinforcement_learns_on_policy_from_the_champion():
    campaign = loop.Campaign({"train": RL})
    states = [{"explore_export": "e0"}, {"explore_export": "e1"}]
    assert loop.learning_exports(campaign, states) == ["e1"], "only the rollouts of the policy being trained"
    view = loop.dataset_args(campaign)
    assert view[:2] == ["--view", "policy-gradient"]
    assert [view[i + 1] for i, a in enumerate(view) if a == "--reward-metric"] == [
        "iou=0.5", "feature_match=0.25", "deviation_score=0.25"]
    assert view[view.index("--fail-ceiling") + 1] == "0.5" and view[view.index("--advantage") + 1] == "standardised"
    train = loop.train_args(campaign, {"adapter": "/a/gen001.safetensors"})
    assert train[train.index("--from") + 1] == "local:B+/a/gen001.safetensors", "the champion's adapter goes on"
    assert train[train.index("--kl-beta") + 1] == "0.02"
    for flag in ("--clip-eps", "--reference", "--loss-agg", "--temperature"):
        assert flag in train
    for refused in ("--eval-every", "--patience", "--monitor-share"):
        assert refused not in train, f"{refused} is refused beside the policy-gradient objective"
    assert loop.train_args(campaign, {"adapter": None})[train.index("--from") + 1] == "local:B"


def a_dataset_can_be_built_to_a_smaller_block_than_the_training_fits():
    campaign = loop.Campaign({"train": {**RL, "dataset_max_block": 15360}})
    view = loop.dataset_args(campaign)
    assert view[view.index("--max-block") + 1] == "15360"
    train = loop.train_args(campaign, {"adapter": None})
    assert train[train.index("--max-block") + 1] == "16384"
    plain = loop.dataset_args(loop.Campaign({"train": RL}))
    assert plain[plain.index("--max-block") + 1] == "16384", "by default the dataset fits the training block"


def an_unknown_objective_is_refused():
    campaign = loop.Campaign({"train": {**TRAIN, "objective": "dpo"}})
    try:
        loop.dataset_args(campaign)
    except SystemExit as error:
        assert "dpo" in str(error)
    else:
        raise AssertionError("an unknown objective must stop the loop")


class Loop(unittest.TestCase):
    def test_imitation_learns_from_every_generation_from_the_base(self):
        imitation_learns_from_every_generation_from_the_base()

    def test_reinforcement_learns_on_policy_from_the_champion(self):
        reinforcement_learns_on_policy_from_the_champion()

    def test_a_dataset_can_be_built_to_a_smaller_block_than_the_training_fits(self):
        a_dataset_can_be_built_to_a_smaller_block_than_the_training_fits()

    def test_an_unknown_objective_is_refused(self):
        an_unknown_objective_is_refused()

    def test_a_second_card_takes_the_exams(self):
        a_second_card_takes_the_exams()

    def test_a_candidate_is_promoted_only_on_a_positive_paired_gain(self):
        a_candidate_is_promoted_only_on_a_positive_paired_gain()

    def test_the_champion_is_the_last_promoted_adapter_with_its_exam(self):
        with tempfile.TemporaryDirectory() as tmp:
            the_champion_is_the_last_promoted_adapter_with_its_exam(Path(tmp))

    def test_a_generation_resumes_after_its_last_finished_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            a_generation_resumes_after_its_last_finished_step(Path(tmp))


if __name__ == "__main__":
    unittest.main()
