# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""loop.py: a candidate replaces the champion only on a paired gain over every
workload's exam with no workload falling too far, every card is used, and the loop
resumes from what its generations recorded."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import loop  # noqa: E402

RULE = {"min_mean_delta": 0.0, "max_workload_drop": 0.2}


def the_best_candidate_over_all_workloads_is_promoted_on_a_gain():
    per = {
        "a": {"repo": {"mean_delta": 0.10, "paired_tasks": 30}, "cad": {"mean_delta": -0.05, "paired_tasks": 10}},
        "b": {"repo": {"mean_delta": 0.02, "paired_tasks": 30}, "cad": {"mean_delta": 0.00, "paired_tasks": 10}},
    }
    decision = loop.decide(per, RULE)
    assert decision["promoted"] and decision["chosen"] == "a"
    assert abs(decision["scores"]["a"]["mean_delta"] - (0.10 * 30 - 0.05 * 10) / 40) < 1e-9


def a_candidate_that_drops_one_workload_too_far_is_not_promoted():
    per = {"a": {"repo": {"mean_delta": 0.30, "paired_tasks": 40}, "cad": {"mean_delta": -0.30, "paired_tasks": 5}}}
    decision = loop.decide(per, RULE)
    assert decision["chosen"] == "a" and not decision["promoted"] and "cad" in decision["reason"]


def no_gain_or_no_pairs_keeps_the_champion():
    assert not loop.decide({"a": {"repo": {"mean_delta": 0.0, "paired_tasks": 9}}}, RULE)["promoted"]
    assert not loop.decide({"a": {"repo": {"mean_delta": None}}}, RULE)["promoted"]


def a_paired_report_counts_its_wins_and_losses():
    report = {"summary": {"mean_delta": 0.1}, "records": [{"task": "x", "delta": 0.5}, {"task": "y", "delta": -0.25},
                                                         {"task": "z", "delta": 0.0}]}
    assert loop.paired(report) == {"mean_delta": 0.1, "paired_tasks": 3, "wins": 1, "losses": 1}


def _gen(work: Path, n: int, state: dict) -> None:
    d = work / "generations" / f"{n:03d}"
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"generation": n, **state}))


def the_champion_is_the_last_promoted_candidate_with_its_exams(tmp_path):
    campaign = loop.Campaign({"paths": {"work": str(tmp_path)}})
    assert loop.champion(campaign)["adapter"] is None
    base = {"name": "base", "adapter": None, "exams": {}}
    _gen(tmp_path, 0, {"champion_in": base, "exams": {"repo": "j0"}, "exam_stack": "s",
                       "candidates": {"a": {"name": "gen001-a", "adapter": "a1", "exams": {"repo": "j1"}}},
                       "decision": {"promoted": False, "chosen": "a"}})
    kept = loop.champion(campaign)
    assert kept["adapter"] is None and kept["exams"] == {"repo": "j0"}, "a kept champion carries its exams"
    _gen(tmp_path, 1, {"champion_in": kept, "exams": {"repo": "j0"},
                       "candidates": {"b": {"name": "gen002-b", "adapter": "a2", "exams": {"repo": "j2"}}},
                       "decision": {"promoted": True, "chosen": "b"}})
    assert loop.champion(campaign) == {"name": "gen002-b", "adapter": "a2", "exams": {"repo": "j2"}}


def a_generation_resumes_after_its_last_finished_step(tmp_path):
    campaign = loop.Campaign({"paths": {"work": str(tmp_path)}})
    loop.Generation(campaign, 0).record("explore", explore_jobs={"repo": "j"})
    again = loop.Generation(campaign, 0)
    assert again.done("explore") and not again.done("learn") and again.state["explore_jobs"] == {"repo": "j"}


TRAIN = {"base": "local:B", "base_precision": "fp8", "max_block": 16384, "rank": 16, "alpha": 32, "lr": 5e-5,
         "steps": 80, "records_per_step": 4, "eval_every": 10, "patience": 3, "monitor_share": 0.1, "seed": 7}  # seed: only passed through, any value


def every_variant_is_a_candidate_over_the_defaults():
    # seed: a variant's own value must replace the default, so the two must differ
    campaign = loop.Campaign({"train": {**TRAIN, "variant": [{"name": "s1"}, {"name": "s2", "seed": 2, "lr": 1e-4}]}})
    recipes = loop.variants(campaign)
    assert [r["name"] for r in recipes] == ["s1", "s2"]
    assert recipes[0]["seed"] == 7 and recipes[1]["seed"] == 2 and recipes[1]["lr"] == 1e-4
    assert loop.variants(loop.Campaign({"train": TRAIN}))[0]["name"] == "a"


def imitation_learns_from_every_generation_decisive_actions_from_the_base():
    train = {**TRAIN, "only_actions": ["edit_file", "write_file"], "roles": ["main"]}
    states = [{"explore_exports": {"repo": "e0"}}, {"explore_exports": {"repo": "e1", "cad": "c1"}}]
    assert loop.learning_exports(train, states, "repo") == ["e0", "e1"]
    view = loop.dataset_args(train)
    assert view[:2] == ["--view", "sft-step"] and view[view.index("--only-actions") + 1] == "edit_file,write_file"
    assert view[view.index("--from") + 1] == "local:B"
    args = loop.train_args(train, {"adapter": "/a/gen001.safetensors"})
    assert args[args.index("--from") + 1] == "local:B" and "--eval-every" in args


def reinforcement_learns_on_policy_from_the_champion():
    train = {**TRAIN, "objective": "policy-gradient",
             "reward": {"metrics": {"iou": 0.5}, "fail_ceiling": 0.5, "advantage": "standardised"},
             "policy_gradient": {"clip_eps": 0.2, "kl_beta": 0.02, "reference": "base", "loss_agg": "seq-mean",
                                 "temperature": 1.0}}
    states = [{"explore_exports": {"cad": "c0"}}, {"explore_exports": {"cad": "c1"}}]
    assert loop.learning_exports(train, states, "cad") == ["c1"]
    args = loop.train_args(train, {"adapter": "/a/g.safetensors"})
    assert args[args.index("--from") + 1] == "local:B+/a/g.safetensors" and "--eval-every" not in args


def self_distillation_distils_the_cliff_and_keeps_the_scalar_advantage_elsewhere():
    train = {**TRAIN, "objective": "self-distill",
             "reward": {"metrics": {"iou": 0.5}, "fail_ceiling": 0.5, "advantage": "standardised"},
             "policy_gradient": {"clip_eps": 0.2, "kl_beta": 0.02, "reference": "start", "loss_agg": "token-sum",
                                 "temperature": 1.0},
             "self_distill": {"teacher_policy": "base", "advantage_bound": 5.0, "sequence_advantage_cap": 40.0}}
    states = [{"explore_exports": {"cad": "c0"}}, {"explore_exports": {"cad": "c1"}}]
    assert loop.learning_exports(train, states, "cad") == ["c1"]  # on-policy: the newest attempts only
    view = loop.dataset_args(train)
    assert view[:2] == ["--view", "policy-gradient"] and "--cliff-distill" in view and "--reward-metric" in view
    args = loop.train_args(train, {"adapter": "/a/g.safetensors"})
    assert args[args.index("--from") + 1] == "local:B+/a/g.safetensors" and "--eval-every" not in args
    assert args[args.index("--teacher-policy") + 1] == "base"
    assert args[args.index("--advantage-bound") + 1] == "5.0"
    assert args[args.index("--sequence-advantage-cap") + 1] == "40.0"
    # Every attempt distilled, no scalar advantage and so no reward shape to pass.
    every = {**train, "self_distill": {**train["self_distill"], "all_attempts": True}}
    view = loop.dataset_args(every)
    assert view[:2] == ["--view", "self-distill"] and "--reward-metric" not in view and "--cliff-distill" not in view


def a_pass_fail_reward_names_no_metric_and_no_ceiling():
    train = {**TRAIN, "objective": "policy-gradient",
             "reward": {"metrics": {}, "fail_ceiling": 0.0, "advantage": "standardised"},
             "policy_gradient": {"clip_eps": 0.2, "kl_beta": 0.02, "reference": "base", "loss_agg": "token-sum",
                                 "temperature": 1.0}}
    view = loop.dataset_args(train)
    assert "--reward-metric" not in view and "--fail-ceiling" not in view, view
    assert view[view.index("--advantage") + 1] == "standardised"


def a_dataset_can_be_bounded_to_what_the_training_reads():
    view = loop.dataset_args({**TRAIN, "dataset_limit": 480})
    assert view[view.index("--limit") + 1] == "480"
    assert "--limit" not in loop.dataset_args(TRAIN)


def an_unknown_objective_is_refused():
    try:
        loop.dataset_args({**TRAIN, "objective": "dpo"})
    except SystemExit as error:
        assert "dpo" in str(error)
    else:
        raise AssertionError("an unknown objective must be refused")


def a_profile_spreads_a_workload_over_its_cards():
    campaign = loop.Campaign({"bench": {"sven": "/s"}, "serve": {"base_port": 8800}})
    text = loop.profile_text(campaign, {"name": "emb", "concurrency": 16, "roles": ["designer"],
                                        "tools": {"vemu": "/v"}}, "M", 4, [0, 2])
    assert 'base_url = ["http://127.0.0.1:8800/v1", "http://127.0.0.1:8802/v1"]' in text
    assert "attempts = 4" in text and "concurrency = 16" in text and 'designer = "brain"' in text
    assert '[tools]\nvemu = "/v"' in text


def with_replicas_one_server_spans_the_cards():
    one_each = loop.Campaign({"bench": {"sven": "/s"}, "serve": {"base_port": 8800}})
    assert loop.servers(one_each, [0, 1]) == [[0], [1]] and loop.endpoints(one_each, [0, 1]) == [8800, 8801]
    spanning = loop.Campaign({"bench": {"sven": "/s"}, "serve": {"base_port": 8800, "replicas": True}})
    assert loop.servers(spanning, [1, 2, 3]) == [[1, 2, 3]] and loop.endpoints(spanning, [1, 2, 3]) == [8801]
    text = loop.profile_text(spanning, {"name": "w", "concurrency": 64}, "M", 1, [0, 1, 2, 3])
    assert 'base_url = ["http://127.0.0.1:8800/v1"]' in text


def a_model_on_a_share_of_the_cards_runs_that_share_of_the_trials():
    campaign = loop.Campaign({"serve": {"cards": [0, 1, 2]}})
    assert loop.scaled(campaign, {"name": "w", "concurrency": 30}, [0, 1, 2])["concurrency"] == 30
    assert loop.scaled(campaign, {"name": "w", "concurrency": 30}, [1])["concurrency"] == 10
    assert loop.scaled(campaign, {"name": "w", "concurrency": 2}, [1])["concurrency"] == 1


def a_workload_can_practise_on_another_pool_than_its_exam_or_not_at_all():
    plain = {"name": "repo", "collection": "c", "concurrency": 20, "explore_attempts": 2, "exam_attempts": 2}
    assert loop.exploration(plain) == {**plain, "attempts": 2}
    frontier = {**plain, "explore_collection": "pool", "explore_concurrency": 36, "explore_attempts": 8}
    got = loop.exploration(frontier)
    assert got["collection"] == "pool" and got["concurrency"] == 36 and got["attempts"] == 8
    assert loop.exploration({**plain, "explore_attempts": 0}) is None, "0 attempts: examined only"


def the_cards_are_shared_among_the_models_served_at_once():
    assert loop.share([0, 1, 2, 3], 1) == [[0, 1, 2, 3]]
    assert loop.share([0, 1, 2, 3], 2) == [[0, 2], [1, 3]]
    assert loop.share([0, 1, 2, 3], 3) == [[0, 3], [1], [2]]
    try:
        loop.share([0, 1], 3)
    except SystemExit:
        pass
    else:
        raise AssertionError("more models than cards must be refused")


def steps_run_side_by_side_and_a_failure_is_raised_after_both_finish():
    met = threading.Barrier(2, timeout=5)
    done = []

    def first():
        met.wait()
        done.append("first")

    def failing():
        met.wait()
        raise RuntimeError("the exam failed")

    try:
        loop.concurrently([first, failing])
    except RuntimeError as error:
        assert "exam failed" in str(error)
    else:
        raise AssertionError("a failed step must stop the loop")
    assert done == ["first"]


def an_exam_the_model_server_failed_decides_nothing():
    assert loop.exam_problems({"model_server_errors": 0, "environment_errors": 0, "errors": 0}) == []
    assert "22 trial(s) ended by the model server" in loop.exam_problems({"model_server_errors": 22})[0]


if __name__ == "__main__":
    import inspect
    import tempfile

    failed = 0
    for name, fn in list(globals().items()):
        if inspect.isfunction(fn) and fn.__module__ == __name__ and not name.startswith("_"):
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    with tempfile.TemporaryDirectory() as tmp:
                        fn(Path(tmp))
                else:
                    fn()
                print(f"ok   {name}")
            except Exception as error:  # noqa: BLE001 - reported, then counted
                failed += 1
                print(f"FAIL {name}: {error!r}")
    sys.exit(1 if failed else 0)
