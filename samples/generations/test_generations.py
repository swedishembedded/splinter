# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Spec of the generations sample's own apparatus: the facts it invents, the
sessions and probes written from them, how answers are graded, and the
criteria the run is judged by. Run: python3 -m unittest test_generations."""

import json
import unittest

import generations as g


class Facts(unittest.TestCase):
    def test_the_same_seed_gives_the_same_facts_and_another_gives_others(self):
        a = g.make_facts(seed=7, generations=3, per_generation=12)
        self.assertEqual(a, g.make_facts(seed=7, generations=3, per_generation=12))
        self.assertNotEqual(a, g.make_facts(seed=8, generations=3, per_generation=12))
        self.assertEqual([f.generation for f in a], [n // 12 for n in range(36)])

    def test_every_fact_has_a_subject_of_its_own_and_a_value_nobody_else_shares_there(self):
        facts = g.make_facts(seed=3, generations=3, per_generation=12)
        self.assertEqual(len({f.subject for f in facts}), len(facts))
        for f in facts:
            self.assertNotEqual(f.value, f.wrong_value)
            self.assertIn(f.subject, f.statement)
            self.assertIn(f.value, f.statement)
            self.assertNotIn(f.wrong_value, f.statement)

    def test_the_wrong_answer_in_a_session_is_not_the_truth(self):
        for f in g.make_facts(seed=5, generations=1, per_generation=12):
            steps = g.session(f)["steps"]
            self.assertEqual([s["source"] for s in steps], ["user", "agent", "user", "agent"])
            self.assertIn(f.wrong_value, steps[1]["message"])
            self.assertNotIn(f.value, steps[1]["message"])
            self.assertIn(f.statement, steps[2]["message"])


class Probes(unittest.TestCase):
    def test_four_probes_per_fact_none_of_them_the_sessions_question_or_each_other(self):
        for f in g.make_facts(seed=11, generations=2, per_generation=12):
            probes = g.probes(f)
            self.assertEqual(len(probes), 4)
            texts = [p["instruction"] for p in probes]
            self.assertEqual(len(set(texts)), 4)
            self.assertNotIn(g.session(f)["steps"][0]["message"], texts)
            for p in probes:
                self.assertEqual(p["reference"], f.value)
                self.assertIn(f.subject, p["instruction"])
                self.assertEqual(p["kind"], "recall")

    def test_the_sealed_probes_are_the_same_questions_in_the_form_absorb_reads(self):
        for f in g.make_facts(seed=2, generations=1, per_generation=7):
            sealed = g.sealed_probes(f)
            self.assertEqual([s["question"] for s in sealed],
                             [p["instruction"] for p in g.probes(f)])
            self.assertEqual(len({s["name"] for s in sealed}), 4)

    def test_the_anchor_is_forty_distinct_questions_with_one_short_reference_each(self):
        anchor = g.anchor_suite()
        self.assertEqual(len(anchor), 40)
        self.assertEqual(len({a["instruction"] for a in anchor}), 40)
        self.assertTrue(all(a["reference"] and len(a["reference"]) < 24 for a in anchor))


class Grading(unittest.TestCase):
    def test_a_fact_is_answered_when_three_of_its_four_probes_are(self):
        facts = g.make_facts(seed=1, generations=1, per_generation=3)
        missed = [p["instruction"] for p in g.probes(facts[0])[:2]]  # two wrong: not answered
        missed += [g.probes(facts[1])[0]["instruction"]]  # one wrong: answered
        answered = g.answered_facts(facts, missed)
        self.assertEqual(answered, {facts[1].key, facts[2].key})


def row(step, generation, current, retained, anchor, kind="update"):
    return {"step": step, "generation": generation, "kind": kind, "current": current,
            "retention": retained, "anchor": anchor}


class Criteria(unittest.TestCase):
    base = 33

    def test_a_run_that_keeps_everything_and_learns_everything_succeeds(self):
        rows = [
            row("g0.u2", 0, (12, 12), (0, 0), 33),
            row("g0.consolidate", 0, (12, 12), (0, 0), 32, "consolidate"),
            row("g1.u2", 1, (12, 12), (12, 12), 33),
        ]
        verdict = g.judge(rows, anchor_baseline=self.base)
        self.assertTrue(verdict["success"], verdict)

    def test_losing_an_earlier_fact_beyond_the_tolerance_fails_retention(self):
        rows = [row("g0.u2", 0, (12, 12), (0, 0), 33),
                row("g1.u0", 1, (4, 4), (10, 12), 33)]
        verdict = g.judge(rows, anchor_baseline=self.base)
        self.assertFalse(verdict["retention"]["held"])
        self.assertFalse(verdict["success"])

    def test_a_generation_that_does_not_reach_full_recall_fails(self):
        verdict = g.judge([row("g0.u2", 0, (11, 12), (0, 0), 33)], anchor_baseline=self.base)
        self.assertFalse(verdict["recall"]["held"])

    def test_the_anchor_may_lose_the_declared_tolerance_and_no_more(self):
        ok = g.judge([row("g0.u2", 0, (12, 12), (0, 0), self.base - g.ANCHOR_TOLERANCE)],
                     anchor_baseline=self.base)
        bad = g.judge([row("g0.u2", 0, (12, 12), (0, 0), self.base - g.ANCHOR_TOLERANCE - 1)],
                      anchor_baseline=self.base)
        self.assertTrue(ok["anchor"]["held"])
        self.assertFalse(bad["anchor"]["held"])


if __name__ == "__main__":
    unittest.main()


FAKE_SPLINTER = r'''#!/usr/bin/env python3
# A stand-in for the splinter binary: answers what a cycle asks of it, from
# the files the cycle wrote, so the driver can be run end to end without a model.
import json, sys
args = sys.argv[1:]
state = args[args.index("--state") + 1]
verb = next(a for a in args[4:] if a in ("eval", "absorb", "consolidate", "runs"))
if verb == "eval":
    if "--freeze" in args:
        print(json.dumps({"anchor": {"version": 1}}))
    else:
        suite = args[args.index("--suite") + 1]
        tasks = [json.loads(l) for l in open(suite)]
        taught = set(open(state + "/taught").read().split()) if __import__("os").path.exists(state + "/taught") else set()
        missed = [{"instruction": t["instruction"]} for t in tasks
                  if t["reference"].isdigit() and t["reference"] not in taught]
        print(json.dumps({"scores": [{"missed": missed}]}))
elif verb == "absorb":
    import os, glob
    os.makedirs(state, exist_ok=True)
    folder = next(a for a in args if "sessions" in a)
    with open(state + "/taught", "a") as f:
        for path in glob.glob(folder + "/*.atif.json"):
            steps = json.load(open(path))["steps"]
            f.write(" ".join(w.strip(".") for w in steps[2]["message"].split() if w.strip(".").isdigit()) + "\n")
    print(json.dumps({"run": "run-1", "release": "blake3:aa", "gate": {"passed": True, "answered": 4, "claims": [1, 2, 3, 4]},
                      "dataset": {"trained": 72}, "replayed_records": 0, "stopped": None}))
elif verb == "consolidate":
    print(json.dumps({"run": "run-2", "consolidated": {"release": "blake3:bb", "gate": {"passed": True},
                      "steps_completed": 9, "final_loss": 0.1, "dataset": {"trained": 200}, "replayed_records": 10}}))
else:
    print(json.dumps({"stages": [{"stage": "train", "duration_ms": 1500}]}))
'''


class Driver(unittest.TestCase):
    def test_a_cycle_runs_end_to_end_against_a_stand_in_and_is_judged(self):
        import argparse, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fake = out / "splinter"
            fake.write_text(FAKE_SPLINTER)
            fake.chmod(0o755)
            g.build(out, seed=3, generations=2)
            args = argparse.Namespace(out=str(out), mode="merge", splinter=str(fake), brain=None,
                                      policy_weights="policy", helper="helper", helper_context=4096,
                                      policy_context=4096, absorb_flag=["--no-stance"],
                                      lr=None, epochs=None)
            g.run(args)
            rows = [json.loads(l) for l in (out / "results-merge.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in rows],
                             ["base"] + [f"g{n}.{s}" for n in (0, 1)
                                         for s in ("u0", "u1", "u2", "consolidate")])
            last = rows[-1]
            self.assertEqual(last["current"], [12, 12])
            self.assertEqual(last["retention"], [12, 12])
            self.assertEqual(rows[1]["stage_s"], {"train": 1.5})
            verdict = json.loads((out / "verdict-merge.json").read_text())
            self.assertTrue(verdict["success"], verdict)
            self.assertTrue((out / "splinter-merge.log").read_text().count("absorb") >= 6)
