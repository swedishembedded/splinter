#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements independent examination of self-trained models
# for its clients. If your team needs expertise in grading protocols whose every
# rule is pinned by a test, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Spec of the reference examiner, on synthetic data: answer extraction and the truncation rule,
the whitespace-normalised comparison, the training-overlap exclusion, the exact sign test, the
template-clustered interval, the step and the per-lineage plan rule, and the base's identity."""

import contextlib
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import examiner as ex  # noqa: E402


class Answers(unittest.TestCase):
    def test_the_answer_is_the_last_complete_fenced_block(self):
        reply = "First\n```\n1\n```\nthen\n```lua\n2\t3\n```\n"
        self.assertEqual(ex.extract_answer(reply), "2\t3\n")

    def test_a_reply_without_a_fence_is_its_own_answer(self):
        self.assertEqual(ex.extract_answer("42"), "42")

    def test_an_empty_thinking_block_is_not_part_of_the_answer(self):
        self.assertTrue(ex.grade_reply("<think>\n\n</think>\n\n42", "42"))

    def test_whitespace_runs_collapse_and_blank_lines_are_ignored(self):
        self.assertTrue(ex.answers_match("a   b\n\n c\n", "a\tb\nc"))
        self.assertFalse(ex.answers_match("a b c", "a b\nc"))
        self.assertFalse(ex.answers_match("ab", "a b"))

    def test_a_reply_cut_by_the_token_limit_is_graded_as_written(self):
        # An unclosed block is no fenced block: the whole cut reply is the answer, and it is wrong.
        self.assertFalse(ex.grade_reply("```\n42", "42"))
        # A complete block before the cut is still the last complete block, and is graded.
        self.assertTrue(ex.grade_reply("```\n42\n```\nOr maybe ```\n4", "42"))

    def test_the_request_is_the_instruction_and_the_one_sentence_suffix(self):
        self.assertEqual(ex.REQUEST_SUFFIX, "\n\nAnswer with exactly what it prints, in one fenced code block.")
        self.assertEqual((ex.MAX_TOKENS, ex.KV_CACHE, ex.MAX_BATCH), (2048, "f32", 64))


class Exclusion(unittest.TestCase):
    def test_the_program_is_the_last_lua_block_of_the_instruction(self):
        self.assertEqual(ex.item_program("What does it print?\n\n```lua\nprint(1)\n```"), "print(1)")
        with self.assertRaises(ValueError):
            ex.item_program("no program")

    def test_comments_are_removed_but_strings_are_kept(self):
        src = 'print("a -- b") -- note\n--[[ block\ncomment ]] print([[x -- y]]) --[==[ z ]==]\nprint(\'q\')'
        self.assertEqual(ex.strip_comments(src), 'print("a -- b") \n print([[x -- y]]) \nprint(\'q\')')

    def test_an_item_equal_exactly_or_after_normalisation_is_excluded(self):
        programs = ["print(1)", "local x = 2\nprint(x)", "print(3)", 'print("a b")']
        trained = ["print(1)", "local x=2 -- two\n  print( x )", 'print("ab")']
        # 0 exact, 1 after removing comments and whitespace, 3 equal once whitespace is removed;
        # 2 matches nothing.
        self.assertEqual(ex.excluded_items(programs, trained), {0, 1, 3})


class Statistics(unittest.TestCase):
    def test_the_sign_test_is_the_exact_binomial_upper_tail(self):
        self.assertAlmostEqual(ex.sign_test_p(16, 1), 18 / 2 ** 17)
        self.assertEqual(ex.sign_test_p(0, 0), 1.0)
        self.assertAlmostEqual(ex.sign_test_p(5, 5), 638 / 1024)
        # The published S4 step r3 generation 3: 67 wins, 36 losses, p 0.0015.
        self.assertEqual(f"{ex.sign_test_p(67, 36):.2g}", "0.0015")

    def test_student_t_quantiles(self):
        self.assertAlmostEqual(ex.t_quantile(0.975, 24), 2.063899, places=5)
        self.assertAlmostEqual(ex.t_quantile(0.95, 24), 1.710882, places=5)
        self.assertAlmostEqual(ex.t_quantile(0.975, 2), 4.302653, places=5)
        self.assertAlmostEqual(ex.t_quantile(0.975, 1), 12.706205, places=4)

    def test_the_clustered_interval_follows_the_papers_formula(self):
        diffs = [1, 1, 0, -1, 1, 0]
        clusters = ["a", "a", "b", "b", "c", "c"]
        g = ex.clustered_gain(diffs, clusters)
        # mean 1/3; cluster sums of d - mean: 4/3, -5/3, 1/3; SE = sqrt(3/2 * 42/9) / 6.
        se = math.sqrt(1.5 * 42 / 9) / 6
        self.assertAlmostEqual(g["mean"], 1 / 3)
        self.assertAlmostEqual(g["se"], se)
        self.assertEqual((g["families"], g["items"]), (3, 6))
        self.assertAlmostEqual(g["interval"][0], 1 / 3 - 4.302653 * se, places=5)
        self.assertAlmostEqual(g["interval"][1], 1 / 3 + 4.302653 * se, places=5)
        self.assertAlmostEqual(g["lower_one_sided"], 1 / 3 - ex.t_quantile(0.95, 2) * se)

    def test_one_cluster_has_no_interval(self):
        self.assertIsNone(ex.clustered_gain([1, 0], ["a", "a"])["interval"])


class StepsAndPlan(unittest.TestCase):
    def test_a_step_counts_wins_and_losses_on_the_compared_items(self):
        cand = {0: True, 1: True, 2: False, 3: True, 4: True}
        pred = {0: False, 1: True, 2: True, 3: False, 4: False}
        fam = ["a", "a", "b", "b", "c"]
        step = ex.decide_step(cand, pred, fam, excluded={4}, alpha=0.05)
        self.assertEqual((step["compared"], step["excluded"], step["wins"], step["losses"]), (4, 1, 2, 1))
        self.assertEqual((step["items_right"], step["predecessor_items_right"], step["unexcluded_items_right"]),
                         (3, 2, 4))
        self.assertAlmostEqual(step["p_one_sided"], 0.5)
        self.assertFalse(step["step_holds"])

    def test_a_step_holds_only_with_more_wins_at_p_below_alpha(self):
        n = 10
        cand = {i: True for i in range(n)}
        pred = {i: False for i in range(n)}
        step = ex.decide_step(cand, pred, ["a"] * 5 + ["b"] * 5, excluded=set(), alpha=0.05)
        self.assertTrue(step["step_holds"])  # 10 wins, 0 losses, p = 2^-10

    def test_the_plan_needs_every_step_of_enough_lineages(self):
        plan = {"alpha": 0.05, "steps": [1, 2, 3], "series_required": 3, "series_expected": 4}

        def step(holds, w=10, lo=1):
            return {"step_holds": holds, "wins": w, "losses": lo, "valid": True}

        steps = {(s, k): step(True) for s in ("r0", "r1", "r2", "r3") for k in (1, 2, 3)}
        steps[("r3", 3)] = step(False, 5, 5)
        verdict = ex.decide_plan(steps, plan)
        self.assertEqual(verdict["succeeded"], ["r0", "r1", "r2"])
        self.assertTrue(verdict["plan_met"])
        self.assertEqual(verdict["pooled"]["3"]["wins"], 35)
        steps[("r2", 1)] = dict(step(True), valid=False)  # an invalid step does not hold
        self.assertFalse(ex.decide_plan(steps, plan)["plan_met"])


class BaseIdentity(unittest.TestCase):
    def test_both_naming_rules_are_computed_from_the_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "config.json").write_text("{}")
            (d / "a.safetensors").write_bytes(b"A")
            (d / "tokenizer.json").write_text("not part of the identity")
            sha = {n: hashlib.sha256(b).hexdigest() for n, b in (("config.json", b"{}"), ("a.safetensors", b"A"))}
            brain = f"a.safetensors=sha256:{sha['a.safetensors']}\nconfig.json=sha256:{sha['config.json']}\n"
            learner = f"a.safetensors sha256:{sha['a.safetensors']}\n"
            got = ex.checkpoint_digests(d)
            self.assertEqual(got["brain"], "sha256:" + hashlib.sha256(brain.encode()).hexdigest())
            self.assertEqual(got["learner"], "sha256:" + hashlib.sha256(learner.encode()).hexdigest())

    def test_the_served_base_is_the_one_card_without_an_adapter(self):
        cards = [{"id": "qwen3"}, {"id": "x:1", "adapter": {"digest": "sha256:aa", "id": "x:1"}}]
        self.assertEqual(ex.served_ids(cards), ("qwen3", {"sha256:aa": "x:1"}))
        with self.assertRaises(RuntimeError):
            ex.served_ids([{"id": "a"}, {"id": "b"}])


class Analyse(unittest.TestCase):
    def test_grades_with_a_failed_request_make_the_step_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in [
                {"index": 0, "right": True, "finish_reason": "stop"},
                {"index": 1, "right": False, "finish_reason": "length"},
                {"index": 2, "right": None, "error": "URLError"}]) + "\n")
            grades, failures, truncated = ex.read_grades(p)
            self.assertEqual((grades, failures, truncated), ({0: True, 1: False, 2: False}, 1, 1))


# A stand-in for `brain serve`: parses the flags the examiner passes, writes the key and the ready
# file, lists the base and the adapters of --serve-adapters by digest, and answers an item right
# when the model's rank (base 0, adapter of step k: k) is at least the item's index modulo 4.
FAKE_BRAIN = r"""
import hashlib, json, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
args = sys.argv[1:]
assert args[args.index("serve") + 1] == "--openai"
opt = {args[i]: args[i + 1] for i in range(args.index("serve") + 1, len(args) - 1) if args[i].startswith("--")}
assert opt["--kv-cache"] == "f32" and opt["--max-batch"] == "64" and opt["--adapter-targets"] == "all"
cards = [{"id": "base"}]
rank = {"base": 0}
for f in sorted(Path(opt["--serve-adapters"]).rglob("*.safetensors")):
    d = "sha256:" + hashlib.sha256(f.read_bytes()).hexdigest()
    cards.append({"id": f"a{f.parent.name}", "adapter": {"id": f"a{f.parent.name}", "digest": d, "version": 1}})
    rank[f"a{f.parent.name}"] = int(f.parent.name)
exam = [json.loads(l) for l in open(Path(opt["--checkpoint"]) / "exam.jsonl")]
key = "sk-test"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        assert self.headers["Authorization"] == "Bearer " + key
        self.reply({"data": cards})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["temperature"] == 0.0 and body["max_tokens"] == 2048
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
        text = body["messages"][0]["content"]
        i = next(n for n, it in enumerate(exam) if text == it["instruction"] + "\n\nAnswer with exactly what it prints, in one fenced code block.")
        ok = rank[body["model"]] >= i % 4
        content = "```\n" + (exam[i]["reference"] if ok else "wrong") + "\n```"
        self.reply({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})
srv = ThreadingHTTPServer(tuple([opt["--openai"].rsplit(":", 1)[0], int(opt["--openai"].rsplit(":", 1)[1])]), H)
Path(opt["--api-keys-out"]).write_text(json.dumps({"openai": key}))
Path(opt["--ready-file"]).write_text("")
srv.serve_forever()
"""


class EndToEnd(unittest.TestCase):
    def test_grade_and_analyse_a_synthetic_series_on_a_stand_in_server(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            exam = [{"family": f"f{i % 3}", "instruction": f"What does it print?\n\n```lua\nprint({i})\n```",
                     "reference": str(i)} for i in range(12)]
            (d / "ck").mkdir()
            (d / "ck" / "exam.jsonl").write_text("".join(json.dumps(e) + "\n" for e in exam))
            (d / "ck" / "model.safetensors").write_bytes(b"weights")
            exam_path = d / "ck" / "exam.jsonl"
            base_digest = ex.checkpoint_digests(d / "ck")["learner"]
            for k in (1, 2, 3):
                adapter = d / "ev" / "adapters" / "s" / "r0" / str(k) / "adapter.safetensors"
                adapter.parent.mkdir(parents=True)
                adapter.write_bytes(f"adapter {k}".encode())
                hand = d / "ev" / "s" / "handoff" / "candidates" / "r0" / str(k)
                hand.mkdir(parents=True)
                # Step 2's list holds exam item 5's program as a dev program: excluded from steps 2 and 3.
                tasks = [{"role": "train", "program": "print(99)"}]
                if k == 2:
                    tasks.append({"role": "dev", "program": "print( 5 ) -- held out"})
                (hand / "training-tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
                (hand / "candidate.json").write_text(json.dumps({
                    "series": "r0", "step": k, "base_digest": base_digest,
                    "adapter": f"adapters/s/r0/{k}/adapter.safetensors",
                    "adapter_digest": "sha256:" + hashlib.sha256(f"adapter {k}".encode()).hexdigest()}))
            brain = d / "brain"
            brain.write_text("#!" + sys.executable + "\n" + FAKE_BRAIN)
            brain.chmod(0o755)
            plan = d / "plan.toml"
            plan.write_text("alpha = 0.05\nsteps = [1, 2, 3]\nseries_required = 1\nseries_expected = 1\n")
            out = d / "out"
            common = ["--exam", str(exam_path), "--evidence", str(d / "ev"), "--series", "s", "--out", str(out)]
            with contextlib.redirect_stdout(io.StringIO()):
                ex.main(["grade", *common, "--brain", str(brain), "--checkpoint", str(d / "ck")])
                ex.main(["analyse", *common, "--plan", str(plan)])
            result = json.loads((out / "aggregate.json").read_text())
            steps = {s["step"]: s for s in result["steps"]}
            # Items right: base 3 (i % 4 == 0), step k: 3 * (k + 1) of 12.
            self.assertEqual([steps[k]["excluded"] for k in (1, 2, 3)], [0, 1, 1])
            self.assertEqual([steps[k]["wins"] for k in (1, 2, 3)], [3, 3, 3])
            self.assertEqual([steps[k]["items_right"] for k in (1, 2, 3)], [6, 8, 11])
            self.assertEqual([steps[k]["losses"] for k in (1, 2, 3)], [0, 0, 0])
            self.assertTrue(all(s["valid"] for s in steps.values()))
            self.assertEqual(result["base_agreement"]["right"], {"r0": 3})
            self.assertIn("| r0 | 3 |", (out / "table.md").read_text())


if __name__ == "__main__":
    unittest.main()
