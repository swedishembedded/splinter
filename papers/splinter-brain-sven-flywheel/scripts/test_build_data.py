#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements measured research tables for its clients. If
# your team needs expertise in turning run records into tables whose numbers are
# recomputed rather than retyped, you can procure our services by sending an
# email to info@swedishembedded.com.

"""Spec: the development-probe table, the probe evidence parsers and the examiner records.

The pilot table is built from data/results.csv, and its p values are the exact
one-sided sign test of its wins and losses, never typed. The verify and time
records of the development probes are parsed the same way the MEASURED claims
are checked, so a value that drifts from its log is a test failure.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data  # noqa: E402


class PilotTable(unittest.TestCase):
    def test_rows_render_their_arm_their_counts_and_the_computed_sign_test(self):
        rows = [{"experiment": "D46-SS", "repository": "D46 (46 families)",
                 "arm": "decisive vs all-actions (seed 13)", "exam_families": "42",
                 "wins": "16", "losses": "1", "status": "PILOT"}]
        tex = build_data.pilot_results_table(rows)
        self.assertIn("decisive vs all-actions (seed 13)", tex)
        self.assertIn("42 & 16 & 1", tex)
        self.assertIn(f"{build_data.upper_tail(16, 17):.3g}", tex)
        self.assertIn("PILOT", tex)

    def test_the_recorded_pilot_rows_are_pilots_with_integer_counts(self):
        rows = [r for r in build_data.read_results() if r["experiment"] in build_data.PILOTS]
        self.assertTrue(rows, "the development probes have rows in data/results.csv")
        for row in rows:
            self.assertEqual(row["status"], "PILOT", row)
            self.assertGreaterEqual(int(row["wins"]), 0, row)
            self.assertGreaterEqual(int(row["losses"]), 0, row)
            self.assertGreater(int(row["wins"]) + int(row["losses"]), 0, row)
            self.assertGreaterEqual(int(row["exam_families"]),
                                    int(row["wins"]) + int(row["losses"]), row)


class ProbeParsers(unittest.TestCase):
    def test_a_verify_report_is_read_as_its_passed_exposed_and_failed(self):
        text = json.dumps({"run": "r", "experiences": 46, "verdicts": {}, "exposed": 1,
                           "exposed_by": {}, "passed": 30, "failed": 16,
                           "undecided": 0, "unverified": [], "items": []})
        self.assertEqual(build_data.parse_log("d46.s13.verify", text),
                         {"experiences": 46, "passed": 30, "failed": 16, "exposed": 1,
                          "undecided": 0})

    def test_a_time_record_is_read_as_its_wall_and_steps(self):
        text = "exit=0 start=1792893807 wall_s=303 steps=6 lr=5e-5\n"
        self.assertEqual(build_data.parse_log("rcv.6.train", text),
                         {"wall_s": 303, "steps": 6})

    def test_the_behaviour_table_is_read_by_its_arms_and_not_by_another_table(self):
        text = ("## 2\n\nBehaviour is seed-stable even where pass@1 is not\n\n"
                "| arm | D46 edited | D46 aborts |\n|---|---|---|\n"
                "| base | 24/46 (52 %) | 13 (28 %) |\n"
                "| decisive seed 1 | 35/46 (76 %) | 1 (2 %) |\n"
                "### The next section\n| base | 21/46 | 9 |\n")
        m = build_data.parse_log("d46.behaviour", text)
        self.assertEqual((m["base_edited"], m["base_aborted"]), (24, 13))
        self.assertEqual((m["decisive_s1_edited"], m["decisive_s1_aborted"]), (35, 1))
        self.assertNotIn("base_edited", build_data.parse_log("d46.behaviour", "no table"))

    def test_a_compare_report_is_read_as_its_paired_families_and_sign_test(self):
        report = {
            "tasks": 46, "paired": 42, "families": [{}] * 46,
            "clusters": {"wins": 16, "losses": 1, "ties": 25},
            "sign_test": {"discordant": 17, "candidate_wins": 16,
                          "p_value": 0.0001373291015625},
            "candidate": {"competence": {"counts": {"verified": 30}}, "exposed": 1},
            "baseline": {"competence": {"counts": {"verified": 13}}, "exposed": 3},
        }
        m = build_data.parse_log("d46.s13.compare", json.dumps(report))
        self.assertEqual((m["paired"], m["wins"], m["losses"], m["p_value"]),
                         (42, 16, 1, 0.0001373291015625))
        self.assertEqual((m["adapter_verified"], m["base_verified"]), (30, 13))

    def test_a_log_of_a_shape_no_parser_reads_is_named_not_guessed(self):
        # Before: an unknown prefix fell through to the throughput probe's regex and crashed the check.
        with self.assertRaises(build_data.NoParser) as raised:
            build_data.parse_log("lua.sgr2", json.dumps({"arm": "self-generate", "rounds": []}))
        self.assertIn("lua.sgr2", str(raised.exception))

    def test_a_flat_metrics_record_is_its_own_metrics(self):
        flat = {"g0_sd_wins": 16, "g0_sd_mean_delta": "0.034"}
        self.assertEqual(build_data.parse_log("gen27b", json.dumps(flat)), flat)
        self.assertEqual(build_data.parse_log("gen27b.tw", json.dumps(flat)), flat)


def verdict(right, before, wins, losses, p, holds, excluded=None, kv=None):
    """An examiner's step verdict of the shape the hand-off carries (aggregates only)."""
    v = {"items_right": right, "predecessor_items_right": before, "wins": wins, "losses": losses,
         "p_one_sided": p, "step_holds": holds, "integrity_flags": [],
         "collection": {"exam_sha256": "60ac3ae44651500f5cdfdf821770b3b4", "items": 889},
         "overlap": {"exact_matches": 0, "near_duplicates": 0, "records_checked": 1000},
         "protocol": {"name": "direct-completion", "temperature": 0.0, "thinking": "off"},
         "serving": {}, "signature": {"alg": "hmac-sha256"}}
    if excluded is not None:
        v["exclusion"] = {"enabled": True, "excluded": excluded, "compared": 889 - excluded}
    if kv is not None:
        v["serving"]["kv_cache"] = kv
        v["protocol"].update(max_tokens=2048, truncated="graded")
    return v


class ExaminerRecords(unittest.TestCase):
    """The RSI series are decided by an independent examiner's signed aggregate verdicts and by the
    learner's own round records; their values are read, never retyped."""

    def test_a_step_verdict_is_read_as_its_counts_its_sign_test_and_its_exclusions(self):
        m = build_data.parse_log("rsi4.v.r2.g1", json.dumps(verdict(118, 12, 109, 3, 4.511e-29, True, 1, "f32")))
        self.assertEqual((m["right"], m["predecessor_right"], m["wins"], m["losses"]), (118, 12, 109, 3))
        self.assertEqual((m["p"], m["holds"], m["excluded"], m["compared"]), ("4.5e-29", "yes", 1, 888))
        self.assertEqual((m["kv"], m["max_tokens"], m["truncated"]), ("f32", 2048, "graded"))
        self.assertEqual(m["exam"], "60ac3ae44651")

    def test_a_verdict_carries_its_family_clustered_interval_and_its_count_before_exclusion(self):
        v = verdict(225, 181, 99, 55, 2.45e-4, True, 0, "f32")
        v["paired_gain"] = {"families": 25, "interval": [-0.01876857, 0.11775620], "items": 889, "mean": 0.04949381}
        v["exclusion"]["unexcluded"] = {"items_right": 226}
        m = build_data.parse_log("rsi4.v.r1.g3", json.dumps(v))
        self.assertEqual((m["gain"], m["gain_lo"], m["gain_hi"], m["gain_families"]), ("0.0495", "-0.0188", "0.1178", 25))
        self.assertEqual(m["unexcluded_right"], 226)

    def test_a_verdict_without_an_exclusion_record_reports_none_rather_than_zero(self):
        m = build_data.parse_log("rsi2.v.r0.g1", json.dumps(verdict(4, 0, 4, 0, 0.0625, False)))
        self.assertNotIn("excluded", m)
        self.assertEqual((m["p"], m["holds"]), ("0.0625", "no"))

    def test_a_plan_verdict_is_read_as_its_rule_outcome_and_pooled_steps(self):
        plan = {"plan_met": True, "series_succeeded": 4, "series_failed": 0,
                "pooled_steps": {"1": {"wins": 422, "losses": 8, "p_one_sided": 9.98e-114, "series": 4}}}
        m = build_data.parse_log("rsi4.plan", json.dumps(plan))
        self.assertEqual(m, {"plan_met": "true", "succeeded": 4, "failed": 0, "g1_wins": 422,
                             "g1_losses": 8, "g1_p": "1e-113", "g1_series": 4})

    def test_lineage_rounds_are_read_per_generation(self):
        rounds = [{"generation": g, "proposed": 600 * g, "mispredicted": 300 * g, "replayed": 0,
                   "trained_records": 300 * g, "trained": {"steps": 75 * g},
                   "frontier": {"measured": 600 * g, "on_frontier": 500 * g, "requests": 900 * g},
                   "decision": {"wins": 25, "losses": 19, "p_value": 0.2257, "champion_rate": 0.6375,
                                "candidate_rate": 0.6525, "retention_losses": 1},
                   "advanced": True, "seconds": 1800.0} for g in (1, 2)]
        m = build_data.parse_log("rsi4.dev.r1", "\n".join(json.dumps(r) for r in rounds))
        self.assertEqual((m["g2_dev_wins"], m["g2_dev_losses"], m["g2_dev_p"]), (25, 19, "0.226"))
        self.assertEqual((m["g1_mispredicted"], m["g2_trained_records"], m["g2_steps"]), (300, 600, 150))
        self.assertEqual((m["g2_on_frontier"], m["g1_advanced"], m["hours"]), (1000, "yes", "1.0"))

    def test_the_pooled_steps_are_computed_from_the_lineages_and_agree_with_the_plan_verdict(self):
        steps = {"rsi9.v.r0.g1.wins": 3, "rsi9.v.r0.g1.losses": 1, "rsi9.v.r1.g1.wins": 5,
                 "rsi9.v.r1.g1.losses": 0, "rsi9.v.r0.g1.holds": "no", "rsi9.v.r1.g1.holds": "yes"}
        m = build_data.rsi_pooled_metrics("rsi9", steps)
        self.assertEqual((m["rsi9.pooled.g1.wins"], m["rsi9.pooled.g1.losses"]), (8, 1))
        self.assertEqual(m["rsi9.pooled.g1.p"], build_data.fmt_p(build_data.upper_tail(8, 9)))

    def test_the_step_table_renders_every_lineage_with_its_counts(self):
        m = {}
        for r in ("r0", "r1"):
            for g in (1, 2, 3):
                m.update({f"rsi9.v.{r}.g{g}.right": 10 * g, f"rsi9.v.{r}.g{g}.predecessor_right": 10 * g - 10,
                          f"rsi9.v.{r}.g{g}.wins": 7, f"rsi9.v.{r}.g{g}.losses": 2,
                          f"rsi9.v.{r}.g{g}.p": "0.07", f"rsi9.v.{r}.g{g}.holds": "no"})
        tex = build_data.rsi_step_rows("rsi9", "RSI9", m)
        self.assertIn("r0 & 0 & 10 & 20 & 30", tex)
        self.assertIn("7/2", tex)
        self.assertEqual(tex.count("\\\\"), 2)

    def test_the_frontier_rule_admits_a_mostly_right_program_with_the_stated_chance(self):
        m = build_data.frontier_entry_metrics()
        self.assertEqual((m["rsi4.frontier.entry_q90"], m["rsi4.frontier.entry_q50"]), ("0.34", "0.94"))

    def test_t_quantiles_match_the_tables(self):
        self.assertAlmostEqual(build_data.t_quantile(0.975, 24), 2.0639, places=3)
        self.assertAlmostEqual(build_data.t_quantile(0.95, 24), 1.7109, places=3)

    def test_a_lineage_clears_the_clustered_test_only_if_every_step_does(self):
        m = {}
        for r, lo3 in (("r0", "-0.0190"), ("r1", "0.0100")):
            for g in (1, 2, 3):
                m.update({f"rsi9.v.{r}.g{g}.gain": "0.0607", f"rsi9.v.{r}.g{g}.gain_lo": lo3 if g == 3 else "0.0200",
                          f"rsi9.v.{r}.g{g}.gain_families": 25})
        out = build_data.rsi_cluster_metrics("rsi9", m)
        self.assertEqual((out["rsi9.v.r0.g3.excludes_zero"], out["rsi9.cluster.lineages_two_sided"]), ("no", 1))
        self.assertEqual(out["rsi9.v.r0.g3.lb_t"], "-0.0054")

    def test_an_rsi1_analysis_is_read_per_step_and_per_replicate(self):
        a = {"gen1-vs-gen0": {"per_replicate": {"0": {"from": 0, "to": 38, "wins": 38, "losses": 0}},
                              "wins": 103, "losses": 0, "sign_p": 9.86e-32, "no_replicate_falls": True}}
        m = build_data.parse_log("rsi1.arm.h", json.dumps(a))
        self.assertEqual((m["g1_wins"], m["g1_losses"], m["g1_p"], m["g1_falls"]), (103, 0, "9.9e-32", "no"))
        self.assertEqual((m["r0_g0_right"], m["r0_g1_right"]), (0, 38))
        hl = {"lineages": [{"lineage": "r3", "steps": [{"wins": 16, "losses": 0, "p": 1.5e-05, "right": 16}],
                            "succeeds": True}], "lineages_succeeding": 1, "criterion_met": False,
              "pooled": {"g1": {"wins": 149, "losses": 0, "p": 1.4e-45}}}
        m = build_data.parse_log("rsi1.arm.hl2", json.dumps(hl))
        self.assertEqual((m["r3_g1_wins"], m["r3_succeeds"], m["succeeding"], m["met"]), (16, "yes", 1, "no"))
        self.assertEqual((m["g1_wins"], m["g1_p"]), (149, "1.4e-45"))


if __name__ == "__main__":
    unittest.main()
