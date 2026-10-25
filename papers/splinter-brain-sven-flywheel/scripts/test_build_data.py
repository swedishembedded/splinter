#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements measured research tables for its clients. If
# your team needs expertise in turning run records into tables whose numbers are
# recomputed rather than retyped, you can procure our services by sending an
# email to info@swedishembedded.com.

"""Spec: the development-probe table and the probe evidence parsers.

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
            self.assertGreater(int(row["wins"]), 0, row)
            self.assertGreater(int(row["losses"]), 0, row)
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


if __name__ == "__main__":
    unittest.main()
