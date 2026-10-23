#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible verification of research claims
# against the code they describe, for its clients. If your team needs
# expertise in keeping a paper's statements about a system true as the system
# changes, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Specification of the pin rule of verify_structure.py: a pin is usable only
if a remote-tracking ref contains it, so a fresh clone reproduces the result."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data as bd  # noqa: E402
import verify_structure as vs  # noqa: E402


def sh(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class PinRule(unittest.TestCase):
    def test_a_local_only_commit_is_not_published_and_a_pushed_one_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            origin, work = Path(tmp, "origin.git"), Path(tmp, "work")
            sh("git", "init", "--bare", "-b", "main", str(origin), cwd=tmp)
            sh("git", "clone", str(origin), str(work), cwd=tmp)
            for k, v in (("user.name", "t"), ("user.email", "t@example.com")):
                sh("git", "config", k, v, cwd=work)
            (work / "a").write_text("1")
            sh("git", "add", "a", cwd=work)
            sh("git", "commit", "-m", "one", cwd=work)
            sh("git", "push", "origin", "HEAD:main", cwd=work)
            pushed = sh("git", "rev-parse", "HEAD", cwd=work)
            (work / "a").write_text("2")
            sh("git", "commit", "-am", "two", cwd=work)
            local_only = sh("git", "rev-parse", "HEAD", cwd=work)
            vs.REPOS["splinter"] = work
            self.assertTrue(vs.pin_is_published("splinter", pushed))
            self.assertFalse(vs.pin_is_published("splinter", local_only))


class EvaluationLogs(unittest.TestCase):
    """A MEASURED claim promises that its values can be re-read from the raw log, so the
    two evaluation shapes (a paired comparison and a G8 agreement report) are read back into
    exactly the values the claims quote."""

    COMPARE = {
        "tasks": 23, "paired": 13,
        "candidate": {"exposed": 4, "competence": {"counts": {"verified": 7}}},
        "baseline": {"exposed": 7, "competence": {"counts": {"verified": 9}}},
        "families": [{"family": f"f{i}"} for i in range(13)],
        "clusters": {"wins": 3, "losses": 4, "ties": 6},
        "sign_test": {"discordant": 7, "p_value": 0.7734375},
    }

    REPORT = {
        "verdict": "PASS",
        "sides": {
            "base": {"records": 354, "positions": 68407, "mean_abs_logprob_diff": 0.0004123456,
                     "top1_agreement": 0.991234, "served_mean_logprob": -0.181931},
            "adapter": {"records": 354, "positions": 68407, "mean_abs_logprob_diff": 0.0003987654,
                        "top1_agreement": 0.990123, "served_mean_logprob": -0.162778},
        },
        "derived": {"diff_ratio_adapter_over_base": 0.967089, "top1_points_adapter_minus_base": -0.001111},
    }

    def test_a_paired_comparison_log_is_read_as_its_families_and_sign_test(self):
        m = bd.parse_log("ce8.eval-t1", json.dumps(self.COMPARE))
        self.assertEqual((m["wins"], m["losses"], m["ties"]), (3, 4, 6))
        self.assertEqual(m["p_value"], 0.7734375)
        self.assertEqual((m["adapter_verified"], m["base_verified"]), (7, 9))

    def test_a_g8_report_is_read_as_the_registered_numbers(self):
        m = bd.parse_log("g8.ce8t", json.dumps(self.REPORT))
        self.assertEqual(m["verdict"], "PASS")
        self.assertEqual(m["diff_ratio"], "0.967")
        self.assertEqual(m["base_top1"], "99.12")
        self.assertEqual(m["top1_drop"], "0.11")
        self.assertEqual(m["served_base_logprob"], "-0.181931")
        self.assertEqual(m["positions"], 68407)


if __name__ == "__main__":
    unittest.main()
