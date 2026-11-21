# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""The spec of scripts/cluster/rank-env: the site's files are sourced inside
the task, and a rank is pinned to the CPUs of the NUMA node its card's PCI
device sits on, as a stand-in sysfs and nvidia-smi say.

    python3 -I scripts/cluster/test_rank_env.py
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RANK_ENV = Path(__file__).resolve().parent / "rank-env"


def tree(root: Path) -> tuple[Path, Path]:
    """A sysfs with card 1 on NUMA node 1 (CPUs 2-3) and card 2 on a node with
    no CPUs, and an nvidia-smi that names their bus ids as the real one does."""
    sysfs = root / "sys"
    for bus, node in [("0009:01:00.0", 0), ("0019:01:00.0", 1), ("0029:01:00.0", 5)]:
        d = sysfs / "bus/pci/devices" / bus
        d.mkdir(parents=True)
        (d / "numa_node").write_text(f"{node}\n")
    for node, cpus in [(0, "0-1"), (1, "2-3"), (5, "")]:
        d = sysfs / f"devices/system/node/node{node}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(f"{cpus}\n")
    smi = root / "nvidia-smi"
    smi.write_text('#!/bin/sh\nwhile [ "$1" != -i ]; do shift; done\n'
                   'case "$2" in 0) echo 00000009:01:00.0 ;; 1) echo 00000019:01:00.0 ;; 2) echo 00000029:01:00.0 ;; esac\n')
    smi.chmod(0o755)
    return sysfs, smi


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([str(RANK_ENV), *args], capture_output=True, text=True, timeout=30)


class RankEnv(unittest.TestCase):
    def test_a_site_file_is_sourced_inside_the_task(self):
        with tempfile.TemporaryDirectory() as d:
            site = Path(d) / "site.sh"
            site.write_text("export NCCL_NET=Libfabric\n")
            r = run(["--env", str(site), "--", "sh", "-c", 'echo "$NCCL_NET"'])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "Libfabric")

    def test_a_rank_runs_on_its_cards_numa_cpus(self):
        with tempfile.TemporaryDirectory() as d:
            sysfs, smi = tree(Path(d))
            allowed = os.sched_getaffinity(0)
            if not {2, 3} <= allowed:
                self.fail(f"this test pins to CPUs 2-3, which this process may not use: {sorted(allowed)}")
            r = run(["--card", "1", "--sysfs", str(sysfs), "--nvidia-smi", str(smi), "--",
                     "sh", "-c", "grep Cpus_allowed_list /proc/self/status"])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("2-3", r.stdout)

    def test_a_card_on_a_node_without_cpus_runs_unpinned_and_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            sysfs, smi = tree(Path(d))
            r = run(["--card", "2", "--sysfs", str(sysfs), "--nvidia-smi", str(smi), "--", "true"])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("running unpinned", r.stderr)

    def test_a_bad_card_or_missing_command_is_refused(self):
        self.assertEqual(run(["--card", "x", "--", "true"]).returncode, 2)
        self.assertEqual(run(["--card", "0"]).returncode, 2)
        self.assertEqual(run(["--env", "/nonexistent", "--", "true"]).returncode, 2)


if __name__ == "__main__":
    sys.exit(unittest.main())
