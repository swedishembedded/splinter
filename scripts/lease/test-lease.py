#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for shared-GPU and shared-build-host
# scheduling for its clients. If your team needs expertise in running many
# training and build lanes on one leased node then you can procure our
# services by sending an email to info@swedishembedded.com.

"""Specification of the standing lease kit (scripts/lease).

Usage: python3 -I scripts/lease/test-lease.py
"""
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

LEASE = Path(__file__).resolve().parent


def run(args, env=None, **kw):
    return subprocess.run(
        [str(a) for a in args], capture_output=True, text=True, env=env, **kw
    )


def fake_nvidia_smi(directory: Path, listing: str) -> dict:
    """Put an nvidia-smi on PATH that prints `listing` for the compute-apps query."""
    exe = directory / "nvidia-smi"
    exe.write_text(f"#!/bin/sh\ncat <<'EOF'\n{listing}\nEOF\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["PATH"] = f"{directory}:{env['PATH']}"
    return env


class SlotTest(unittest.TestCase):
    def test_single_slot_serialises_concurrent_holders(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log"
            body = f"echo start >> {log}; sleep 0.6; echo end >> {log}"
            procs = [
                subprocess.Popen(
                    [str(LEASE / "slot"), "--lock-dir", d, "--count", "1",
                     "--", "sh", "-c", body]
                )
                for _ in range(2)
            ]
            for p in procs:
                self.assertEqual(p.wait(timeout=30), 0)
            self.assertEqual(log.read_text().split(), ["start", "end"] * 2)

    def test_two_slots_run_in_parallel(self):
        with tempfile.TemporaryDirectory() as d:
            t0 = time.monotonic()
            procs = [
                subprocess.Popen(
                    [str(LEASE / "slot"), "--lock-dir", d, "--count", "2",
                     "--", "sleep", "1"]
                )
                for _ in range(2)
            ]
            for p in procs:
                self.assertEqual(p.wait(timeout=30), 0)
            self.assertLess(time.monotonic() - t0, 1.8)

    def test_exit_status_of_command_is_propagated(self):
        with tempfile.TemporaryDirectory() as d:
            r = run([LEASE / "slot", "--lock-dir", d, "--count", "1",
                     "--", "sh", "-c", "exit 7"])
            self.assertEqual(r.returncode, 7)


class GpuTest(unittest.TestCase):
    def test_exclusive_refuses_when_another_context_exists(self):
        with tempfile.TemporaryDirectory() as d:
            env = fake_nvidia_smi(Path(d), "4242, python, 900 MiB")
            r = run([LEASE / "gpu", "--lock", Path(d) / "gpu.lock", "--exclusive",
                     "--", "true"], env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("4242", r.stderr)

    def test_exclusive_runs_on_an_idle_gpu_and_records_contexts(self):
        with tempfile.TemporaryDirectory() as d:
            env = fake_nvidia_smi(Path(d), "")
            r = run([LEASE / "gpu", "--lock", Path(d) / "gpu.lock", "--exclusive",
                     "--", "sh", "-c", "echo ran"], env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("ran", r.stdout)
            self.assertIn("other_contexts_start=0", r.stderr)
            self.assertIn("other_contexts_end=0", r.stderr)

    def test_shared_run_records_contexts_without_refusing(self):
        with tempfile.TemporaryDirectory() as d:
            env = fake_nvidia_smi(Path(d), "4242, python, 900 MiB")
            r = run([LEASE / "gpu", "--lock", Path(d) / "gpu.lock",
                     "--", "true"], env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("other_contexts_start=1", r.stderr)
            self.assertIn("other_contexts_end=1", r.stderr)

    def fake_splinter(self, d: Path) -> Path:
        """A program named splinter that prints the arguments it was given."""
        path = d / "splinter"
        path.write_text('#!/bin/sh\necho "args=[$*]"\n')
        path.chmod(0o755)
        return path

    def test_splinter_is_told_how_many_foreign_contexts_there_were(self):
        with tempfile.TemporaryDirectory() as d:
            env = fake_nvidia_smi(Path(d), "4242, python, 900 MiB")
            r = run([LEASE / "gpu", "--lock", Path(d) / "gpu.lock",
                     "--", self.fake_splinter(Path(d)), "status"], env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("args=[--gpu-contexts 1 status]", r.stdout)

    def test_splinter_is_told_nothing_when_contexts_cannot_be_counted(self):
        with tempfile.TemporaryDirectory() as d:
            env = fake_nvidia_smi(Path(d), "")
            (Path(d) / "nvidia-smi").write_text("#!/bin/sh\nexit 9\n")
            r = run([LEASE / "gpu", "--lock", Path(d) / "gpu.lock",
                     "--", self.fake_splinter(Path(d)), "status"], env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("args=[status]", r.stdout)

    def test_another_command_is_run_as_given(self):
        with tempfile.TemporaryDirectory() as d:
            env = fake_nvidia_smi(Path(d), "4242, python, 900 MiB")
            r = run([LEASE / "gpu", "--lock", Path(d) / "gpu.lock",
                     "--", "sh", "-c", "echo args=[$*]", "sh", "x"], env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("args=[x]", r.stdout)

    def test_lock_serialises_and_status_propagates(self):
        with tempfile.TemporaryDirectory() as d:
            env = fake_nvidia_smi(Path(d), "")
            r = run([LEASE / "gpu", "--lock", Path(d) / "gpu.lock",
                     "--", "sh", "-c", "exit 5"], env=env)
            self.assertEqual(r.returncode, 5)


class CardTest(unittest.TestCase):
    """`card` reserves VRAM shares on one of several GPUs (11 GiB each)."""

    def card(self, d, args, body):
        return run([LEASE / "card", "--lock-dir", d, *args, "--", "sh", "-c", body])

    def test_the_command_is_told_its_card_and_the_device_flag(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.card(d, ["--card", "1", "--gb", "11"], 'echo "$CARD|$BRAIN_DEVICE_FLAG"')
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "1|--device gpu1")

    def test_cards_lock_independently_and_a_full_card_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            holder = subprocess.Popen(
                [str(LEASE / "card"), "--lock-dir", d, "--card", "0", "--gb", "88",
                 "--", "sleep", "5"],
                stderr=subprocess.DEVNULL,
            )
            try:
                time.sleep(0.5)
                busy = self.card(d, ["--card", "0", "--gb", "11"], "true")
                self.assertEqual(busy.returncode, 75, busy.stderr)
                other = self.card(d, ["--card", "1", "--gb", "88"], "true")
                self.assertEqual(other.returncode, 0, other.stderr)
                anywhere = self.card(d, ["--card", "any", "--gb", "11"], 'echo "$CARD"')
                self.assertEqual(anywhere.stdout.strip(), "1")
            finally:
                holder.kill()
                holder.wait()

    def test_shares_are_rounded_up_and_the_command_status_propagates(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.card(d, ["--card", "0", "--gb", "89"], "true").returncode, 2)
            self.assertEqual(self.card(d, ["--card", "0", "--gb", "1"], "exit 7").returncode, 7)


class DoctorTest(unittest.TestCase):
    def test_prints_cgroup_cap_lease_end_and_disk(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "cg").mkdir()
            (d / "cg" / "memory.max").write_text("17179869184\n")
            (d / "targets" / "lane-a").mkdir(parents=True)
            (d / "targets" / "lane-a" / "blob").write_bytes(b"x" * 4096)
            env = fake_nvidia_smi(d, "")
            env["SLURM_JOB_END_TIME"] = "1893456000"
            r = run([LEASE / "doctor", "--cgroup-dir", d / "cg",
                     "--lock", d / "gpu.lock", "--targets", d / "targets",
                     "--disk", d], env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("memory_cap_bytes=17179869184", r.stdout)
            self.assertIn("lease_end=1893456000", r.stdout)
            self.assertIn("disk_free_kib=", r.stdout)
            self.assertIn("gpu_contexts=0", r.stdout)
            self.assertIn("lock_holder=none", r.stdout)
            self.assertIn("target lane-a", r.stdout)

    def test_unbounded_cgroup_and_missing_lease_are_reported(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "cg").mkdir()
            (d / "cg" / "memory.max").write_text("max\n")
            env = fake_nvidia_smi(d, "")
            env.pop("SLURM_JOB_END_TIME", None)
            r = run([LEASE / "doctor", "--cgroup-dir", d / "cg",
                     "--lock", d / "gpu.lock", "--disk", d], env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("memory_cap_bytes=unlimited", r.stdout)
            self.assertIn("lease_end=unknown", r.stdout)


class WorktreeTest(unittest.TestCase):
    def test_add_list_remove_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            repo = d / "repo"
            git = lambda *a, cwd=repo: subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a],
                cwd=cwd, check=True, capture_output=True, text=True)
            repo.mkdir()
            git("init", "-q", "-b", "main")
            (repo / "f").write_text("x")
            git("add", "f")
            git("commit", "-q", "-m", "init")
            common = ["--repo", repo, "--worktrees", d / "wt", "--targets", d / "tg"]
            r = run([LEASE / "worktree", "add", "lane1", *common, "--base", "main",
                     "--date", "2026-01-02"])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((d / "wt" / "lane1" / "f").exists())
            self.assertTrue((d / "tg" / "lane1").is_dir())
            r = run([LEASE / "worktree", "list", *common])
            self.assertIn("lane1", r.stdout)
            self.assertIn("lane1-2026-01-02", r.stdout)
            r = run([LEASE / "worktree", "remove", "lane1", *common])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse((d / "wt" / "lane1").exists())
            self.assertFalse((d / "tg" / "lane1").exists())

    def test_add_refuses_an_existing_lane(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "wt" / "lane1").mkdir(parents=True)
            r = run([LEASE / "worktree", "add", "lane1", "--repo", d,
                     "--worktrees", d / "wt", "--targets", d / "tg"])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("exists", r.stderr)


class EnvTest(unittest.TestCase):
    def test_sources_build_env_and_pins_cargo_settings(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "build-env.sh").write_text("export LEASE_PROBE=sourced\n")
            r = run(["sh", "-c",
                     f"set -- {d / 'build-env.sh'} {d / 'tgt'}; . {LEASE / 'env.sh'}; "
                     "echo $LEASE_PROBE $CARGO_INCREMENTAL $CARGO_TARGET_DIR"])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.split(), ["sourced", "0", str(d / "tgt")])


if __name__ == "__main__":
    unittest.main()
