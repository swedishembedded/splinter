#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements architecture gates that keep large Rust
# workspaces layered as they grow. If your team needs expertise in
# dependency architecture or build-time design rules, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Specification of the architecture gate, on synthetic workspaces.

Each case builds the smallest workspace that breaks one rule and asserts the
gate names the breakage; the clean workspace asserts it stays quiet. The
gate itself reads the real workspace (architecture.py); this file never
does, so a rule cannot pass merely because today's tree happens to obey it.

Usage: python3 scripts/gates/test-architecture.py
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import architecture  # noqa: E402

TIERS = ["foundation", "storage", "domain", "adapters", "sdk", "sample"]


def workspace():
    """A clean four-crate workspace and the facts the gate is given."""
    arch = {
        "tiers": {"order": TIERS},
        "crates": {
            "core": "foundation",
            "store": "storage",
            "eval": "domain",
            "model": "adapters",
            "demo": "sample",
        },
        "brain": {"allowed": ["model"]},
        "limits": {"max_file_lines": 100},
    }
    packages = [
        pkg("core"),
        pkg("store", "core"),
        pkg("eval", "core", "store"),
        pkg("model", "eval", "brain"),
        pkg("demo", "model"),
    ]
    return arch, packages, {"crates/eval/src/lib.rs": 90}


def pkg(name, *deps, dev=()):
    return {
        "name": name,
        "deps": [{"name": d, "kind": None} for d in deps]
        + [{"name": d, "kind": "dev"} for d in dev],
    }


def run(arch, packages, files, mentions=lambda *_: True):
    return architecture.check(arch, packages, files, mentions)


class Gate(unittest.TestCase):
    def failures(self, mutate):
        arch, packages, files = workspace()
        arch, packages, files = copy.deepcopy((arch, packages, files))
        mutate(arch, packages, files)
        return "\n".join(run(arch, packages, files))

    def test_a_clean_workspace_passes(self):
        arch, packages, files = workspace()
        self.assertEqual(run(arch, packages, files), [])

    def test_every_package_is_listed(self):
        out = self.failures(lambda a, p, f: p.append(pkg("stray")))
        self.assertIn("stray: not listed", out)

    def test_every_listed_crate_exists(self):
        out = self.failures(lambda a, p, f: a["crates"].update(ghost="domain"))
        self.assertIn("ghost: listed in architecture.toml but not a workspace package", out)

    def test_a_crate_names_a_declared_tier(self):
        out = self.failures(lambda a, p, f: a["crates"].update(eval="middle"))
        self.assertIn("eval: unknown tier 'middle'", out)

    def test_a_dependency_on_a_higher_tier_fails(self):
        out = self.failures(lambda a, p, f: p[1]["deps"].append({"name": "eval", "kind": None}))
        self.assertIn("store -> eval", out)
        self.assertIn("higher tier", out)

    def test_a_same_tier_dependency_needs_a_reviewed_exception(self):
        def mutate(a, p, f):
            a["crates"]["twin"] = "domain"
            p.append(pkg("twin", "eval"))
        self.assertIn("twin -> eval", self.failures(mutate))

        def allowed(a, p, f):
            mutate(a, p, f)
            a["same_layer"] = [{"from": "twin", "to": "eval", "why": "temporary"}]
        self.assertEqual(self.failures(allowed), "")

    def test_a_same_layer_exception_must_still_be_needed(self):
        def mutate(a, p, f):
            a["same_layer"] = [{"from": "eval", "to": "store", "why": "temporary"}]
        out = self.failures(mutate)
        self.assertIn("same_layer eval -> store", out)
        self.assertIn("different tiers", out)

        def gone(a, p, f):
            a["crates"]["twin"] = "domain"
            p.append(pkg("twin"))
            a["same_layer"] = [{"from": "twin", "to": "eval", "why": "temporary"}]
        self.assertIn("no longer exists", self.failures(gone))

    def test_dev_dependencies_do_not_count_against_the_tiers(self):
        def mutate(a, p, f):
            p[1]["deps"].append({"name": "demo", "kind": "dev"})
        self.assertEqual(self.failures(mutate), "")

    def test_a_forbidden_internal_dependency_is_caught_at_any_depth(self):
        def mutate(a, p, f):
            a["forbidden"] = [{"from": "demo", "to": "store", "why": "x"}]
        out = self.failures(mutate)
        self.assertIn("demo reaches store", out)
        self.assertIn("demo -> model -> eval -> store", out)

    def test_a_forbidden_external_dependency_is_caught_through_a_crate(self):
        def mutate(a, p, f):
            a["forbidden"] = [{"from": "demo", "to": "brain", "why": "x"}]
        self.assertIn("demo reaches brain", self.failures(mutate))

    def test_a_dev_dependency_does_not_make_something_reachable(self):
        def mutate(a, p, f):
            p[0]["deps"].append({"name": "libc", "kind": "dev"})
            a["forbidden"] = [{"from": "eval", "to": "libc", "why": "x"}]
        self.assertEqual(self.failures(mutate), "")

    def test_only_an_allowed_crate_depends_on_brain(self):
        out = self.failures(lambda a, p, f: p[2]["deps"].append({"name": "brain", "kind": None}))
        self.assertIn("eval -> brain", out)

    def test_a_brain_crate_beyond_the_sdk_is_refused(self):
        out = self.failures(lambda a, p, f: p[3]["deps"].append({"name": "brain-core", "kind": None}))
        self.assertIn("model -> brain-core", out)

    def test_a_dependency_the_sources_never_name_fails(self):
        arch, packages, files = workspace()
        out = "\n".join(run(arch, packages, files, mentions=lambda pk, dep, kind: dep != "store"))
        self.assertIn("eval -> store: declared but never named", out)

    def test_surface_crates_may_depend_on_the_sdk_alone(self):
        def mutate(a, p, f):
            a["crates"]["sdk"] = "sdk"
            p.append(pkg("sdk", "model"))
            a["surface"] = {"tiers": ["sample"], "only": "sdk"}
            p[4]["deps"] = [{"name": "model", "kind": None}]
        out = self.failures(mutate)
        self.assertIn("demo -> model: a sample crate may depend only on sdk", out)

        def fine(a, p, f):
            mutate(a, p, f)
            p[4]["deps"] = [{"name": "sdk", "kind": None}, {"name": "serde", "kind": None}]
        self.assertEqual(self.failures(fine), "")

    def test_a_source_file_over_the_limit_fails_unless_ratcheted(self):
        def over(a, p, f):
            f["crates/eval/src/big.rs"] = 150
        self.assertIn("crates/eval/src/big.rs: 150 lines, over the 100-line limit", self.failures(over))

        def ratcheted(a, p, f):
            over(a, p, f)
            a["allow"] = {"large_file": [{"path": "crates/eval/src/big.rs", "lines": 150, "why": "split soon"}]}
        self.assertEqual(self.failures(ratcheted), "")

        def grown(a, p, f):
            ratcheted(a, p, f)
            f["crates/eval/src/big.rs"] = 151
        self.assertIn("grew past its recorded 150 lines", self.failures(grown))

        def stale(a, p, f):
            ratcheted(a, p, f)
            f["crates/eval/src/big.rs"] = 90
        self.assertIn("is back under the limit; delete its allow.large_file entry", self.failures(stale))


if __name__ == "__main__":
    unittest.main(verbosity=1)
