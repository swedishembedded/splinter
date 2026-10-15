#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible randomness in
# machine-learning pipelines for its clients. If your team needs expertise in
# seeding, experiment provenance or statistically honest evaluation, you can
# procure our services by sending an email to info@swedishembedded.com.

"""Specification of the seed-literal gate, on synthetic sources.

Each case feeds the gate the smallest source that shows one rule. The gate
itself reads the real tree (seed-literals.py); this file never does, so a rule
cannot pass merely because today's tree happens to obey it.

Usage: python3 scripts/gates/test-seed-literals.py
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("seed_literals", HERE / "seed-literals.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def production(src, path="crates/demo/src/lib.rs"):
    """Symbols of the production hits in a Rust source."""
    return sorted(h.symbol for h in gate.scan_rust(path, src) if not h.test)


def tests(src):
    return [h for h in gate.scan_rust("crates/demo/src/lib.rs", src) if h.test]


class Detection(unittest.TestCase):
    def test_a_literal_default_is_found_in_each_shape(self):
        src = """
pub struct Opts { pub seed: u64 }
impl Default for Opts {
    fn default() -> Self { Self { seed: 1337 } }
}
fn parse(a: &[String]) -> u64 {
    let train_seed = 7;
    let s = flag(a, "--seed").map_or(Ok(1), |s| s.parse())?;
    let t = u64_or("--noise-seed", 3);
    let r = Rng::new(0x5EED);
    let b = Builder::new().seed(5);
    s
}
pub const REPLAY_SEED: u64 = 0;
"""
        self.assertEqual(
            production(src),
            sorted([
                "Opts::default::seed",
                "parse::train_seed",
                "parse::map_or",
                "parse::--noise-seed",
                "parse::Rng::new",
                "parse::.seed()",
                "REPLAY_SEED",
            ]),
        )

    def test_a_clap_default_is_found_by_the_field_it_belongs_to(self):
        src = """
struct Args {
    /// Seed of the run.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long, default_value_t = 4)]
    steps: u64,
}
"""
        self.assertEqual(production(src), ["Args::seed"])

    def test_a_seed_parameter_default_is_found(self):
        src = """
fn spec() -> Vec<ParamSpec> {
    vec![ParamSpec::new("seed", Kind::Int)
        .default(json!(0))]
}
"""
        self.assertEqual(production(src), ["spec::param:seed"])

    def test_drawn_or_passed_through_seeds_are_not_literals(self):
        src = """
pub struct Opts { pub seed: u64, pub seeds: Vec<u64> }
fn run(o: &Opts, seed: u64) -> u64 {
    let rng = SplitMix64::new(seed);
    let derived = seed ^ 0x9E37;
    o.seed == 3 || seed_count.len() > 2
}
"""
        self.assertEqual(production(src), [])

    def test_comments_and_strings_do_not_hit(self):
        src = '''
// let seed = 4;
/// Defaults to `seed: 1`.
fn f() -> &'static str { "seed: 9 { and a brace" }
fn g() { let seed = 2; }
'''
        self.assertEqual(production(src), ["g::seed"])

    def test_a_symbol_survives_the_line_moving(self):
        a = production("fn f() {\n    let seed = 1;\n}\n")
        b = production("\n\n// moved\nfn f() {\n\n    let seed = 1;\n}\n")
        self.assertEqual(a, b)


class TestContext(unittest.TestCase):
    def test_test_code_is_not_production(self):
        src = """
fn f() { let seed = 1; }
#[cfg(test)]
mod tests {
    #[test]
    fn t() { let seed = 2; }
}
#[test]
fn t2() { let seed = 3; }
"""
        self.assertEqual(production(src), ["f::seed"])
        self.assertEqual(len(tests(src)), 2)

    def test_code_after_a_test_module_is_production_again(self):
        src = "#[cfg(test)]\nmod tests {\n    fn t() { let seed = 2; }\n}\nfn f() { let seed = 1; }\n"
        self.assertEqual(production(src), ["f::seed"])

    def test_a_test_reason_comment_is_recognised(self):
        src = (
            "#[test]\nfn t() {\n"
            "    // seed: the fixture is derived from this value\n"
            "    let a = 1; let seed = 2;\n"
            "    let seed = 3; // seed: a golden digest\n"
            "\n    let seed = 4;\n}\n"
        )
        self.assertEqual([h.marked for h in tests(src)], [True, True, False])

    def test_files_under_tests_and_cfg_test_module_files_are_test_code(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "crates/demo/src").mkdir(parents=True)
            (root / "crates/demo/tests").mkdir(parents=True)
            (root / "crates/demo/src/lib.rs").write_text("#[cfg(test)]\nmod fixtures;\nfn f() { let seed = 1; }\n")
            (root / "crates/demo/src/fixtures.rs").write_text("pub fn s() { let seed = 2; }\n")
            (root / "crates/demo/tests/it.rs").write_text("fn s() { let seed = 3; }\n")
            hits = gate.scan_tree(root, [
                "crates/demo/src/lib.rs", "crates/demo/src/fixtures.rs", "crates/demo/tests/it.rs",
            ])
            self.assertEqual(sorted(h.path for h in hits if not h.test), ["crates/demo/src/lib.rs"])


class OtherLanguages(unittest.TestCase):
    def test_python_and_make_defaults_are_found(self):
        py = 'ap.add_argument(' '"--seed", type=int, default=20261010)\nSEED = 7\nrng = Random(args.seed)\n'
        self.assertEqual(sorted(h.symbol for h in gate.scan_other("a.py", py)), ["--seed", "SEED"])
        self.assertEqual([h.symbol for h in gate.scan_other("Makefile", "SEED ?= 1337\nSEED ?=\n")], ["SEED"])

    def test_python_seeding_calls_with_a_literal_are_found_by_enclosing_def(self):
        py = (
            "SEED = 20261006\n"
            "def a():\n"
            "    rng = np.random.default_rng(7)\n"
            "    old = np.random.RandomState(0)\n"
            "    torch.manual_seed(42)\n"
            "    random.seed(3)\n"
            "def b(seed):\n"
            "    return np.random.default_rng(seed), np.random.RandomState(seed + 1), random.Random(seed)\n"
            "top = np.random.default_rng(1)\n"
        )
        self.assertEqual(
            sorted(h.symbol for h in gate.scan_other("a.py", py)),
            sorted([
                "SEED",
                "a:default_rng", "a:RandomState", "a:manual_seed", "a:random.seed",
                "-:default_rng",
            ]),
        )

    def test_python_seed_parameters_keywords_and_variables_with_a_literal_are_found(self):
        py = (
            "def analyse(data, bootstrap, seed=1):\n"
            "    seed = 20261006\n"
            "    return run(data, noise_seed=3, seed=seed, same=seed == 4)\n"
        )
        self.assertEqual(
            sorted(h.symbol for h in gate.scan_other("a.py", py)),
            ["analyse:noise_seed", "analyse:seed", "analyse:seed"],
        )

    def test_python_test_files_are_test_code_and_a_hash_reason_marks_them(self):
        py = "def t():\n    a = np.random.RandomState(1)\n    b = np.random.RandomState(2)  # seed: pinned digest\n"
        for path in ("samples/x/test_a.py", "crates/y/tests/fixtures/make.py"):
            hits = gate.scan_other(path, py)
            self.assertEqual([(h.test, h.marked) for h in hits], [(True, False), (True, True)], path)
        self.assertEqual([h.test for h in gate.scan_other("samples/x/a.py", py)], [False, False])


class Ratchet(unittest.TestCase):
    def hits(self):
        return gate.scan_rust("crates/demo/src/lib.rs", "fn f() { let seed = 1; }\n")

    def allow(self, text):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "demo.allow").write_text(text)
            return gate.read_allow(d)

    def test_an_allowlisted_site_passes(self):
        entries, problems = self.allow("crates/demo/src/lib.rs\tf::seed\tworld: the demo world\n")
        self.assertEqual(problems, [])
        self.assertEqual(gate.check(self.hits(), entries), [])

    def test_a_new_literal_fails_and_names_the_site(self):
        out = "\n".join(gate.check(self.hits(), {}))
        self.assertIn("crates/demo/src/lib.rs:1: seed literal f::seed", out)

    def test_a_test_literal_without_a_reason_fails_and_names_the_site(self):
        src = "#[test]\nfn t() {\n    let seed = 4;\n}\n"
        out = "\n".join(gate.check(tests(src), {}))
        self.assertIn("crates/demo/src/lib.rs:3: test seed literal t::seed", out)
        self.assertIn("// seed:", out)

    def test_a_test_literal_with_a_reason_passes_without_an_allowlist_entry(self):
        src = "#[test]\nfn t() {\n    let seed = 4; // seed: a golden digest\n}\n"
        self.assertEqual(gate.check(tests(src), {}), [])

    def test_a_python_test_literal_without_a_reason_fails(self):
        hits = gate.scan_other("samples/x/test_a.py", "def t():\n    a = np.random.RandomState(1)\n")
        self.assertIn("# seed:", "\n".join(gate.check(hits, {})))

    def test_a_stale_entry_fails(self):
        entries, _ = self.allow("crates/demo/src/lib.rs\tgone::seed\tsalt: x\n")
        out = "\n".join(gate.check([], entries))
        self.assertIn("gone::seed matches no seed literal", out)

    def test_a_reason_needs_a_known_prefix(self):
        _, problems = self.allow("crates/demo/src/lib.rs\tf::seed\tbecause\n")
        self.assertIn("reason must start with", "\n".join(problems))

    def test_a_todo_reason_is_accepted(self):
        entries, problems = self.allow("crates/demo/src/lib.rs\tf::seed\ttodo: drawn once the stage records it\n")
        self.assertEqual(problems, [])
        self.assertEqual(gate.check(self.hits(), entries), [])

    def test_a_malformed_or_duplicate_entry_is_a_problem(self):
        _, problems = self.allow("only two\tfields\n")
        self.assertIn("expected path<TAB>symbol<TAB>reason", "\n".join(problems))
        _, problems = self.allow("p\ts\tsalt: a\np\ts\tsalt: b\n")
        self.assertIn("listed twice", "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
