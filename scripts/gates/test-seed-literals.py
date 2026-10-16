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


def hidden(src, path="crates/demo/src/lib.rs"):
    """Symbols of the production hits in a Rust source, with its own parameter index."""
    index = gate.seed_param_index([src])
    return sorted(h.symbol for h in gate.scan_rust(path, src, seed_params=index) if not h.test)


class HiddenDefaults(unittest.TestCase):
    """Seeds the field-and-assignment shapes cannot see."""

    def test_a_seed_constructor_with_a_literal_is_found(self):
        src = """
fn unrecorded() -> Seed { Seed::new(0) }
fn noise() { let a = Mt19937::new(5); let b = StdRng::seed_from_u64(1); let c = ChaCha8Rng::from_seed(3); }
fn pool() { let p = Philox4x32::new(9); }
fn drawn(s: u64) -> Seed { Seed::new(s) }
"""
        self.assertEqual(
            hidden(src),
            sorted([
                "unrecorded::Seed::new",
                "noise::Mt19937::new", "noise::StdRng::seed_from_u64", "noise::ChaCha8Rng::from_seed",
                "pool::Philox4x32::new",
            ]),
        )

    def test_a_function_that_returns_a_fixed_seed_is_found(self):
        src = """
fn default_seed() -> u64 { 7 }
fn other() -> u64 {
    42
}
fn named_by_serde() -> u64 { 3 }
#[derive(Deserialize)]
struct Opts {
    #[serde(default = "named_by_serde")]
    seed: u64,
}
fn count() -> usize { 4 }
fn drawn_seed() -> u64 { draw() }
"""
        self.assertEqual(
            hidden(src),
            sorted(["default_seed::returns", "named_by_serde::returns"]),
        )

    def test_a_literal_passed_where_a_function_takes_a_seed_is_found(self):
        src = """
const PREREGISTERED: u64 = 5;
const GROUPS: usize = 10;
fn run(n: usize, seed: u64) -> u64 { n as u64 + seed }
fn cv(data: &Data, arm: &str, base_seed: u64) {}
impl Spec { fn build(&self, name: &str, seed: u64) {} }
fn main2(s: u64, data: &Data) {
    run(3, 7);
    run(3, s);
    run(3, PREREGISTERED);
    run(3, GROUPS);
    run(3, 7, 9);
    cv(data, "a",
       0x10);
    spec.build("x", 5);
    spec.build("x", s);
    run(3, s + 1);
}
"""
        self.assertEqual(
            hidden(src),
            sorted(["main2::run#1", "main2::run#1", "main2::cv#2", "main2::build#1"]),
        )

    def test_a_range_of_literal_seeds_spanning_lines_is_found(self):
        src = "fn report() {\n    let by_seed = (1..=SEEDS)\n        .map(|seed| runs(seed))\n        .collect();\n}\n"
        self.assertEqual(hidden(src), ["report::seed-range"])

    def test_a_range_of_seeds_in_a_test_is_a_sweep_not_a_default(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "crates/demo/tests").mkdir(parents=True)
            (root / "crates/demo/tests/it.rs").write_text("fn s() { for seed in 0..9 { go(seed); } }\n")
            self.assertEqual(gate.scan_tree(root, ["crates/demo/tests/it.rs"]), [])

    def test_a_literal_seed_argument_in_a_test_is_read_only_on_request(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "crates/demo/tests").mkdir(parents=True)
            (root / "crates/demo/Cargo.toml").write_text("")
            (root / "crates/demo/tests/it.rs").write_text("fn run(seed: u64) {}\nfn t() { run(4); }\n")
            files = ["crates/demo/tests/it.rs", "crates/demo/Cargo.toml"]
            self.assertEqual(gate.scan_tree(root, files), [])
            self.assertEqual([h.symbol for h in gate.scan_tree(root, files, positional_tests=True)], ["t::run#0"])

    def test_a_name_defined_with_the_seed_in_two_places_is_not_resolved(self):
        src = """
mod a { pub fn make(x: u64, seed: u64) {} }
mod b { pub fn make(seed: u64, x: u64) {} }
fn go() { make(1, 2); }
"""
        self.assertEqual(hidden(src), [])

    def test_a_loop_or_range_over_literal_seeds_is_found(self):
        src = """
fn ensemble(members: u64) {
    for seed in 1..=members { train(seed); }
    for i in 0..members { work(i); }
}
fn report() {
    let runs = (1..=SEEDS_REPORTED).map(|seed| cv_runs(seed)).collect();
    let ok = (0..n).map(|i| i + 1);
}
"""
        self.assertEqual(hidden(src), sorted(["ensemble::seed-range", "report::seed-range"]))

    def test_python_defaults_in_every_shape_are_found(self):
        py = (
            "@dataclass\n"
            "class Spec:\n"
            "    seed: int = 7\n"
            "def forecast(series, seed: int = 0, n=3):\n"
            "    req = {\"seed\": 7, \"n\": 2}\n"
            "    s = os.environ.get(\"RUN_SEED\", \"11\")\n"
            "    t = seed if seed is not None else 5\n"
            "    u = args.seed or 9\n"
            "    return req\n"
        )
        self.assertEqual(
            sorted(h.symbol for h in gate.scan_other("a.py", py)),
            sorted([
                "Spec:seed",
                "forecast:seed",
                "forecast:seed", "forecast:seed-fallback", "forecast:seed-fallback",
                "forecast:RUN_SEED",
            ]),
        )

    def test_shell_flags_and_defaults_are_found_by_enclosing_function(self):
        sh = (
            "run() {\n"
            "    brain serve --seed 7 --temp 1\n"
            "    other --noise-seed=3\n"
            "}\n"
            "SEED=${SEED:-11}\n"
            "seq 1 3 | while read -r s; do tool --seed \"$s\"; done\n"
            "tool --seed \"$S\" --steps 4\n"
        )
        self.assertEqual(
            sorted(h.symbol for h in gate.scan_other("run.sh", sh)),
            sorted(["run:--seed", "run:--noise-seed", "-:SEED", "-:seq-seed"]),
        )

    def test_a_makefile_recipe_is_found_by_target(self):
        mk = "SEED ?= 5\nfixtures: deps\n\tpython3 dump.py --seed 7 --out x\n\tpython3 dump.py --seed $(SEED)\n"
        self.assertEqual(
            sorted(h.symbol for h in gate.scan_other("Makefile", mk)),
            sorted(["SEED", "fixtures:--seed"]),
        )

    def test_a_documented_command_with_a_literal_seed_needs_the_word_reproduce_beside_it(self):
        md = (
            "Run it:\n\n```bash\n"
            "tool synth --seed 7\n"
            "```\n\n"
            "To reproduce the table:\n\n```bash\n"
            "tool synth --seed 3\n"
            "```\n\n"
            "Prose may say `--seed 9` freely.\n"
        )
        self.assertEqual([h.symbol for h in gate.scan_other("README.md", md)], ["doc:--seed 7"])


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

    def test_a_reason_covers_the_run_of_lines_it_heads_up_to_the_next_blank_line(self):
        src = (
            "// seed: two arbitrary seeds; only that\n"
            "// the same one repeats matters.\n"
            "const FIRST: Seed = Seed::new(1);\n"
            "const OTHER: Seed = Seed::new(2);\n"
            "\n"
            "const THIRD: Seed = Seed::new(3);\n"
        )
        self.assertEqual([h.marked for h in tests("#[cfg(test)]\nmod t {\n" + src + "}\n")], [True, True, False])

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
    def test_a_constant_passed_as_a_seed_needs_no_second_entry_when_it_is_listed_itself(self):
        src = "const FIXED: u64 = 5;\nfn run(seed: u64) {}\nfn go() { run(FIXED); }\n"
        idx = gate.seed_param_index([src])
        hits = gate.scan_rust("crates/demo/src/lib.rs", src, seed_params=idx)
        self.assertIn("go::run#0", [h.symbol for h in hits])
        listed = {("crates/demo/src/lib.rs", "FIXED"): "world: x"}
        out = "\n".join(gate.check(hits, listed))
        self.assertNotIn("go::run#0", out)

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

    def test_a_reason_that_says_any_seed_will_do_is_no_reason(self):
        src = "#[test]\nfn t() {\n    // seed: Any value shows it is read back\n    let seed = 4;\n}\n"
        out = "\n".join(gate.check(tests(src), {}))
        self.assertIn("crates/demo/src/lib.rs:4: test seed literal t::seed", out)
        self.assertIn("drop the literal", out)
        py = gate.scan_other("samples/x/test_a.py", "def t():\n    f(seed=1)  # seed: any value; the folds matter\n")
        self.assertIn("# seed:", "\n".join(gate.check(py, {})))
        kept = "#[test]\nfn t() {\n    // seed: a fixed pair; a drawn one could resample alike\n    let seed = 4;\n}\n"
        self.assertEqual(gate.check(tests(kept), {}), [])

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
