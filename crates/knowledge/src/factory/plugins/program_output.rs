// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements procedurally generated, interpreter-graded
// training tasks for code-reading skills, for its clients. If your team
// needs expertise in verifiable program-understanding environments for
// reinforcement learning, you can procure our services by sending an email
// to info@swedishembedded.com.

//! Predict what a program prints: the stateless verifiable environment.
//!
//! A generated program (a handful of blocks: arithmetic chains, loops, lists,
//! strings, small functions, closures) is shown in the instruction, and the
//! solver writes the exact standard output to `answer.txt`. The verifier
//! holds a pinned copy of the program in `tests/` and runs it in the real
//! interpreter: the interpreter is the oracle, and no expected answer is
//! stored anywhere in the folder.
//!
//! The same program is rendered in Lua 5.1 and in Python 3 from one
//! description. Admission runs both interpreters and a small evaluator of the
//! description and requires all three to print the same lines, so a
//! disagreement between the generator's idea of the program and the
//! interpreter (an overflow, a rounding, a language difference) is found
//! before a model is trained on it.
//!
//! Difficulty knobs: `blocks` (how many parts the program has), `features`
//! (which kinds of parts may appear: 1 arithmetic and loops, 2 adds lists and
//! small functions, 3 adds nesting and strings, 4 adds closures) and
//! `magnitude` (how large the numbers and loop bounds are).

use std::collections::BTreeMap;

use splinter_core::rng::SplitMix64;
use splinter_core::verifiability::VerifierClass;

use crate::factory::artifact::{
    environment_id, EnvMeta, Instance, Parts, SolverTools, SCHEMA_VERSION,
};
use crate::factory::attack::{contains_lines, GUESSES};
use crate::factory::folder::Folder;
use crate::factory::host::Host;
use crate::factory::plugin::{resolve_difficulty, InstanceRequest, Knob, Plugin, PluginError};

use super::{PROGRAM_OUTPUT_LUA, PROGRAM_OUTPUT_PYTHON};

/// The plugin's version. Any change to the generator, the renderings, the
/// verifier script or the instruction text is a new version.
pub const VERSION: &str = "1";

const KNOBS: &[Knob] = &[
    Knob { name: "blocks", min: 1, max: 6, default: 2, doc: "how many parts the program has" },
    Knob { name: "features", min: 1, max: 4, default: 2, doc: "which parts may appear: 1 arithmetic and loops, 2 adds lists and functions, 3 adds nesting and strings, 4 adds closures" },
    Knob { name: "magnitude", min: 1, max: 4, default: 2, doc: "how large numbers and loop bounds get" },
];

/// How many times a request's stream is drawn from before the generator
/// gives up on finding a program whose output cannot be guessed.
const MAX_REDRAWS: usize = 32;

const WORDS: [&str; 8] = [
    "kettle", "marble", "lantern", "orchard", "pillow", "quartz", "saddle", "thimble",
];

/// A part of a program. Every part prints two lines.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Block {
    Chain {
        start: i64,
        steps: Vec<(i64, i64)>,
        modulus: i64,
    },
    Loop {
        n: i64,
        k: i64,
        a: i64,
    },
    Nested {
        n: i64,
        m: i64,
        d: i64,
    },
    List {
        len: i64,
        mul: i64,
        modulus: i64,
        pick: i64,
    },
    Text {
        word: &'static str,
        n: i64,
        k: i64,
    },
    Gcd {
        a: i64,
        b: i64,
    },
    Fib {
        n: i64,
    },
    Collatz {
        n: i64,
    },
    PowMod {
        base: i64,
        exp: i64,
        modulus: i64,
    },
    Counter {
        step: i64,
        calls: i64,
    },
}

/// The language a program is written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Language {
    Lua,
    Python,
}

impl Block {
    /// What the block prints, by the generator's own evaluation.
    fn eval(&self) -> Vec<String> {
        match self {
            Self::Chain {
                start,
                steps,
                modulus,
            } => {
                let mut x = *start;
                let mut out = Vec::new();
                for (i, (k, c)) in steps.iter().enumerate() {
                    if i == steps.len() / 2 {
                        out.push(x.to_string());
                    }
                    x = (x * k + c).rem_euclid(*modulus);
                }
                out.push(x.to_string());
                out
            }
            Self::Loop { n, k, a } => {
                let (mut s, mut hits) = (0, 0);
                for i in 1..=*n {
                    if i % k == 0 {
                        s += i * a;
                        hits += 1;
                    } else {
                        s -= 1;
                    }
                }
                vec![s.to_string(), hits.to_string()]
            }
            Self::Nested { n, m, d } => {
                let (mut t, mut hits) = (0, 0);
                for i in 1..=*n {
                    for j in 1..=*m {
                        if (i + j) % d == 0 {
                            t += i * j;
                            hits += 1;
                        }
                    }
                }
                vec![t.to_string(), hits.to_string()]
            }
            Self::List {
                len,
                mul,
                modulus,
                pick,
            } => {
                let values: Vec<i64> = (1..=*len).map(|i| (i * mul) % modulus).collect();
                let kept: Vec<String> = values
                    .iter()
                    .filter(|v| **v % pick == 0)
                    .map(ToString::to_string)
                    .collect();
                vec![
                    format!(
                        "{} {}",
                        values.iter().sum::<i64>(),
                        values.iter().max().copied().unwrap_or(0)
                    ),
                    kept.join(","),
                ]
            }
            Self::Text { word, n, k } => {
                let letters: Vec<char> = word.chars().collect();
                let built: String = (1..=*n)
                    .map(|i| letters[(i * k) as usize % letters.len()])
                    .collect();
                vec![built.clone(), built.to_uppercase()]
            }
            Self::Gcd { a, b } => {
                let (mut x, mut y) = (*a, *b);
                while y != 0 {
                    (x, y) = (y, x % y);
                }
                vec![x.to_string(), (a / x * b).to_string()]
            }
            Self::Fib { n } => {
                let (mut a, mut b) = (0_i64, 1_i64);
                for _ in 0..*n {
                    (a, b) = (b, a + b);
                }
                vec![a.to_string(), (a % 7).to_string()]
            }
            Self::Collatz { n } => {
                let (mut v, mut steps, mut peak) = (*n, 0, *n);
                while v != 1 {
                    v = if v % 2 == 0 { v / 2 } else { 3 * v + 1 };
                    peak = peak.max(v);
                    steps += 1;
                }
                vec![steps.to_string(), peak.to_string()]
            }
            Self::PowMod { base, exp, modulus } => {
                let mut r = 1;
                for _ in 0..*exp {
                    r = (r * base) % modulus;
                }
                vec![r.to_string(), ((r * r) % modulus).to_string()]
            }
            Self::Counter { step, calls } => {
                let last = step * calls;
                vec![last.to_string(), (last + step).to_string()]
            }
        }
    }

    /// The block's source; `i` makes its names unique in the program.
    fn render(&self, language: Language, i: usize) -> String {
        match language {
            Language::Lua => self.lua(i),
            Language::Python => self.python(i),
        }
    }

    fn lua(&self, i: usize) -> String {
        match self {
            Self::Chain { start, steps, modulus } => {
                let mut s = format!("local x{i} = {start}\n");
                for (n, (k, c)) in steps.iter().enumerate() {
                    if n == steps.len() / 2 {
                        s.push_str(&format!("print(x{i})\n"));
                    }
                    s.push_str(&format!("x{i} = (x{i} * {k} + {c}) % {modulus}\n"));
                }
                s.push_str(&format!("print(x{i})\n"));
                s
            }
            Self::Loop { n, k, a } => format!(
                "local s{i}, h{i} = 0, 0\nfor i = 1, {n} do\n  if i % {k} == 0 then\n    s{i} = s{i} + i * {a}\n    h{i} = h{i} + 1\n  else\n    s{i} = s{i} - 1\n  end\nend\nprint(s{i})\nprint(h{i})\n"
            ),
            Self::Nested { n, m, d } => format!(
                "local t{i}, h{i} = 0, 0\nfor i = 1, {n} do\n  for j = 1, {m} do\n    if (i + j) % {d} == 0 then\n      t{i} = t{i} + i * j\n      h{i} = h{i} + 1\n    end\n  end\nend\nprint(t{i})\nprint(h{i})\n"
            ),
            Self::List { len, mul, modulus, pick } => format!(
                "local v{i}, sum{i}, top{i}, kept{i} = {{}}, 0, 0, {{}}\nfor i = 1, {len} do\n  v{i}[i] = (i * {mul}) % {modulus}\n  sum{i} = sum{i} + v{i}[i]\n  if v{i}[i] > top{i} then top{i} = v{i}[i] end\n  if v{i}[i] % {pick} == 0 then table.insert(kept{i}, v{i}[i]) end\nend\nprint(sum{i} .. \" \" .. top{i})\nprint(table.concat(kept{i}, \",\"))\n"
            ),
            Self::Text { word, n, k } => format!(
                "local w{i} = \"{word}\"\nlocal s{i} = \"\"\nfor i = 1, {n} do\n  local at = (i * {k}) % #w{i} + 1\n  s{i} = s{i} .. string.sub(w{i}, at, at)\nend\nprint(s{i})\nprint(string.upper(s{i}))\n"
            ),
            Self::Gcd { a, b } => format!(
                "local a{i}, b{i} = {a}, {b}\nwhile b{i} ~= 0 do\n  a{i}, b{i} = b{i}, a{i} % b{i}\nend\nprint(a{i})\nprint({a} / a{i} * {b})\n"
            ),
            Self::Fib { n } => format!(
                "local a{i}, b{i} = 0, 1\nfor _ = 1, {n} do\n  a{i}, b{i} = b{i}, a{i} + b{i}\nend\nprint(a{i})\nprint(a{i} % 7)\n"
            ),
            Self::Collatz { n } => format!(
                "local v{i}, c{i}, p{i} = {n}, 0, {n}\nwhile v{i} ~= 1 do\n  if v{i} % 2 == 0 then v{i} = v{i} / 2 else v{i} = 3 * v{i} + 1 end\n  if v{i} > p{i} then p{i} = v{i} end\n  c{i} = c{i} + 1\nend\nprint(c{i})\nprint(p{i})\n"
            ),
            Self::PowMod { base, exp, modulus } => format!(
                "local r{i} = 1\nfor _ = 1, {exp} do\n  r{i} = (r{i} * {base}) % {modulus}\nend\nprint(r{i})\nprint((r{i} * r{i}) % {modulus})\n"
            ),
            Self::Counter { step, calls } => format!(
                "local function make{i}(step)\n  local c = 0\n  return function()\n    c = c + step\n    return c\n  end\nend\nlocal next{i} = make{i}({step})\nlocal last{i} = 0\nfor _ = 1, {calls} do\n  last{i} = next{i}()\nend\nprint(last{i})\nprint(next{i}())\n"
            ),
        }
    }

    fn python(&self, i: usize) -> String {
        match self {
            Self::Chain { start, steps, modulus } => {
                let mut s = format!("x{i} = {start}\n");
                for (n, (k, c)) in steps.iter().enumerate() {
                    if n == steps.len() / 2 {
                        s.push_str(&format!("print(x{i})\n"));
                    }
                    s.push_str(&format!("x{i} = (x{i} * {k} + {c}) % {modulus}\n"));
                }
                s.push_str(&format!("print(x{i})\n"));
                s
            }
            Self::Loop { n, k, a } => format!(
                "s{i} = 0\nh{i} = 0\nfor i in range(1, {n} + 1):\n    if i % {k} == 0:\n        s{i} = s{i} + i * {a}\n        h{i} = h{i} + 1\n    else:\n        s{i} = s{i} - 1\nprint(s{i})\nprint(h{i})\n"
            ),
            Self::Nested { n, m, d } => format!(
                "t{i} = 0\nh{i} = 0\nfor i in range(1, {n} + 1):\n    for j in range(1, {m} + 1):\n        if (i + j) % {d} == 0:\n            t{i} = t{i} + i * j\n            h{i} = h{i} + 1\nprint(t{i})\nprint(h{i})\n"
            ),
            Self::List { len, mul, modulus, pick } => format!(
                "v{i} = []\nfor i in range(1, {len} + 1):\n    v{i}.append((i * {mul}) % {modulus})\nprint(str(sum(v{i})) + \" \" + str(max(v{i})))\nprint(\",\".join(str(x) for x in v{i} if x % {pick} == 0))\n"
            ),
            Self::Text { word, n, k } => format!(
                "w{i} = \"{word}\"\ns{i} = \"\"\nfor i in range(1, {n} + 1):\n    at = (i * {k}) % len(w{i})\n    s{i} = s{i} + w{i}[at]\nprint(s{i})\nprint(s{i}.upper())\n"
            ),
            Self::Gcd { a, b } => format!(
                "a{i} = {a}\nb{i} = {b}\nwhile b{i} != 0:\n    a{i}, b{i} = b{i}, a{i} % b{i}\nprint(a{i})\nprint({a} // a{i} * {b})\n"
            ),
            Self::Fib { n } => format!(
                "a{i} = 0\nb{i} = 1\nfor _ in range({n}):\n    a{i}, b{i} = b{i}, a{i} + b{i}\nprint(a{i})\nprint(a{i} % 7)\n"
            ),
            Self::Collatz { n } => format!(
                "v{i} = {n}\nc{i} = 0\np{i} = {n}\nwhile v{i} != 1:\n    if v{i} % 2 == 0:\n        v{i} = v{i} // 2\n    else:\n        v{i} = 3 * v{i} + 1\n    if v{i} > p{i}:\n        p{i} = v{i}\n    c{i} = c{i} + 1\nprint(c{i})\nprint(p{i})\n"
            ),
            Self::PowMod { base, exp, modulus } => format!(
                "r{i} = 1\nfor _ in range({exp}):\n    r{i} = (r{i} * {base}) % {modulus}\nprint(r{i})\nprint((r{i} * r{i}) % {modulus})\n"
            ),
            Self::Counter { step, calls } => format!(
                "def make{i}(step):\n    c = 0\n    def bump():\n        nonlocal c\n        c = c + step\n        return c\n    return bump\nnext{i} = make{i}({step})\nlast{i} = 0\nfor _ in range({calls}):\n    last{i} = next{i}()\nprint(last{i})\nprint(next{i}())\n"
            ),
        }
    }
}

/// A program: its parts in order.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Program {
    blocks: Vec<Block>,
}

impl Program {
    fn source(&self, language: Language) -> String {
        self.blocks
            .iter()
            .enumerate()
            .map(|(i, b)| b.render(language, i))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn output(&self) -> String {
        self.blocks
            .iter()
            .flat_map(Block::eval)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn draw_block(rng: &mut SplitMix64, features: u32, magnitude: i64) -> Block {
    let big = 4 + 4 * magnitude;
    let kinds: &[u8] = match features {
        1 => &[0, 1],
        2 => &[0, 1, 2, 3, 4, 5, 6],
        3 => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        _ => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    };
    let range = |rng: &mut SplitMix64, lo: i64, hi: i64| rng.between(lo as u64, hi as u64) as i64;
    match rng.pick(kinds).copied().unwrap_or(0) {
        0 => {
            let steps = (0..range(rng, 2, 2 + magnitude + 1))
                .map(|_| (range(rng, 2, 9), range(rng, 1, 9)))
                .collect();
            Block::Chain {
                start: range(rng, 1, 20),
                steps,
                modulus: range(rng, 50, 100 * magnitude),
            }
        }
        1 => Block::Loop {
            n: range(rng, 5, big),
            k: range(rng, 2, 5),
            a: range(rng, 2, 9),
        },
        2 => {
            let (len, modulus) = (range(rng, 5, big.min(14)), range(rng, 10, 30));
            let mul = range(rng, 3, 9);
            let values: Vec<i64> = (1..=len).map(|i| (i * mul) % modulus).collect();
            let pick = (2..=5)
                .find(|p| values.iter().any(|v| v % p == 0 && *v > 0))
                .unwrap_or(1);
            Block::List {
                len,
                mul,
                modulus,
                pick,
            }
        }
        3 => {
            let b = range(rng, 6, 20 * magnitude + 20);
            Block::Gcd {
                a: b * range(rng, 2, 6),
                b: b + range(rng, 1, 9),
            }
        }
        4 => Block::Fib {
            n: range(rng, 6, 12 + 3 * magnitude),
        },
        5 => Block::PowMod {
            base: range(rng, 2, 9),
            exp: range(rng, 3, big),
            modulus: range(rng, 7, 40 + 20 * magnitude),
        },
        6 => Block::Collatz {
            n: range(rng, 3, 10 + 20 * magnitude),
        },
        7 => Block::Nested {
            n: range(rng, 3, 3 + magnitude + 3),
            m: range(rng, 3, 3 + magnitude + 3),
            d: range(rng, 2, 4),
        },
        8 => Block::Text {
            word: WORDS[rng.below(WORDS.len())],
            n: range(rng, 5, 8 + 2 * magnitude),
            k: range(rng, 2, 7),
        },
        _ => Block::Counter {
            step: range(rng, 2, 9),
            calls: range(rng, 3, 3 + 2 * magnitude),
        },
    }
}

/// Whether a program's output could be written without running it: a guess
/// or a run of lines the program text already holds.
fn guessable(program: &Program, text: &str) -> bool {
    let output = program.output();
    GUESSES.iter().any(|g| g.trim() == output.trim()) || contains_lines(text, &output)
}

fn program_for(request: &InstanceRequest) -> Result<Program, PluginError> {
    let get = |name: &str| i64::from(request.difficulty.get(name).copied().unwrap_or(1));
    let (blocks, features, magnitude) = (get("blocks"), get("features") as u32, get("magnitude"));
    let mut rng = SplitMix64::new(request.seed);
    for _ in 0..MAX_REDRAWS {
        let program = Program {
            blocks: (0..blocks)
                .map(|_| draw_block(&mut rng, features, magnitude))
                .collect(),
        };
        let both = format!(
            "{}\n{}",
            program.source(Language::Lua),
            program.source(Language::Python)
        );
        if !guessable(&program, &both) {
            return Ok(program);
        }
    }
    Err(PluginError::Generation(format!(
        "no program whose output cannot be guessed in {MAX_REDRAWS} draws at {:?}",
        request.difficulty
    )))
}

/// The program-output plugin for one language.
pub struct ProgramOutput {
    language: Language,
}

impl ProgramOutput {
    /// Lua 5.1 programs, run by `lua`.
    #[must_use]
    pub fn lua() -> Self {
        Self {
            language: Language::Lua,
        }
    }

    /// Python 3 programs, run by `python3`.
    #[must_use]
    pub fn python() -> Self {
        Self {
            language: Language::Python,
        }
    }

    fn id_of(language: Language) -> &'static str {
        match language {
            Language::Lua => PROGRAM_OUTPUT_LUA,
            Language::Python => PROGRAM_OUTPUT_PYTHON,
        }
    }

    fn file(&self) -> &'static str {
        match self.language {
            Language::Lua => "program.lua",
            Language::Python => "program.py",
        }
    }

    /// The command line that runs a program of this language, and the check
    /// that the interpreter is the version the task is written for.
    fn interpreter(&self) -> (&'static str, &'static str) {
        match self.language {
            Language::Lua => ("lua", "lua -v 2>&1 | grep -q 'Lua 5.1' || { echo 'the oracle must be Lua 5.1' >&2; exit 1; }"),
            Language::Python => ("python3 -I", "python3 -c 'import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)' || { echo 'the oracle must be Python 3' >&2; exit 1; }"),
        }
    }

    fn verifier(&self) -> String {
        let (run, version_check) = self.interpreter();
        let file = self.file();
        format!(
            r#"#!/bin/sh
# Grades answer.txt by what the interpreter prints for the pinned program.
# The program is in tests/, never in the workspace; the expected output is
# computed here and stored nowhere. No reward is written when the oracle
# itself cannot run: that is a broken verifier, not a wrong answer.
APP="${{APP_DIR:-/app}}"; TESTS="${{TESTS_DIR:-/tests}}"; LOGS="${{LOGS_DIR:-/logs/verifier}}"
mkdir -p "$LOGS"
{version_check}
{run} "$TESTS/{file}" > "$LOGS/expected.raw" 2> "$LOGS/expected.err" || {{ echo "the oracle failed" >&2; exit 1; }}
norm() {{ if [ -f "$1" ]; then sed -e 's/[[:space:]]*$//' "$1"; fi; }}
norm "$LOGS/expected.raw" > "$LOGS/expected.txt"
norm "$APP/answer.txt" > "$LOGS/got.txt"
if [ "$(cat "$LOGS/expected.txt")" = "$(cat "$LOGS/got.txt")" ]; then exact=1; else exact=0; fi
lines=$(awk 'NR==FNR {{ e[FNR] = $0; n = FNR; next }} FNR <= n && $0 == e[FNR] {{ m++ }} END {{ printf "%.4f", (n > 0 ? m / n : 0) }}' "$LOGS/expected.txt" "$LOGS/got.txt")
printf '{{"reward": %s, "exact": %s, "lines": %s}}\n' "$exact" "$exact" "$lines" > "$LOGS/reward.json"
"#
        )
    }

    fn instruction(&self, source: &str) -> String {
        let (name, fence) = match self.language {
            Language::Lua => ("Lua 5.1", "lua"),
            Language::Python => ("Python 3", "python"),
        };
        format!(
            "# Predict the output\n\nBelow is a {name} program. Work out exactly what it prints to standard output and write that output, line by line, to `/app/answer.txt`.\n\nDo not run the program; read it and predict. Trailing spaces on a line and trailing blank lines are ignored. Anything else must match exactly.\n\n```{fence}\n{source}```\n"
        )
    }
}

impl Plugin for ProgramOutput {
    fn id(&self) -> &'static str {
        Self::id_of(self.language)
    }

    fn version(&self) -> &'static str {
        VERSION
    }

    fn verifier_class(&self) -> VerifierClass {
        VerifierClass::Objective
    }

    fn knobs(&self) -> &'static [Knob] {
        KNOBS
    }

    fn generate(&self, request: &InstanceRequest) -> Result<Instance, PluginError> {
        let difficulty = resolve_difficulty(self.id(), KNOBS, &request.difficulty)?;
        let request = InstanceRequest {
            difficulty: difficulty.clone(),
            ..request.clone()
        };
        let program = program_for(&request)?;
        let source = program.source(self.language);

        let mut workspace = Folder::new();
        workspace.put_text("answer.txt", "")?;
        let mut tests = Folder::new();
        tests.put_script("test.sh", &self.verifier())?;
        tests.put_text(self.file(), &source)?;
        let mut solution = Folder::new();
        let (run, _) = self.interpreter();
        solution.put_script(
            "solve.sh",
            &format!("#!/bin/sh\n# The reference: run the program and keep what it prints.\n{run} \"${{SOLUTION_DIR:-/solution}}/{}\" > \"${{APP_DIR:-/app}}/answer.txt\"\n", self.file()),
        )?;
        solution.put_text(self.file(), &source)?;

        let meta = EnvMeta {
            schema_version: SCHEMA_VERSION,
            environment: environment_id(self.id(), VERSION, &difficulty),
            plugin: self.id().into(),
            plugin_version: VERSION.into(),
            verifier_class: VerifierClass::Objective,
            verifier: format!("interpreter-output@{VERSION}"),
            difficulty,
            split: request.split,
            seed: request.seed,
            solver_tools: SolverTools::Files,
        };
        let packages = match self.language {
            Language::Lua => vec!["lua5.1".to_string()],
            Language::Python => vec!["python3".to_string()],
        };
        Ok(Instance::assemble(
            meta,
            Parts {
                instruction: self.instruction(&source),
                workspace,
                packages,
                tests,
                solution,
                category: "program-output".into(),
            },
        )?)
    }

    fn novelty_text(&self, instance: &Instance) -> String {
        instance
            .folder()
            .text(&format!("tests/{}", self.file()))
            .unwrap_or_default()
            .to_string()
    }

    fn cross_check(
        &self,
        request: &InstanceRequest,
        host: &Host,
    ) -> Option<Result<String, String>> {
        Some(self.compare_interpreters(request, host))
    }
}

impl ProgramOutput {
    /// Runs the program in Lua and in Python and compares both with the
    /// generator's evaluation.
    fn compare_interpreters(
        &self,
        request: &InstanceRequest,
        host: &Host,
    ) -> Result<String, String> {
        let difficulty =
            resolve_difficulty(self.id(), KNOBS, &request.difficulty).map_err(|e| e.to_string())?;
        let request = InstanceRequest {
            difficulty,
            ..request.clone()
        };
        let program = program_for(&request).map_err(|e| e.to_string())?;
        let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        let mut files = Folder::new();
        files
            .put_text("p.lua", &program.source(Language::Lua))
            .map_err(|e| e.to_string())?;
        files
            .put_text("p.py", &program.source(Language::Python))
            .map_err(|e| e.to_string())?;
        let root = dir.path().join("p");
        files.write_to(&root).map_err(|e| e.to_string())?;
        let expected = program.output();
        for (argv, name) in [
            (vec!["lua", "p.lua"], "lua"),
            (vec!["python3", "-I", "p.py"], "python"),
        ] {
            let ran = host
                .exec(
                    &root,
                    argv.iter().map(ToString::to_string).collect(),
                    BTreeMap::new(),
                )
                .map_err(|e| e.to_string())?;
            if !ran.succeeded() {
                return Err(format!("{name} failed: {}", ran.stderr.trim()));
            }
            if ran.stdout.trim_end() != expected {
                return Err(format!(
                    "{name} printed {:?}, the generator expected {expected:?}",
                    ran.stdout.trim_end()
                ));
            }
        }
        Ok(format!(
            "lua, python and the generator's evaluator agree on {} lines",
            expected.lines().count()
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;
    use splinter_core::verifiability::Split;

    fn request(seed: u64, blocks: u32, features: u32) -> InstanceRequest {
        InstanceRequest {
            split: Split::Train,
            seed,
            difficulty: BTreeMap::from([
                ("blocks".into(), blocks),
                ("features".into(), features),
                ("magnitude".into(), 2),
            ]),
        }
    }

    #[test]
    fn the_same_request_gives_the_same_folder_and_another_seed_another_program() {
        // seed: determinism and difference are what is tested
        for plugin in [ProgramOutput::lua(), ProgramOutput::python()] {
            let a = plugin.generate(&request(3, 3, 4)).unwrap();
            assert_eq!(
                a.version(),
                plugin.generate(&request(3, 3, 4)).unwrap().version()
            );
            assert_ne!(
                a.version(),
                plugin.generate(&request(4, 3, 4)).unwrap().version()
            );
        }
    }

    #[test]
    fn the_answer_is_stored_nowhere_in_the_folder_and_the_program_is_shown_in_the_instruction() {
        let plugin = ProgramOutput::lua();
        let instance = plugin.generate(&request(11, 3, 3)).unwrap(); // seed: any seed must keep the answer out
        let source = instance.folder().text("tests/program.lua").unwrap();
        assert!(instance.instruction().contains(source));
        let program = program_for(&request(11, 3, 3)).unwrap();
        for (path, entry) in instance.folder().entries() {
            if let Ok(text) = std::str::from_utf8(&entry.bytes) {
                assert!(
                    !contains_lines(text, &program.output()),
                    "{path} holds the answer"
                );
            }
        }
    }

    #[test]
    fn a_knob_changes_the_environment_id_and_the_size_of_the_program() {
        let plugin = ProgramOutput::python();
        let small = plugin.generate(&request(5, 1, 1)).unwrap();
        let large = plugin.generate(&request(5, 6, 4)).unwrap();
        assert_ne!(small.meta().environment, large.meta().environment);
        assert!(large.instruction().len() > small.instruction().len());
    }

    #[test]
    fn every_feature_level_generates_for_many_seeds_without_giving_up() {
        for features in 1..=4 {
            for seed in 0..60 {
                // seed: coverage of the generator over many seeds
                program_for(&request(seed, 4, features)).unwrap();
            }
        }
    }

    #[test]
    fn an_unknown_knob_is_refused() {
        let mut bad = request(1, 1, 1);
        bad.difficulty.insert("depth".into(), 3);
        assert!(ProgramOutput::lua().generate(&bad).is_err());
    }

    #[test]
    fn the_evaluator_agrees_with_both_interpreters_on_many_programs() {
        let host = Host::default();
        for plugin in [ProgramOutput::lua(), ProgramOutput::python()] {
            for seed in 0..40 {
                // seed: coverage of the generator over many seeds
                let found = plugin.cross_check(&request(seed, 5, 4), &host).unwrap();
                assert!(found.is_ok(), "seed {seed}: {found:?}");
            }
        }
    }
}
