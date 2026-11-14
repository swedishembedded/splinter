// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that put an informative on-policy
// reinforcement-learning signal through a language model against verifiers it
// cannot influence, for its clients. If your team needs expertise in
// verifiable task families for policy-gradient training you can procure our
// services by sending an email to info@swedishembedded.com.

//! Lua output prediction as a verifiable task family: a program is shown, the
//! policy writes what it prints, and the Lua interpreter that actually ran the
//! program is the verifier. Programs come in four levels of difficulty
//! (sums, products, strings, loops, tables, long arithmetic) from a seeded generator, so a pool
//! can be made as large and as hard as a campaign needs, and the dev and
//! retention sets are drawn from other seeds than the training pool.

pub mod family;
pub mod frontier;
pub mod frontier_round;
pub mod handoff;
pub mod journal;
pub mod lineage;
pub mod neutral;
pub mod plan;
pub mod proposer;
pub mod recipe;
pub mod sections;
pub mod sft;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};
use splinter_sdk::campaign::brain::{Graded, TaskFamily};
use splinter_sdk::campaign::CampaignError;
use splinter_sdk::vocabulary::rng::SplitMix64;

/// The system message of every task.
const SYSTEM: &str =
    "You are a Lua interpreter. Reply with exactly what the program prints and nothing else.";

/// Levels of difficulty.
pub const LEVELS: usize = 6;

/// One generated program and what the interpreter printed for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    /// Its source.
    pub source: String,
    /// What the interpreter printed.
    pub expected: String,
    /// Its level, from 1.
    pub level: usize,
}

/// The family: programs by task id, in a training pool, a dev set and a
/// retention set.
pub struct LuaPredict {
    programs: BTreeMap<String, Program>,
    pool: Vec<String>,
    dev: Vec<String>,
    retention: Vec<String>,
    settings: Value,
}

/// How many programs of each part, and the seeds they are drawn from.
#[derive(Clone, Copy, Debug)]
pub struct Sizes {
    /// Training programs.
    pub pool: usize,
    /// Dev programs.
    pub dev: usize,
    /// Retention programs (level 1 and 2, which the base already solves).
    pub retention: usize,
}

fn word(rng: &mut SplitMix64) -> String {
    const SYLLABLES: [&str; 8] = ["ka", "lo", "mi", "ru", "te", "so", "ba", "ne"];
    (0..2 + rng.below(2))
        .map(|_| SYLLABLES[rng.below(SYLLABLES.len())])
        .collect()
}

/// A program of `level`, Lua 5.1 compatible. The levels climb from sums the
/// base answers without reasoning to programs it cannot do without it, so a
/// pool spans the pass rates a policy-gradient run learns from: 1 a sum of two
/// small numbers, 2 a product plus a digit, 3 a short string transformation,
/// 4 a small accumulating loop with a start and a step, 5 a three-item table sort, 6 a longer
/// arithmetic expression.
fn generate(level: usize, rng: &mut SplitMix64) -> String {
    let n = |rng: &mut SplitMix64, low: u64, high: u64| rng.between(low, high);
    match level {
        1 => format!("print({} + {})", n(rng, 1, 30), n(rng, 1, 30)),
        2 => format!(
            "print({} * {} + {})",
            n(rng, 2, 9),
            n(rng, 2, 9),
            n(rng, 1, 9)
        ),
        3 => {
            let w = word(rng);
            match rng.below(3) {
                0 => format!("print(string.upper(\"{w}\"))"),
                1 => format!("print(string.rep(\"{}\", {}))", &w[..2], n(rng, 2, 4)),
                _ => format!("print(#\"{w}\" + {})", n(rng, 1, 9)),
            }
        }
        4 => format!(
            "local s = {}\nfor i = 1, {} do\n  s = s + i * {}\nend\nprint(s)",
            n(rng, 0, 9),
            n(rng, 3, 9),
            n(rng, 1, 5)
        ),
        5 => {
            let items: Vec<String> = (0..3).map(|_| n(rng, 1, 9).to_string()).collect();
            format!(
                "local t = {{{}}}\ntable.sort(t)\nprint(table.concat(t, \",\"))",
                items.join(", ")
            )
        }
        _ => format!(
            "print({} * {} + {} - {})",
            n(rng, 12, 29),
            n(rng, 3, 19),
            n(rng, 10, 99),
            n(rng, 5, 49)
        ),
    }
}

/// Runs one program. Every program goes through the same scratch file: a file
/// per program would put hundreds of files on a disk with a quota on their
/// number.
fn run_lua(lua: &Path, source: &str, scratch: &Path) -> Result<String, CampaignError> {
    let file = scratch.join("program.lua");
    std::fs::write(&file, source)
        .map_err(|e| CampaignError::Config(format!("writing {}: {e}", file.display())))?;
    let out = Command::new(lua)
        .arg(&file)
        .output()
        .map_err(|e| CampaignError::Config(format!("running {}: {e}", lua.display())))?;
    if !out.status.success() {
        return Err(CampaignError::Config(format!(
            "the interpreter refused a generated program ({source:?}): {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

impl LuaPredict {
    /// Generates the family: every program is run by the interpreter at
    /// `lua` once, in `scratch`, to record what it prints. The pool, dev and
    /// retention sets are drawn from `seed` under different salts, and no
    /// program is in two of them.
    ///
    /// # Errors
    /// The interpreter cannot run, or refuses a program.
    pub fn generate(
        lua: &Path,
        scratch: &Path,
        sizes: Sizes,
        seed: u64,
    ) -> Result<Self, CampaignError> {
        let mut programs: BTreeMap<String, Program> = BTreeMap::new();
        let mut seen: BTreeMap<String, ()> = BTreeMap::new();
        let mut part = |salt: u64,
                        count: usize,
                        levels: &[usize],
                        prefix: &str|
         -> Result<Vec<String>, CampaignError> {
            let mut rng = SplitMix64::new(seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut ids = Vec::new();
            let mut attempts = 0;
            while ids.len() < count {
                attempts += 1;
                if attempts > count * 50 {
                    return Err(CampaignError::Config(format!(
                        "the generator cannot make {count} distinct programs of levels {levels:?}"
                    )));
                }
                let level = levels[ids.len() % levels.len()];
                let source = generate(level, &mut rng);
                if seen.insert(source.clone(), ()).is_some() {
                    continue;
                }
                let id = format!("{prefix}-l{level}-{:04}", ids.len());
                let expected = run_lua(lua, &source, scratch)?;
                programs.insert(
                    id.clone(),
                    Program {
                        source,
                        expected,
                        level,
                    },
                );
                ids.push(id);
            }
            Ok(ids)
        };
        let all: Vec<usize> = (1..=LEVELS).collect();
        let pool = part(1, sizes.pool, &all, "train")?;
        let dev = part(2, sizes.dev, &all, "dev")?;
        let retention = part(3, sizes.retention, &[1, 2], "keep")?;
        Ok(Self {
            programs,
            pool,
            dev,
            retention,
            settings: json!({"family": "lua-predict", "levels": LEVELS, "sizes": [sizes.pool, sizes.dev, sizes.retention], "seed": seed}),
        })
    }

    /// The program of `task`.
    #[must_use]
    pub fn program(&self, task: &str) -> Option<&Program> {
        self.programs.get(task)
    }

    /// The source of every program of the family.
    pub fn sources(&self) -> impl Iterator<Item = &str> {
        self.programs.values().map(|p| p.source.as_str())
    }

    fn get(&self, task: &str) -> Result<&Program, CampaignError> {
        self.programs
            .get(task)
            .ok_or_else(|| CampaignError::Config(format!("{task:?} is not a task of the family")))
    }
}

/// What the policy wrote, reduced to the answer: code fences and blank lines
/// are not part of it, and the answer is the last line left.
#[must_use]
pub fn answer_of(completion: &str) -> String {
    completion
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty() && !l.starts_with("```"))
        .unwrap_or("")
        .trim_matches('`')
        .to_string()
}

impl TaskFamily for LuaPredict {
    fn describe(&self) -> Value {
        self.settings.clone()
    }

    fn pool(&self) -> Vec<String> {
        self.pool.clone()
    }

    fn dev(&self) -> Vec<String> {
        self.dev.clone()
    }

    fn retention(&self) -> Vec<String> {
        self.retention.clone()
    }

    fn prompt(&self, task: &str) -> Result<Vec<(String, String)>, CampaignError> {
        let program = self.get(task)?;
        Ok(vec![
            ("system".into(), SYSTEM.into()),
            (
                "user".into(),
                format!(
                    "```lua\n{}\n```\nWhat does this program print?",
                    program.source
                ),
            ),
        ])
    }

    fn grade(&self, task: &str, completion: &str) -> Result<Graded, CampaignError> {
        let program = self.get(task)?;
        let passed = answer_of(completion) == program.expected;
        Ok(Graded {
            reward: f64::from(u8::from(passed)),
            passed,
            feedback: (!passed)
                .then(|| "That is not what the program prints. Try again.".to_string()),
            privileged: Some(format!("The program prints exactly: {}", program.expected)),
        })
    }
}

/// The interpreter's path, for a caller that wants a default.
#[must_use]
pub fn default_interpreter() -> PathBuf {
    PathBuf::from("lua")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn family() -> (LuaPredict, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let sizes = Sizes {
            pool: 24,
            dev: 12,
            retention: 8,
        };
        let seed = splinter_sdk::seed::resolve("seed", None);
        (
            LuaPredict::generate(&default_interpreter(), dir.path(), sizes, seed).unwrap(),
            dir,
        )
    }

    #[test]
    fn the_interpreter_is_the_verifier_and_the_sets_are_disjoint() {
        let (family, _dir) = family();
        let (pool, dev, keep) = (family.pool(), family.dev(), family.retention());
        assert_eq!((pool.len(), dev.len(), keep.len()), (24, 12, 8));
        let sources: std::collections::BTreeSet<&str> = pool
            .iter()
            .chain(&dev)
            .chain(&keep)
            .map(|t| family.program(t).unwrap().source.as_str())
            .collect();
        assert_eq!(sources.len(), 44, "no program is in two parts or twice");
        let task = &pool[0];
        let program = family.program(task).unwrap();
        assert!(family.grade(task, &program.expected).unwrap().passed);
        assert!(!family.grade(task, "definitely not it").unwrap().passed);
    }

    #[test]
    fn a_fenced_or_chatty_answer_is_reduced_to_its_last_line() {
        assert_eq!(answer_of("```\n42\n```"), "42");
        assert_eq!(answer_of("The output is:\n\n42\n"), "42");
        assert_eq!(answer_of(""), "");
    }

    #[test]
    fn the_prompt_shows_the_program_and_never_the_answer() {
        let (family, _dir) = family();
        let task = &family.pool()[3];
        let turns = family.prompt(task).unwrap();
        let program = family.program(task).unwrap();
        assert!(turns[1].1.contains(&program.source));
        let graded = family.grade(task, "x").unwrap();
        assert!(graded.privileged.unwrap().contains(&program.expected));
        assert!(!graded.feedback.unwrap().contains(&program.expected));
    }
}
