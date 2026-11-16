// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tool-use task generators that teach agents
// skills rather than templates, for its clients. If your team needs
// expertise in training data that generalises to unseen phrasings and
// formats, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The primitive workspace skills, drawn as task instances: read a value,
//! follow a chain of lookups across files, compute with a helper whose usage
//! is discovered, set a key in a file, count matching lines.
//!
//! Each instance has a [`Plan`]: what to read and do, never the answer. The
//! expert follows it and takes every value out of tool results; the verifier
//! judges the workspace. Requests come from a grammar of frames and synonyms
//! with the data in one of several file formats, so no phrasing dominates;
//! the held-out probe draws from frames, synonyms and formats training never
//! uses.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::formats::{Format, HELD_OUT, TRAINING};
use crate::world::word;

/// The file the agent writes its answer to.
pub const ANSWER_FILE: &str = "answer.txt";

/// A skill.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Skill {
    /// The value of one key.
    Read,
    /// A key followed through two to four files.
    Chain,
    /// Two values combined by a helper.
    Compute,
    /// A key set to a value read elsewhere.
    Edit,
    /// Lines of a log naming a word.
    Count,
}

/// Every skill, in draw order.
pub const SKILLS: [Skill; 5] = [
    Skill::Read,
    Skill::Chain,
    Skill::Compute,
    Skill::Edit,
    Skill::Count,
];

impl Skill {
    /// The skill's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Chain => "chain",
            Self::Compute => "compute",
            Self::Edit => "edit",
            Self::Count => "count",
        }
    }
}

/// A keyed value in a file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slot {
    /// The file.
    pub file: String,
    /// Its format.
    pub format: Format,
    /// The key.
    pub key: String,
}

/// What an instance asks for, without its answer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Plan {
    /// The value at `slot`.
    Read {
        /// Where it is.
        slot: Slot,
    },
    /// Start at `start`; each value is the key to look up in the next file.
    Chain {
        /// The first key.
        start: Slot,
        /// The files after the first, in order, with their formats.
        next: Vec<(String, Format)>,
    },
    /// The helper `helper` applied to the values at `a` and `b`.
    Compute {
        /// The helper, a program in the workspace (`./mix`).
        helper: String,
        /// The first operand.
        a: Slot,
        /// The second operand.
        b: Slot,
    },
    /// Set `target` to the value at `source`.
    Edit {
        /// The key to set.
        target: Slot,
        /// Where the new value is.
        source: Slot,
    },
    /// Lines of `file` that contain `word`.
    Count {
        /// The log.
        file: String,
        /// The word.
        word: String,
    },
}

/// What the verifier checks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Check {
    /// `answer.txt` holds this value.
    Answer {
        /// The value.
        value: String,
    },
    /// The file holds exactly these entries.
    File {
        /// The file.
        file: String,
        /// Its format.
        format: Format,
        /// Its entries, by key.
        entries: BTreeMap<String, String>,
    },
}

/// One drawn task: its request, workspace, plan and check.
#[derive(Clone, Debug)]
pub struct Drawn {
    /// The skill.
    pub skill: Skill,
    /// The request.
    pub instruction: String,
    /// The workspace's files.
    pub files: BTreeMap<String, String>,
    /// The files that must be executable.
    pub executables: Vec<String>,
    /// What to do.
    pub plan: Plan,
    /// What the verifier checks.
    pub check: Check,
}

fn pick<'a>(rng: &mut SplitMix64, items: &[&'a str]) -> &'a str {
    items[rng.below(items.len())]
}

/// Distinct lowercase names.
fn names(rng: &mut SplitMix64, n: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    while out.len() < n {
        let w = word(rng, 2).to_lowercase();
        if !out.contains(&w) {
            out.push(w);
        }
    }
    out
}

/// A file path for `format` (a directory sometimes).
fn path(rng: &mut SplitMix64, format: Format, taken: &[String]) -> String {
    loop {
        let dir = pick(rng, &["", "", "data/", "config/", "inputs/", "notes/"]);
        let p = format!(
            "{dir}{}.{}",
            word(rng, 1).to_lowercase(),
            format.extension()
        );
        if !taken.contains(&p) {
            return p;
        }
    }
}

fn format(rng: &mut SplitMix64, held_out: bool) -> Format {
    if held_out {
        HELD_OUT[rng.below(HELD_OUT.len())]
    } else {
        TRAINING[rng.below(TRAINING.len())]
    }
}

/// The request's grammar: frames and synonyms, a training set and a disjoint
/// held-out set.
struct Grammar {
    find: &'static [&'static str],
    save: &'static [&'static str],
    into: &'static [&'static str],
    value: &'static [&'static str],
}

const TRAIN_GRAMMAR: Grammar = Grammar {
    find: &["Find", "Look up", "Get", "Read", "Fetch", "Retrieve"],
    save: &["write", "save", "put", "store", "record"],
    into: &["to answer.txt", "in answer.txt", "into answer.txt"],
    value: &["value", "entry", "setting"],
};

const HELD_OUT_GRAMMAR: Grammar = Grammar {
    find: &["Determine", "Work out", "Check"],
    save: &["place", "leave", "enter"],
    into: &["in the file answer.txt", "inside answer.txt"],
    value: &["number", "figure"],
};

fn grammar(held_out: bool) -> &'static Grammar {
    if held_out {
        &HELD_OUT_GRAMMAR
    } else {
        &TRAIN_GRAMMAR
    }
}

/// "the value of KEY in FILE", in one of several orders.
fn of(rng: &mut SplitMix64, g: &Grammar, slot: &Slot) -> String {
    let v = pick(rng, g.value);
    match rng.below(3) {
        0 => format!("the {v} of {} in {}", slot.key, slot.file),
        1 => format!("the {v} stored under {} in {}", slot.key, slot.file),
        _ => format!("{}'s {v} from {}", slot.key, slot.file),
    }
}

/// A file of `n` entries with fresh keys, `value` drawing each value.
fn table(
    rng: &mut SplitMix64,
    n: usize,
    value: &mut dyn FnMut(&mut SplitMix64) -> String,
) -> Vec<(String, String)> {
    let keys = names(rng, n);
    keys.into_iter().map(|k| (k, value(rng))).collect()
}

fn number(rng: &mut SplitMix64) -> String {
    rng.between(10, 9999).to_string()
}

/// Draws one instance of `skill`; `held_out` selects the held-out grammar
/// and formats.
#[must_use]
pub fn draw(rng: &mut SplitMix64, skill: Skill, held_out: bool) -> Drawn {
    let g = grammar(held_out);
    let mut files = BTreeMap::new();
    let mut taken: Vec<String> = Vec::new();
    let mut add = |rng: &mut SplitMix64,
                   entries: &[(String, String)],
                   files: &mut BTreeMap<String, String>| {
        let f = format(rng, held_out);
        let mut entries = entries.to_vec();
        if f == Format::Json {
            entries.sort();
        }
        let p = path(rng, f, &taken);
        taken.push(p.clone());
        files.insert(p.clone(), f.render(&entries));
        (p, f, entries)
    };
    let save = pick(rng, g.save);
    let into = pick(rng, g.into);
    match skill {
        Skill::Read => {
            let n = rng.between(3, 7) as usize;
            let t = table(rng, n, &mut number);
            let (file, format, entries) = add(rng, &t, &mut files);
            let (key, value) = entries[rng.below(entries.len())].clone();
            let slot = Slot { file, format, key };
            let find = pick(rng, g.find);
            let instruction = match rng.below(2) {
                0 => format!("{find} {} and {save} it {into}.", of(rng, g, &slot)),
                _ => format!(
                    "In {}, {} the {} for {}; {save} it {into}.",
                    slot.file,
                    find.to_lowercase(),
                    pick(rng, g.value),
                    slot.key
                ),
            };
            Drawn {
                skill,
                instruction,
                files,
                executables: vec![],
                plan: Plan::Read { slot },
                check: Check::Answer { value },
            }
        }
        Skill::Chain => {
            let hops = rng.between(2, 4) as usize;
            // The key names of each step; the last file maps to a number.
            let mut keys: Vec<Vec<String>> = (0..=hops).map(|_| names(rng, 4)).collect();
            let path_keys: Vec<String> = keys.iter_mut().map(|ks| ks[0].clone()).collect();
            let mut first = None;
            let mut next = Vec::new();
            let mut value = String::new();
            for h in 0..hops {
                let mut entries: Vec<(String, String)> = keys[h]
                    .iter()
                    .map(|k| {
                        (
                            k.clone(),
                            if h + 1 == hops {
                                number(rng)
                            } else {
                                keys[h + 1][rng.below(4)].clone()
                            },
                        )
                    })
                    .collect();
                // The path's own step.
                entries[0].1 = if h + 1 == hops {
                    number(rng)
                } else {
                    path_keys[h + 1].clone()
                };
                if h + 1 == hops {
                    value = entries[0].1.clone();
                }
                rng.shuffle(&mut entries);
                let (p, f, _) = add(rng, &entries, &mut files);
                if h == 0 {
                    first = Some(Slot {
                        file: p,
                        format: f,
                        key: path_keys[0].clone(),
                    });
                } else {
                    next.push((p, f));
                }
            }
            let start = first.unwrap_or_else(|| unreachable!("a chain has a first file"));
            let rest: Vec<&str> = next.iter().map(|(p, _)| p.as_str()).collect();
            let find = pick(rng, g.find);
            let instruction = match rng.below(2) {
                0 => format!(
                    "{find} {} in {}. That value is a key in {}{}. Follow the chain to the final number and {save} it {into}.",
                    start.key, start.file, rest[0],
                    if rest.len() > 1 { format!(", and so on through {}", rest[1..].join(", ")) } else { String::new() }
                ),
                _ => format!(
                    "Starting from {} in {}, look each value up as a key in the next file ({}). {} the number you end with {into}.",
                    start.key, start.file, rest.join(", then "), capital(save)
                ),
            };
            Drawn {
                skill,
                instruction,
                files,
                executables: vec![],
                plan: Plan::Chain { start, next },
                check: Check::Answer { value },
            }
        }
        Skill::Compute => {
            let helper =
                pick(rng, &["calc", "mix", "blend", "merge", "fold", "combine"]).to_string();
            let style = rng.below(4);
            let (m, n, p) = (
                rng.between(3, 17),
                rng.between(3, 17),
                rng.between(101, 997),
            );
            let two_files = rng.coin();
            let a_entries = {
                let n = rng.between(2, 5) as usize;
                table(rng, n, &mut number)
            };
            let (fa, fmt_a, ea) = add(rng, &a_entries, &mut files);
            let (fb, fmt_b, eb) = if two_files {
                let b_entries = {
                    let n = rng.between(2, 5) as usize;
                    table(rng, n, &mut number)
                };
                add(rng, &b_entries, &mut files)
            } else {
                (fa.clone(), fmt_a, ea.clone())
            };
            let ia = rng.below(ea.len());
            let mut ib = rng.below(eb.len());
            if !two_files && eb.len() > 1 && ib == ia {
                ib = (ia + 1) % eb.len();
            }
            let a = Slot {
                file: fa,
                format: fmt_a,
                key: ea[ia].0.clone(),
            };
            let b = Slot {
                file: fb,
                format: fmt_b,
                key: eb[ib].0.clone(),
            };
            let (va, vb): (u64, u64) =
                (ea[ia].1.parse().unwrap_or(0), eb[ib].1.parse().unwrap_or(0));
            files.insert(helper.clone(), helper_script(&helper, style, m, n, p));
            let tool = format!("./{helper}");
            let instruction = match rng.below(2) {
                0 => format!("Use {tool} to combine {} with {}, in that order, and {save} the result {into}.", of(rng, g, &a), of(rng, g, &b)),
                _ => format!("Apply {tool} to {} and {} (first {}, then {}). {} its output {into}.", a.key, b.key, a.key, b.key, capital(save))
                    + &format!(" {} is in {}; {} is in {}.", a.key, a.file, b.key, b.file),
            };
            let value = ((va * m + vb * n) % p).to_string();
            Drawn {
                skill,
                instruction,
                files,
                executables: vec![helper.clone()],
                plan: Plan::Compute { helper, a, b },
                check: Check::Answer { value },
            }
        }
        Skill::Edit => {
            let target_entries = {
                let n = rng.between(3, 6) as usize;
                table(rng, n, &mut number)
            };
            let source_entries = {
                let n = rng.between(2, 5) as usize;
                table(rng, n, &mut number)
            };
            let (ft, fmt_t, et) = add(rng, &target_entries, &mut files);
            let (fs, fmt_s, es) = add(rng, &source_entries, &mut files);
            let it = rng.below(et.len());
            let is = rng.below(es.len());
            let target = Slot {
                file: ft.clone(),
                format: fmt_t,
                key: et[it].0.clone(),
            };
            let source = Slot {
                file: fs,
                format: fmt_s,
                key: es[is].0.clone(),
            };
            let mut entries: BTreeMap<String, String> = et.iter().cloned().collect();
            entries.insert(target.key.clone(), es[is].1.clone());
            let set = pick(
                rng,
                if held_out {
                    &["Change", "Overwrite"]
                } else {
                    &["Set", "Update", "Replace"]
                },
            );
            let instruction = match rng.below(2) {
                0 => format!(
                    "{set} {} in {} to {}. Leave the other entries as they are.",
                    target.key,
                    target.file,
                    of(rng, g, &source)
                ),
                _ => format!(
                    "{} holds {}. {set} it with {} and keep everything else in the file.",
                    target.file,
                    target.key,
                    of(rng, g, &source)
                ),
            };
            Drawn {
                skill,
                instruction,
                files,
                executables: vec![],
                plan: Plan::Edit { target, source },
                check: Check::File {
                    file: ft,
                    format: fmt_t,
                    entries,
                },
            }
        }
        Skill::Count => {
            let word_ = pick(rng, &["ERROR", "WARN", "TIMEOUT", "FAIL", "RETRY"]).to_string();
            let file = format!(
                "{}{}.log",
                pick(rng, &["", "logs/"]),
                word(rng, 1).to_lowercase()
            );
            let lines = rng.between(6, 20) as usize;
            let mut log = String::new();
            let mut hits = 0;
            for i in 0..lines {
                if rng.below(3) == 0 {
                    hits += 1;
                    log.push_str(&format!("{i:03} {word_} {}\n", word(rng, 1).to_lowercase()));
                } else {
                    log.push_str(&format!("{i:03} ok {}\n", word(rng, 1).to_lowercase()));
                }
            }
            files.insert(file.clone(), log);
            let instruction = match rng.below(2) {
                0 => format!(
                    "How many lines of {file} contain {word_}? {} the count {into}.",
                    capital(save)
                ),
                _ => format!(
                    "Count the lines in {file} that mention {word_} and {save} that number {into}."
                ),
            };
            Drawn {
                skill,
                instruction,
                files,
                executables: vec![],
                plan: Plan::Count { file, word: word_ },
                check: Check::Answer {
                    value: hits.to_string(),
                },
            }
        }
    }
}

fn capital(w: &str) -> String {
    let mut c = w.chars();
    c.next()
        .map_or_else(String::new, |f| f.to_uppercase().chain(c).collect())
}

/// A helper that combines two whole numbers by `(a * m + b * n) mod p`,
/// called in one of four argument styles, with `--help` printing its usage.
fn helper_script(name: &str, style: usize, m: u64, n: u64, p: u64) -> String {
    let (usage, parse) = match style {
        0 => (format!("./{name} A B"), "a=$1; b=$2".to_string()),
        1 => (format!("./{name} combine A B"), "[ \"$1\" = combine ] || { echo \"usage: $USAGE\" >&2; exit 2; }; a=$2; b=$3".to_string()),
        2 => (format!("./{name} --left A --right B"), "a=$2; b=$4; [ \"$1\" = --left ] && [ \"$3\" = --right ] || { echo \"usage: $USAGE\" >&2; exit 2; }".to_string()),
        _ => (format!("./{name} -a A -b B"), "a=$2; b=$4; [ \"$1\" = -a ] && [ \"$3\" = -b ] || { echo \"usage: $USAGE\" >&2; exit 2; }".to_string()),
    };
    format!(
        "#!/bin/sh\n# {name}: combines two whole numbers by this workspace's rule.\nUSAGE='{usage}'\n\
         if [ \"$1\" = --help ] || [ $# -eq 0 ]; then echo \"usage: $USAGE\"; echo \"Combines two whole numbers A and B.\"; exit 0; fi\n\
         {parse}\necho $(( ($a * {m} + $b * {n}) % {p} ))\n"
    )
}

/// How `./helper --help` says to call it with `a` and `b`: its usage line
/// with `A` and `B` replaced.
#[must_use]
pub fn usage_call(help: &str, a: &str, b: &str) -> Option<String> {
    let line = help
        .lines()
        .find_map(|l| l.trim().strip_prefix("usage: "))?;
    let words: Vec<String> = line
        .split_whitespace()
        .map(|w| match w {
            "A" => a.to_string(),
            "B" => b.to_string(),
            other => other.to_string(),
        })
        .collect();
    Some(words.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_helper_style_computes_its_rule_through_the_usage_it_prints() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        for style in 0..4 {
            let script = dir.path().join(format!("h{style}"));
            std::fs::write(
                &script,
                helper_script(&format!("h{style}"), style, 3, 5, 101),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            let run = |args: &[&str]| {
                let out = std::process::Command::new("sh")
                    .arg(&script)
                    .args(args)
                    .current_dir(dir.path())
                    .output()
                    .unwrap_or_else(|e| panic!("{e}"));
                String::from_utf8_lossy(&out.stdout).to_string()
            };
            let call = usage_call(&run(&["--help"]), "7", "9").unwrap_or_default();
            let args: Vec<&str> = call.split_whitespace().skip(1).collect();
            assert_eq!(
                run(&args).trim(),
                (7 * 3 + 9 * 5).to_string(),
                "style {style}: {call}"
            );
        }
    }

    #[test]
    fn held_out_requests_use_formats_and_words_training_never_does() {
        let seed = splinter_sdk::model::seed::draw().value();
        let mut rng = SplitMix64::new(seed);
        for _ in 0..40 {
            for skill in SKILLS {
                let d = draw(&mut rng, skill, true);
                for file in d.files.keys() {
                    assert!(
                        !TRAINING
                            .iter()
                            .any(|f| file.ends_with(&format!(".{}", f.extension())))
                            || file.ends_with(".txt") && skill == Skill::Count,
                        "{file}"
                    );
                }
                let t = draw(&mut rng, skill, false);
                for w in HELD_OUT_GRAMMAR.find.iter().chain(HELD_OUT_GRAMMAR.into) {
                    assert!(
                        !t.instruction.contains(w),
                        "seed {seed}: {w} in {}",
                        t.instruction
                    );
                }
            }
        }
    }
}
