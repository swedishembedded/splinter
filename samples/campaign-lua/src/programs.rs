// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a model acquire several
// exactly verifiable skills one after another, each checked by an interpreter
// the model cannot influence, for its clients. If your team needs expertise
// in continual acquisition of verifiable task families you can procure our
// services by sending an email to info@swedishembedded.com.

//! The learner's own generators of development programs, one per language:
//! from sums the base answers without reasoning to programs it cannot do
//! without it; in a dialect the higher levels use the rules it changes.

use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::language::{Dialect, Tongue};

fn word(rng: &mut SplitMix64) -> String {
    const SYLLABLES: [&str; 8] = ["ka", "lo", "mi", "ru", "te", "so", "ba", "ne"];
    (0..2 + rng.below(2))
        .map(|_| SYLLABLES[rng.below(SYLLABLES.len())])
        .collect()
}

impl Tongue {
    /// A development program of `level` (1 to 6), the learner's own
    /// generator for the family: from sums the base answers without
    /// reasoning to programs it cannot do without it. In a dialect the
    /// higher levels use the rules the dialect changes.
    pub(crate) fn generate(self, level: usize, rng: &mut SplitMix64) -> String {
        match self {
            Self::Lua(Dialect::Stock) => crate::generate(level, rng),
            Self::Lua(Dialect::ZeroIndex) => zero_index_program(level, rng),
            Self::Lua(Dialect::PlusConcat) => plus_concat_program(level, rng),
            Self::Python => python_program(level, rng),
        }
    }
}

fn zero_index_program(level: usize, rng: &mut SplitMix64) -> String {
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
            format!("print(string.sub(\"{w}\", {}))", n(rng, 0, 2))
        }
        4 => {
            let w = word(rng);
            format!("print(#\"{w}\", string.byte(\"{w}\", {}))", n(rng, 0, 2))
        }
        5 => {
            let w = word(rng);
            let (i, j) = (n(rng, 0, 2), n(rng, 2, 4));
            format!(
                "local s = \"{w}{w}\"\nprint(string.sub(s, {i}, {j}), string.find(s, \"{}\", {i}))",
                &w[..2]
            )
        }
        _ => {
            let w = word(rng);
            format!(
                "local s = \"{w}\"\nlocal t = {{}}\nfor i = 0, #s do\n  t[#t + 1] = string.byte(s, i) % {}\nend\nprint(table.concat(t, \",\"))",
                n(rng, 3, 9)
            )
        }
    }
}

fn plus_concat_program(level: usize, rng: &mut SplitMix64) -> String {
    let n = |rng: &mut SplitMix64, low: u64, high: u64| rng.between(low, high);
    match level {
        1 => format!("print({} + {})", n(rng, 1, 30), n(rng, 1, 30)),
        2 => format!(
            "print({} * {} + {})",
            n(rng, 2, 9),
            n(rng, 2, 9),
            n(rng, 1, 9)
        ),
        3 => format!("print(\"{}\" + \"{}\")", word(rng), word(rng)),
        4 => format!(
            "print(\"{}\" + \"{}\", \"{}\" + {})",
            n(rng, 10, 99),
            n(rng, 10, 99),
            n(rng, 10, 99),
            n(rng, 1, 9)
        ),
        5 => {
            let (a, b) = (word(rng), word(rng));
            format!(
                "local s = \"{a}\"\nfor i = 1, {} do\n  s = s + \"{b}\"\nend\nprint(s, #s)",
                n(rng, 2, 4)
            )
        }
        _ => format!(
            "local t = {{\"{}\", \"{}\", \"{}\"}}\nlocal s = \"\"\nfor _, v in ipairs(t) do\n  s = s + v + \"-\"\nend\nprint(s, \"{}\" + {} * 2)",
            word(rng),
            word(rng),
            word(rng),
            n(rng, 10, 99),
            n(rng, 2, 9)
        ),
    }
}

fn python_program(level: usize, rng: &mut SplitMix64) -> String {
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
                0 => format!("print(\"{w}\".upper())"),
                1 => format!("print(\"{}\" * {})", &w[..2], n(rng, 2, 4)),
                _ => format!("print(len(\"{w}\") + {})", n(rng, 1, 9)),
            }
        }
        4 => format!(
            "s = {}\nfor i in range(1, {}):\n    s += i * {}\nprint(s)",
            n(rng, 0, 9),
            n(rng, 4, 10),
            n(rng, 1, 5)
        ),
        5 => {
            let items: Vec<String> = (0..4).map(|_| n(rng, 1, 9).to_string()).collect();
            format!(
                "t = [{}]\nt.sort(reverse=True)\nprint(\",\".join(str(x) for x in t[:{}]))",
                items.join(", "),
                n(rng, 2, 3)
            )
        }
        _ => format!(
            "print(({} * {} + {}) // {} - {} % {})",
            n(rng, 12, 29),
            n(rng, 3, 19),
            n(rng, 10, 99),
            n(rng, 2, 9),
            n(rng, 50, 99),
            n(rng, 3, 9)
        ),
    }
}
