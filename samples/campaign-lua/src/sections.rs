// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that keep a self-generating
// training loop from saturating on the same few topics, for its clients. If
// your team needs expertise in steering synthetic task generation toward what
// a model still gets wrong you can procure our services by sending an email
// to info@swedishembedded.com.

//! What the policy is asked to propose from: the sections of the manual, and
//! which of them to ask about next.
//!
//! A model asked for programs over the same windows of the manual every time
//! saturates after a few hundred distinct programs. Here every request shows
//! one section, and the sections are chosen so that the standard-library
//! functions the current policy answers worst on its dev set are asked about
//! first and most, while no section is starved.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context as _, Result};
use splinter_sdk::knowledge::concepts::SectionRef;
use splinter_sdk::knowledge::tasks::SourceText;
use splinter_sdk::vocabulary::source::SourceId;
use splinter_sdk::Context;

use crate::language::{Dialect, Tongue};
use crate::vocabulary;

/// Standard libraries whose functions are families of programs.
const LIBRARIES: [&str; 5] = ["string", "table", "math", "os", "io"];

/// The functions of the string library, which a program may call as methods
/// (`s:rep(3)`).
const STRING_METHODS: [&str; 17] = [
    "byte", "char", "dump", "find", "format", "gmatch", "gsub", "len", "lower", "match", "rep",
    "reverse", "sub", "upper", "gfind", "lines", "write",
];

/// A family needs this many dev programs before its accuracy steers anything.
pub const MIN_FAMILY_ITEMS: usize = 3;

/// A family whose dev accuracy is below this is named to the policy as one to
/// write programs about.
const WEAK_BELOW: f64 = 0.5;

/// One section of a manual page.
#[derive(Clone, Debug)]
pub struct ManualSection {
    /// Where it is.
    pub at: SectionRef,
    /// The standard-library functions it mentions, as `string.format`.
    pub functions: BTreeSet<String>,
}

impl ManualSection {
    /// The key a proposal records for it: `part#position` of its source.
    #[must_use]
    pub fn key(&self) -> String {
        format!("{}#{}", self.at.part, self.at.section)
    }
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// The library functions `text` calls or mentions: `string.format`,
/// `table.sort`, and a string method call `s:rep(` as `string.rep`.
#[must_use]
pub fn functions_in(text: &str) -> BTreeSet<String> {
    let bytes = text.as_bytes();
    let mut found = BTreeSet::new();
    let ident_at = |from: usize| {
        let end = bytes[from..]
            .iter()
            .position(|&b| !is_ident(b))
            .map_or(bytes.len(), |n| from + n);
        (&text[from..end], end)
    };
    let mut i = 0;
    while i < bytes.len() {
        let starts_word = bytes[i].is_ascii_alphabetic()
            && (i == 0 || (!is_ident(bytes[i - 1]) && bytes[i - 1] != b'.'));
        if starts_word {
            let (word, end) = ident_at(i);
            let is_call = LIBRARIES.contains(&word)
                && bytes.get(end) == Some(&b'.')
                && bytes.get(end + 1).is_some_and(u8::is_ascii_alphabetic);
            if is_call {
                let (name, after) = ident_at(end + 1);
                found.insert(format!("{word}.{name}"));
                i = after;
            } else {
                i = end;
            }
        } else if bytes[i] == b':' && bytes.get(i + 1).is_some_and(u8::is_ascii_alphabetic) {
            let (name, end) = ident_at(i + 1);
            if STRING_METHODS.contains(&name) && bytes.get(end) == Some(&b'(') {
                found.insert(format!("string.{name}"));
            }
            i = end;
        } else {
            i += 1;
        }
    }
    found
}

/// Every section of the manual pages in `sources`, with the functions each
/// mentions.
///
/// # Errors
/// A page cannot be read from the source store.
pub fn manual_sections(
    ctx: &Context,
    sources: &[SourceId],
    tongue: Tongue,
) -> Result<Vec<ManualSection>> {
    let mut out = Vec::new();
    for id in sources {
        let source = ctx
            .sources()
            .get_source(id)
            .with_context(|| format!("reading source {id}"))?;
        for part in &source.parts {
            let text = SourceText::load(&ctx.sources(), id, &part.name)
                .with_context(|| format!("splitting {}", part.name))?;
            for position in 0..text.sections().len() {
                let body = text.section_text(position).unwrap_or_default();
                out.push(ManualSection {
                    at: SectionRef {
                        source: id.clone(),
                        part: part.name.clone(),
                        section: position,
                    },
                    functions: vocabulary::functions_in(tongue, body),
                });
            }
        }
    }
    Ok(out)
}

/// A family's dev accuracy: programs answered right of programs asked.
pub type Accuracy = BTreeMap<String, (usize, usize)>;

/// The dev accuracy of each library function over `outcomes` (a program and
/// whether the policy answered it right); a program counts for every
/// function it uses.
#[must_use]
pub fn family_accuracy<'a>(outcomes: impl IntoIterator<Item = (&'a str, bool)>) -> Accuracy {
    family_accuracy_in(Tongue::Lua(Dialect::Stock), outcomes)
}

/// [`family_accuracy`] for programs written in `tongue`.
#[must_use]
pub fn family_accuracy_in<'a>(
    tongue: Tongue,
    outcomes: impl IntoIterator<Item = (&'a str, bool)>,
) -> Accuracy {
    let mut by_family = Accuracy::new();
    for (program, passed) in outcomes {
        for family in vocabulary::functions_in(tongue, program) {
            let entry = by_family.entry(family).or_default();
            entry.0 += usize::from(passed);
            entry.1 += 1;
        }
    }
    by_family
}

/// The weakness of a family, in `[0, 1]`: one minus its accuracy, zero while
/// the dev set holds too few programs of it to say.
fn weakness(accuracy: &Accuracy, family: &str) -> f64 {
    match accuracy.get(family) {
        Some(&(right, asked)) if asked >= MIN_FAMILY_ITEMS => 1.0 - right as f64 / asked as f64,
        _ => 0.0,
    }
}

/// The functions of `section` the policy answers worst on (below
/// `WEAK_BELOW` with enough dev programs), worst first.
#[must_use]
pub fn weak_functions(section: &ManualSection, accuracy: &Accuracy) -> Vec<String> {
    let mut weak: Vec<(f64, &String)> = section
        .functions
        .iter()
        .map(|f| (weakness(accuracy, f), f))
        .filter(|(w, _)| *w > 1.0 - WEAK_BELOW)
        .collect();
    weak.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(b.1)));
    weak.into_iter().map(|(_, f)| f.clone()).collect()
}

/// The sections to ask about, `requests` of them in order: each pick is the
/// section with the fewest proposals made from it so far (and picks of this
/// plan) per unit of weight, where a section's weight is `1 + 3w` for the
/// weakness `w` of the worst family it mentions. Every section is asked
/// before any is asked a second time unless its weight says otherwise, and
/// ties go to the earlier section.
#[must_use]
pub fn plan_requests(
    sections: &[ManualSection],
    asked: &BTreeMap<String, usize>,
    accuracy: &Accuracy,
    requests: usize,
) -> Vec<usize> {
    let weights: Vec<f64> = sections
        .iter()
        .map(|s| {
            let worst = s
                .functions
                .iter()
                .map(|f| weakness(accuracy, f))
                .fold(0.0, f64::max);
            1.0 + 3.0 * worst
        })
        .collect();
    let mut counts: Vec<f64> = sections
        .iter()
        .map(|s| asked.get(&s.key()).copied().unwrap_or(0) as f64)
        .collect();
    let mut plan = Vec::with_capacity(requests);
    for _ in 0..requests {
        let pick = (0..sections.len()).min_by(|&a, &b| {
            (counts[a] / weights[a])
                .total_cmp(&(counts[b] / weights[b]))
                .then(a.cmp(&b))
        });
        let Some(pick) = pick else { break };
        counts[pick] += 1.0;
        plan.push(pick);
    }
    plan
}

#[cfg(test)]
mod tests {
    use splinter_sdk::vocabulary::digest::Digest;

    use super::*;

    fn section(position: usize, functions: &[&str]) -> ManualSection {
        ManualSection {
            at: SectionRef {
                source: SourceId(Digest::of(b"a manual page")),
                part: "page.html".into(),
                section: position,
            },
            functions: functions.iter().map(|f| (*f).to_string()).collect(),
        }
    }

    #[test]
    fn library_functions_are_found_as_calls_and_as_string_methods() {
        let found = functions_in("local s = string.format('%d', 3)\nprint(s:rep(2), table.concat(t), math.floor(1.5))\nx:len\nmystring.format");
        let expect: BTreeSet<String> =
            ["string.format", "string.rep", "table.concat", "math.floor"]
                .iter()
                .map(|s| (*s).to_string())
                .collect();
        assert_eq!(
            found, expect,
            "`:len` without a call and a longer word are not calls"
        );
    }

    #[test]
    fn a_family_is_weak_only_when_the_dev_set_holds_enough_of_it() {
        let accuracy = family_accuracy([
            ("print(string.rep('a', 2))", false),
            ("print(string.rep('b', 2))", false),
            ("print(string.rep('c', 2))", false),
            ("print(table.concat({1}))", false),
            ("print(math.floor(2.5))", true),
            ("print(math.floor(3.5))", true),
            ("print(math.floor(4.5))", true),
        ]);
        let sections = [
            section(0, &["string.rep", "table.concat"]),
            section(1, &["math.floor"]),
        ];
        assert_eq!(weak_functions(&sections[0], &accuracy), vec!["string.rep"]);
        assert!(weak_functions(&sections[1], &accuracy).is_empty());
    }

    #[test]
    fn every_section_is_asked_before_any_is_asked_twice_and_weak_ones_come_back_sooner() {
        let sections = [
            section(0, &["string.rep"]),
            section(1, &["math.floor"]),
            section(2, &["table.sort"]),
        ];
        let even = plan_requests(&sections, &BTreeMap::new(), &Accuracy::new(), 6);
        assert_eq!(even, vec![0, 1, 2, 0, 1, 2]);
        let accuracy = family_accuracy((0..4).map(|_| ("print(string.rep('a', 2))", false)));
        let weighted = plan_requests(&sections, &BTreeMap::new(), &accuracy, 8);
        let of = |n| weighted.iter().filter(|&&p| p == n).count();
        assert!(
            of(0) > of(1) && of(0) > of(2),
            "weak section asked most: {weighted:?}"
        );
        assert!(
            of(1) >= 1 && of(2) >= 1,
            "no section is starved: {weighted:?}"
        );
        let asked = BTreeMap::from([(sections[0].key(), 5), (sections[1].key(), 5)]);
        let resumed = plan_requests(&sections, &asked, &Accuracy::new(), 1);
        assert_eq!(resumed, vec![2], "proposals made earlier count");
    }
}
