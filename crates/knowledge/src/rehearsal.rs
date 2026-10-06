// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning that rehearses a model's own
// answers so new training does not cost it what it could already do, for
// its clients. If your team needs expertise in continual learning without
// catastrophic forgetting, you can procure our services by sending an email
// to info@swedishembedded.com.

//! General tasks a base model answers so that its own answers can be
//! rehearsed beside new training: what a persona fine-tune erodes first is
//! working a sum through and following a format, and after them plain
//! general knowledge, so a rehearsal set holds all three, with no corpus
//! from outside.
//!
//! * [`form_tasks`]: `arithmetic` and `format` tasks built by code from a
//!   seed, fresh numbers and subjects every time, each with its reference
//!   computed, so a wrong answer by the model can be told from a right one
//!   and never rehearsed. A sum is asked worked through or for the number
//!   alone, turn about. The templates are the generator's own, not an
//!   anchor suite's.
//! * [`GENERAL_BRIEF`] and [`GENERAL_DOMAINS`]: what a model is briefed with
//!   to write general requests of a domain, in place of any corpus. What it
//!   answers them is a [`splinter_core::kinds::GENERAL`] task: no reference,
//!   graded by nothing.
//! * [`Admission`]: the one rule every rehearsal task passes before it is
//!   stored. The anchor suite is the release gate's frozen check of general
//!   behaviour; a candidate rehearsed on an anchor task, or a near copy of
//!   one, would pass that check by having trained on it. So every anchor
//!   instruction is admitted to the generator's near-duplicate rule
//!   ([`Seen`]) first, and a task that repeats one - the same text once
//!   normalised, an overlap of word shingles at the generator's threshold,
//!   lying inside one or holding one within it - is refused, as is a task
//!   that repeats one already admitted.

use splinter_core::experience::{Environment, Privileged, PrivilegedKind, Task};

use crate::seeded::SplitMix64;
use crate::tasks::dedup::{Repeat, Seen};
use crate::tasks::{DEFAULT_MAX_OVERLAP, DEFAULT_SHINGLE_WORDS};

/// The generator's name in a task set's provenance.
pub const GENERATOR: &str = "splinter-knowledge/rehearsal@1";

/// How an `arithmetic` instruction ends, as the kind's brief has every such
/// task end: what tells the model to state its answer last.
pub const ARITHMETIC_ENDING: &str =
    "Work it out, then end your reply with 'The answer is' and the number.";

/// The other ending an arithmetic task is given, turn about with
/// [`ARITHMETIC_ENDING`]: the number alone, as a question of fact is
/// answered. A fine-tune that answers at length where a number was asked is
/// held to this as much as to the sum.
pub const TERSE_ENDING: &str = "Answer with the number.";

/// The brief a model writes general requests from: one domain at a time,
/// the requests an assistant is put every day, of varied form, each
/// standing on its own.
pub const GENERAL_BRIEF: &str = "You write requests that a user might put to a general-purpose \
    assistant, in the domain given. Each request stands on its own in one to three sentences and \
    is of the kind a capable assistant answers directly, in a paragraph or a short list. Vary the \
    form across the requests: a question of fact, a request for an explanation, for advice on a \
    concrete situation, for a short piece of writing, for a comparison of two things, for steps \
    to follow, for a small calculation. Do not ask about any one person's life or opinions, do \
    not refer to a document the assistant would need, and do not number the requests. Reply with \
    JSON of exactly this shape: {\"prompts\": [\"...\", \"...\"]}.";

/// The domains general requests are written in, one request batch each,
/// so a rehearsal set spans what an assistant is asked rather than one
/// corner of it.
pub const GENERAL_DOMAINS: &[&str] = &[
    "science and nature",
    "history and geography",
    "health, fitness and first aid",
    "cooking and food",
    "personal finance and budgeting",
    "everyday use of computers, phones and software",
    "programming and working with data",
    "mathematics and logic puzzles",
    "language, grammar and writing",
    "travel and transport",
    "home, garden and repairs",
    "work, careers and study",
    "arts, music and literature",
    "sports and games",
    "law, government and civics in general terms",
    "planning and everyday decisions",
];

/// What a model that writes general requests is shown with the brief.
#[derive(Clone, Debug, serde::Serialize)]
pub struct GeneralRequest<'a> {
    /// The domain the requests are of.
    pub domain: &'a str,
    /// How many requests to write.
    pub count: usize,
}

/// Why a rehearsal task was not admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// It repeats an excluded instruction - an anchor task - or lies inside
    /// one.
    Excluded,
    /// It repeats a task already admitted.
    Repeat,
}

/// The admission every rehearsal task passes; see the module documentation.
pub struct Admission {
    excluded: Seen,
    admitted: Seen,
    /// Tasks refused for repeating an excluded instruction.
    pub excluded_count: usize,
    /// Tasks refused for repeating an admitted one.
    pub repeated_count: usize,
}

impl Admission {
    /// An admission that refuses every one of `excluded` and its near
    /// copies.
    pub fn new<'a>(excluded: impl IntoIterator<Item = &'a str>) -> Self {
        let mut seen = Seen::new(DEFAULT_SHINGLE_WORDS, DEFAULT_MAX_OVERLAP);
        for instruction in excluded {
            seen.admit(instruction);
        }
        Self {
            excluded: seen,
            admitted: Seen::new(DEFAULT_SHINGLE_WORDS, DEFAULT_MAX_OVERLAP),
            excluded_count: 0,
            repeated_count: 0,
        }
    }

    /// Admits `instruction`, or says why not. An admitted instruction is
    /// then one later ones must not repeat.
    pub fn admit(&mut self, instruction: &str) -> Result<(), Refusal> {
        if self.excluded.leaks(instruction).is_some() || self.excluded.encloses(instruction) {
            self.excluded_count += 1;
            return Err(Refusal::Excluded);
        }
        match self.admitted.repeats(instruction) {
            Some(Repeat::Exact | Repeat::Near) => {
                self.repeated_count += 1;
                Err(Refusal::Repeat)
            }
            None => {
                self.admitted.admit(instruction);
                Ok(())
            }
        }
    }
}

/// `count` form tasks from `seed`: `arithmetic` and `format` turn about,
/// each instance new for its numbers and subjects. The same seed and count
/// give the same tasks.
#[must_use]
pub fn form_tasks(seed: u64, count: usize) -> Vec<Task> {
    let mut rng = SplitMix64(seed);
    let mut tasks = Vec::with_capacity(count);
    for n in 0..count {
        let (kind, instruction, reference) = if n % 2 == 0 {
            let (instruction, answer) = arithmetic(&mut rng);
            let ending = if n % 4 == 0 {
                ARITHMETIC_ENDING
            } else {
                TERSE_ENDING
            };
            ("arithmetic", format!("{instruction} {ending}"), answer)
        } else {
            let (instruction, lines) = format_request(&mut rng);
            ("format", instruction, lines.to_string())
        };
        let task = Task::new(
            kind,
            Vec::new(),
            Environment::closed_book(),
            instruction,
            vec![Privileged {
                kind: PrivilegedKind::Reference,
                content: reference,
                span: None,
            }],
        );
        match task {
            Ok(task) => tasks.push(task),
            Err(_) => unreachable!("a form task has a kind, an instruction and a reference"),
        }
    }
    tasks
}

/// A number as a reference states it: whole when it is, else with its
/// decimals and no trailing zeros.
fn number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        let text = format!("{value:.2}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// One arithmetic word problem and its answer, from one of the templates.
fn arithmetic(rng: &mut SplitMix64) -> (String, String) {
    let pick = |rng: &mut SplitMix64, items: &[&str]| items[rng.below(items.len())].to_string();
    let goods = [
        "bottles",
        "tiles",
        "notebooks",
        "candles",
        "oranges",
        "screws",
    ];
    let names = ["Lena", "Omar", "Priya", "Jonas", "Amara", "Felix"];
    match rng.below(10) {
        0 => {
            let (a, b, c) = (rng.between(6, 24), rng.between(3, 12), rng.between(1, 9));
            let what = pick(rng, &goods);
            (
                format!(
                    "A crate holds {a} {what}. {b} crates are delivered and {c} of the {what} are \
                     damaged. How many undamaged {what} are delivered?"
                ),
                number((a * b - c) as f64),
            )
        }
        1 => {
            let (a, b) = (rng.between(3, 40), rng.between(4, 25));
            (
                format!(
                    "A printer prints {a} pages a minute. How many pages does it print in {b} \
                     minutes?"
                ),
                number((a * b) as f64),
            )
        }
        2 => {
            let (b, q) = (rng.between(2, 9), rng.between(3, 15));
            let a = b * q;
            (
                format!(
                    "A rope {a} metres long is cut into pieces of {b} metres. How many pieces are \
                     there?"
                ),
                number(q as f64),
            )
        }
        3 => {
            let (a, b, c, d) = (
                rng.between(8, 30),
                rng.between(3, 15),
                rng.between(1, 4),
                rng.between(1, 5),
            );
            (
                format!(
                    "A museum ticket costs {a} dollars for an adult and {b} dollars for a child. \
                     What do {c} adults and {d} children pay together, in dollars?"
                ),
                number((a * c + b * d) as f64),
            )
        }
        4 => {
            let (b, q) = (rng.between(2, 7), rng.between(4, 30));
            let a = b * q;
            let who = pick(rng, &names);
            (
                format!(
                    "{who} ran {a} kilometres over {b} days, the same distance each day. How many \
                     kilometres a day was that?"
                ),
                number(q as f64),
            )
        }
        5 => {
            let (a, b) = (rng.between(4, 30), rng.between(4, 30));
            (
                format!(
                    "A field is {a} metres long and {b} metres wide. What is its area in square \
                     metres?"
                ),
                number((a * b) as f64),
            )
        }
        6 => {
            let (h1, m1) = (rng.between(6, 11), rng.between(0, 5) * 10);
            let (hours, minutes) = (rng.between(1, 5), rng.between(0, 5) * 5);
            let total = h1 * 60 + m1 + hours * 60 + minutes;
            let (h2, m2) = (total / 60, total % 60);
            (
                format!(
                    "A lesson starts at {h1}:{m1:02} and ends at {h2}:{m2:02}. How many minutes \
                     long is it?"
                ),
                number((hours * 60 + minutes) as f64),
            )
        }
        7 => {
            let (a, b, c) = (rng.between(3, 9), rng.between(5, 20), rng.between(2, 15));
            let what = pick(rng, &goods);
            let sold = (a * b).min(c + 1) - 1;
            (
                format!(
                    "A shop has {a} shelves with {b} {what} on each and sells {sold} of them. How \
                     many {what} are left?"
                ),
                number((a * b - sold) as f64),
            )
        }
        8 => {
            let (a, b) = (rng.between(3, 9) * 25, rng.between(2, 12));
            (
                format!("A bus ticket costs {a} cents. How many dollars do {b} tickets cost?"),
                number((a * b) as f64 / 100.0),
            )
        }
        _ => {
            let (a, b, c) = (rng.between(5, 40), rng.between(5, 40), rng.between(5, 40));
            let total = a + b + c;
            let who = pick(rng, &names);
            (
                format!(
                    "{who} scored {a}, {b} and {c} points in three games. How many points did \
                     {who} score in all?"
                ),
                number(total as f64),
            )
        }
    }
}

/// One request for a set number of items, one per line, and that number.
fn format_request(rng: &mut SplitMix64) -> (String, u64) {
    let categories = [
        "birds",
        "trees",
        "colours",
        "sports played with a ball",
        "board games",
        "kitchen utensils",
        "dog breeds",
        "flowers",
        "mountain ranges",
        "languages spoken in Asia",
        "shapes",
        "dairy products",
        "hand tools",
        "styles of music",
        "parts of a car",
        "insects",
        "cereal grains",
        "capital cities",
        "units of length",
        "household appliances",
    ];
    let count = rng.between(2, 7);
    let word = ["two", "three", "four", "five", "six", "seven"][(count - 2) as usize];
    let category = categories[rng.below(categories.len())];
    let instruction = match rng.below(3) {
        0 => format!("List {word} {category}, one per line, and write nothing else."),
        1 => format!(
            "Write down exactly {word} {category}, each on a line of its own, with no other \
             words."
        ),
        _ => format!(
            "I need {word} examples of {category}. Put each on its own line and add no other \
             text."
        ),
    };
    (instruction, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::Catalogue;
    use splinter_eval::verifiers::form::{final_number_is, line_count_is};

    /// Form tasks are the catalogue's two form kinds, each with a reference
    /// its own verifier reads and a worked answer satisfies; the same seed
    /// gives the same tasks, another seed other numbers.
    #[test]
    fn form_tasks_are_graded_by_their_kinds_verifiers_and_follow_their_seed() {
        let catalogue = Catalogue::builtin();
        let tasks = form_tasks(7, 40);
        assert_eq!(tasks.len(), 40);
        for task in &tasks {
            assert!(
                catalogue.get(&task.task.kind).is_some(),
                "{}",
                task.task.kind
            );
            let reference = &task.privileged[0].content;
            match task.task.kind.as_str() {
                "arithmetic" => {
                    assert!(
                        task.instruction.ends_with(ARITHMETIC_ENDING)
                            || task.instruction.ends_with(TERSE_ENDING)
                    );
                    assert_eq!(
                        final_number_is(&format!("The answer is {reference}"), reference),
                        Some(true),
                        "{}",
                        task.instruction
                    );
                }
                "format" => {
                    let lines: usize = reference.parse().unwrap_or_else(|_| panic!("{reference}"));
                    assert_eq!(
                        line_count_is(&vec!["x"; lines].join("\n"), reference),
                        Some(true)
                    );
                }
                other => panic!("{other}"),
            }
        }
        assert_eq!(form_tasks(7, 40), tasks, "a seed means the same tasks");
        assert_eq!(
            form_tasks(7, 20),
            tasks[..20],
            "a longer count extends the sequence"
        );
        assert!(tasks.iter().any(|t| t.instruction.ends_with(TERSE_ENDING)));
        assert!(tasks
            .iter()
            .any(|t| t.instruction.ends_with(ARITHMETIC_ENDING)));
        assert_ne!(form_tasks(8, 40), tasks, "another seed, other tasks");
        let distinct: std::collections::BTreeSet<&str> =
            tasks.iter().map(|t| t.instruction.as_str()).collect();
        assert!(distinct.len() >= 30, "{} distinct of 40", distinct.len());
    }

    /// Nothing an anchor suite holds, nor a near copy of it, is admitted:
    /// not the same text, not the same text in other case and spacing, not a
    /// close rewording; and a repeat of an admitted task is refused too.
    #[test]
    fn an_anchor_task_and_its_near_copies_are_refused() {
        let anchor = [
            "What is the capital of France?",
            "Name three fruits. Give exactly three fruits, one per line, and no other text.",
        ];
        let mut admission = Admission::new(anchor);
        assert_eq!(admission.admit(anchor[0]), Err(Refusal::Excluded));
        assert_eq!(
            admission.admit("  what is the CAPITAL of france?  "),
            Err(Refusal::Excluded),
            "case and spacing"
        );
        assert_eq!(
            admission.admit(
                "Name three fruits. Give exactly three fruits, one per line, and no other text \
                 at all."
            ),
            Err(Refusal::Excluded),
            "a close rewording"
        );
        assert_eq!(
            admission.admit("Quick question before we start: what is the capital of France?"),
            Err(Refusal::Excluded),
            "the anchor's words inside a longer prompt"
        );
        assert_eq!(admission.admit("Explain how a bicycle pump works."), Ok(()));
        assert_eq!(
            admission.admit("Explain how a bicycle  pump works."),
            Err(Refusal::Repeat)
        );
        assert_eq!(admission.excluded_count, 4);
        assert_eq!(admission.repeated_count, 1);
    }
}
