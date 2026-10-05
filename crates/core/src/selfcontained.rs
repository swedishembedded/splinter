// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Whether an instruction stands on its own: the one rule for what a
//! student may be shown without the material it lacks.
//!
//! The reference answer is the target of a task, not context: dropping it
//! from the student's turn needs no justification, and neither does
//! dropping what grades an answer (executable checks, generated tests, and
//! the check that established a shown program's output) or a critique:
//! feedback on an earlier attempt, written after the instruction and about
//! it, so a critique quoting the instruction is no sign the instruction
//! depends on it. Dropping any other item does: the instruction must stand
//! on its own without it. [`check_self_contained`] decides that. Task
//! generators hold a generated instruction to it, and a training view
//! excludes an experience whose instruction fails it.

use crate::experience::{Privileged, PrivilegedKind};
use crate::kinds::{EXECUTABLE_CHECK, GENERATED_TEST, OUTPUT_CHECK};

/// Phrases by which an instruction points at material outside itself.
/// Matched case-insensitively with runs of whitespace collapsed. The list
/// errs towards excluding: a false match costs one training record, a
/// missed one trains the student on a question it cannot answer.
pub const REFERRING_PHRASES: &[&str] = &[
    "the passage above",
    "the text above",
    "the document above",
    "the code above",
    "the context above",
    "the above passage",
    "the above text",
    "the above code",
    "the following text",
    "the following passage",
    "the following document",
    "the given code",
    "the given text",
    "the given passage",
    "the given document",
    "the provided code",
    "the provided text",
    "the provided passage",
    "the provided document",
    "the attached",
    "according to the passage",
    "according to the text",
    "according to the document",
    "the writer",
    "the author",
    "the letter",
    "this letter",
];

/// The shortest run of words of a dropped privileged item that, found in the
/// instruction (case and punctuation aside), marks the instruction as
/// quoting it: the length at which a run of words is a claim to a passage
/// rather than the name of a matter. A question is told to name what it is
/// about in the source's own words - a title, a law, a place as the writer
/// calls it - and such a name is a noun phrase of a few words; a clause of
/// this many is lifted from the text.
pub const MIN_QUOTED_WORDS: usize = 8;

/// Why an instruction does not stand on its own without the privileged
/// context dropped from it.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NotSelfContained {
    /// It points at material outside itself.
    #[error("the instruction refers to material outside itself ({phrase:?})")]
    Refers {
        /// The phrase found.
        phrase: &'static str,
    },
    /// It quotes a dropped item.
    #[error("the instruction quotes the dropped {kind:?} item")]
    Quotes {
        /// The kind of the item quoted.
        kind: PrivilegedKind,
    },
}

/// Whether `item` is never context the instruction relies on: what grades
/// an answer (the reference, and the executable checks and generated tests
/// the executable verifiers run), and a critique of an earlier attempt.
fn never_context(item: &Privileged) -> bool {
    match &item.kind {
        PrivilegedKind::Reference | PrivilegedKind::Critique => true,
        PrivilegedKind::Other(name) => {
            [EXECUTABLE_CHECK, GENERATED_TEST, OUTPUT_CHECK].contains(&name.as_str())
        }
        _ => false,
    }
}

/// Whether `instruction` stands on its own once `dropped` - privileged
/// items left out of the student's turn - are gone. Dropped items that
/// grade the answer (the reference, executable checks, generated tests)
/// and critiques do not count: they are never context. With any other
/// item dropped, the instruction must neither contain one of
/// [`REFERRING_PHRASES`] nor repeat a run of [`MIN_QUOTED_WORDS`] or more
/// words of a dropped item.
pub fn check_self_contained(
    instruction: &str,
    dropped: &[&Privileged],
) -> Result<(), NotSelfContained> {
    let context: Vec<&Privileged> = dropped
        .iter()
        .copied()
        .filter(|item| !never_context(item))
        .collect();
    if context.is_empty() {
        return Ok(());
    }
    let normalised = instruction
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    if let Some(phrase) = REFERRING_PHRASES
        .iter()
        .find(|phrase| normalised.contains(*phrase))
    {
        return Err(NotSelfContained::Refers { phrase });
    }
    let asked = words(instruction);
    let quoted: std::collections::HashSet<&[String]> = asked.windows(MIN_QUOTED_WORDS).collect();
    if quoted.is_empty() {
        return Ok(());
    }
    for item in context {
        let content = words(&item.content);
        if content
            .windows(MIN_QUOTED_WORDS)
            .any(|run| quoted.contains(run))
        {
            return Err(NotSelfContained::Quotes {
                kind: item.kind.clone(),
            });
        }
    }
    Ok(())
}

/// The words of `text`, lower-cased, punctuation and whitespace aside: the
/// units a quoted run is counted in.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}
