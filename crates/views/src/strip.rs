// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a student sees of the teacher's material.
//!
//! An experience separates the instruction (what the student is shown)
//! from privileged items (what only the teacher saw: a passage, a hint, a
//! critique, the reference answer, an oracle's output). A [`Strip`] policy
//! decides which privileged items a view puts into the student's turn:
//!
//! * [`Strip::All`], the default: none; the student's turn is the
//!   instruction alone.
//! * [`Strip::Keep`]: the items of the listed kinds, each as a labelled
//!   block before the instruction (`Hint:\n<content>\n\n<instruction>`).
//!   Context distillation is then off for those kinds: the student learns
//!   with them in view.
//! * [`Strip::Mix`]: for a fixed fraction of subjects, every privileged
//!   item except the reference answer, as `Keep` renders it; for the rest,
//!   none. Which subjects is decided by a seeded hash of the subject's id
//!   (an experience's, or a task's for a view that reads tasks), so the
//!   choice is the same on every run and in every corpus order.
//!
//! The reference answer is the target of a task, not context: dropping it
//! from the student's turn needs no justification, and neither does
//! dropping what grades an answer (executable checks, generated tests, and
//! the check that established a shown program's output) or a critique:
//! feedback on an earlier attempt, written after the instruction and about
//! it, so a critique quoting the instruction is no sign the instruction
//! depends on it. Dropping any other item does: the instruction must stand
//! on its own without it.
//! [`check_self_contained`] decides that, and a view excludes an
//! experience whose instruction fails it, counting it as
//! [`Exclusion::NotSelfContained`].

use serde::{Deserialize, Serialize};
use splinter_lab::verifiers::executable::{CHECK_KIND, OUTPUT_CHECK_KIND};
use splinter_lab::verifiers::mutation::TEST_KIND;
use splinter_store::digest::Digest;
use splinter_store::experience::{Privileged, PrivilegedKind};

use crate::{Exclusion, ViewError};

/// The domain of `mix:F`'s draw, which picks the subjects that keep their
/// privileged context.
const MIX_DOMAIN: &str = "splinter-views/strip-mix";

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
];

/// The shortest run of characters of a dropped privileged item that, found
/// verbatim in the instruction, marks the instruction as quoting it: long
/// enough that ordinary shared wording does not count, short enough to
/// catch a quoted clause.
pub const MIN_QUOTED_CHARS: usize = 24;

/// A share of subjects, in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Fraction(f64);

impl Fraction {
    /// `value` as a fraction; refused unless it is in `[0, 1]`.
    pub fn new(value: f64) -> Result<Self, ViewError> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(ViewError::Parameter {
                name: "keep_fraction",
                reason: format!("{value} is not in [0, 1]"),
            })
        }
    }

    /// The value.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Fraction {
    type Error = ViewError;

    fn try_from(value: f64) -> Result<Self, ViewError> {
        Self::new(value)
    }
}

impl From<Fraction> for f64 {
    fn from(fraction: Fraction) -> Self {
        fraction.0
    }
}

/// Which privileged items a view puts into the student's turn; see the
/// module documentation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strip {
    /// None: the student sees only the instruction.
    #[default]
    All,
    /// The items of these kinds.
    Keep(Vec<PrivilegedKind>),
    /// Every item but the reference, for `keep_fraction` of subjects.
    Mix {
        /// The share of subjects that keep their privileged context.
        keep_fraction: Fraction,
        /// Seeds the hash that picks them.
        seed: u64,
    },
}

impl Strip {
    /// Whether `item` goes into the student's turn of `subject`.
    fn keeps(&self, subject: &Digest, item: &Privileged) -> bool {
        match self {
            Self::All => false,
            Self::Keep(kinds) => kinds.contains(&item.kind),
            Self::Mix {
                keep_fraction,
                seed,
            } => {
                item.kind != PrivilegedKind::Reference
                    && draw(MIX_DOMAIN, *seed, subject) < keep_fraction.get()
            }
        }
    }

    /// The student's turn for `instruction` with `privileged` beside it:
    /// the kept items as labelled blocks, then the instruction. Excluded
    /// when privileged context is dropped from an instruction that is not
    /// self-contained without it.
    pub(crate) fn student_turn(
        &self,
        subject: &Digest,
        instruction: &str,
        privileged: &[Privileged],
    ) -> Result<String, Exclusion> {
        let (kept, dropped): (Vec<&Privileged>, Vec<&Privileged>) = privileged
            .iter()
            .partition(|item| self.keeps(subject, item));
        check_self_contained(instruction, &dropped).map_err(|_| Exclusion::NotSelfContained)?;
        let mut turn = String::new();
        for item in kept {
            turn.push_str(&label(&item.kind));
            turn.push_str(":\n");
            turn.push_str(&item.content);
            turn.push_str("\n\n");
        }
        turn.push_str(instruction);
        Ok(turn)
    }
}

/// A uniform draw in `[0, 1)` determined by `domain`, `seed` and `subject`
/// alone; `domain` keeps two uses of one seed independent.
pub(crate) fn draw(domain: &str, seed: u64, subject: &Digest) -> f64 {
    let hashed = Digest::of(format!("{domain}\n{seed}\n{subject}").as_bytes());
    // A digest's hex is 64 validated lowercase hex digits, so every one
    // converts; the first 16 are 64 uniform bits.
    let bits = hashed.hex().chars().take(16).fold(0u64, |acc, c| {
        (acc << 4) | u64::from(c.to_digit(16).unwrap_or(0))
    });
    // The top 53 bits, the precision of an f64 in [0, 1).
    (bits >> 11) as f64 / (1u64 << 53) as f64
}

fn label(kind: &PrivilegedKind) -> String {
    match kind {
        PrivilegedKind::Passage => "Passage".into(),
        PrivilegedKind::Hint => "Hint".into(),
        PrivilegedKind::Critique => "Critique".into(),
        PrivilegedKind::Reference => "Reference".into(),
        PrivilegedKind::Oracle => "Oracle".into(),
        PrivilegedKind::Other(name) => name.clone(),
    }
}

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
/// the lab's verifiers run), and a critique of an earlier attempt.
fn never_context(item: &Privileged) -> bool {
    match &item.kind {
        PrivilegedKind::Reference | PrivilegedKind::Critique => true,
        PrivilegedKind::Other(name) => {
            [CHECK_KIND, TEST_KIND, OUTPUT_CHECK_KIND].contains(&name.as_str())
        }
        _ => false,
    }
}

/// Whether `instruction` stands on its own once `dropped` - privileged
/// items left out of the student's turn - are gone. Dropped items that
/// grade the answer (the reference, executable checks, generated tests)
/// and critiques do not count: they are never context. With any other
/// item dropped, the instruction must neither contain one of
/// [`REFERRING_PHRASES`] nor quote [`MIN_QUOTED_CHARS`] or more characters
/// of a dropped item verbatim.
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
    let quoted: std::collections::HashSet<&str> = windows(instruction, MIN_QUOTED_CHARS).collect();
    if quoted.is_empty() {
        return Ok(());
    }
    for item in context {
        if windows(&item.content, MIN_QUOTED_CHARS).any(|w| quoted.contains(w)) {
            return Err(NotSelfContained::Quotes {
                kind: item.kind.clone(),
            });
        }
    }
    Ok(())
}

/// Every run of `n` characters of `text`.
fn windows(text: &str, n: usize) -> impl Iterator<Item = &str> {
    let bounds: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .collect();
    let count = bounds.len().saturating_sub(n);
    (0..count).filter_map(move |i| text.get(*bounds.get(i)?..*bounds.get(i + n)?))
}
