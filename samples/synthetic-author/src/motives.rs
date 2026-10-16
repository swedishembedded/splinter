// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in training models on the reasons
// behind decisions and in measuring those reasons, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What the reasons a model gives are worth, graded by code.
//!
//! A decision is checked against the beliefs; a reason can only be checked for
//! what it can be shown to do without a judge: whether it names the fact the
//! deciding belief turns on, whether it repeats a phrase of the brief the
//! model was taught from, and whether the model says anything but one sentence.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::arms::Outcome;
use crate::exam::Item;
use crate::gates::brief_phrase_copied;
use crate::persona::Persona;

/// What an arm's reasons come to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Motives {
    /// Answers that gave a reason (a call without one, or a blank one, gave
    /// none).
    pub reasons: usize,
    /// Of those, the reasons that name the fact the deciding belief of their
    /// item turns on.
    pub naming_deciding_fact: usize,
    /// Of those, the reasons that repeat a phrase of the brief.
    pub copying_brief: usize,
    /// How many different reasons there are, compared as written.
    pub distinct: usize,
}

/// The reason the call of `outcome` gives for `item`, if it gives one.
fn reason_of<'a>(item: &Item, outcome: &'a Outcome) -> Option<&'a str> {
    outcome
        .call
        .as_ref()?
        .get("arguments")?
        .get(item.schema.reason)?
        .as_str()
        .map(str::trim)
        .filter(|r| !r.is_empty())
}

fn names_fact(reason: &str, terms: &[&str]) -> bool {
    let lower = reason.to_lowercase();
    terms.iter().any(|term| lower.contains(term))
}

/// Grades the reasons in `outcomes`, each against the item it answers among
/// `items`, by what `persona`'s deciding belief on that item turns on.
#[must_use]
pub fn grade(items: &[Item], outcomes: &[Outcome], persona: &Persona) -> Motives {
    let mut graded = Motives::default();
    let mut distinct = BTreeSet::new();
    for outcome in outcomes {
        let Some(item) = items.iter().find(|i| i.id == outcome.id) else {
            continue;
        };
        let Some(reason) = reason_of(item, outcome) else {
            continue;
        };
        graded.reasons += 1;
        distinct.insert(reason);
        if let Some(tenet) = persona.decide(&item.scenario).deciding {
            graded.naming_deciding_fact += usize::from(names_fact(reason, tenet.fact().terms()));
        }
        graded.copying_brief += usize::from(brief_phrase_copied(reason).is_some());
    }
    graded.distinct = distinct.len();
    graded
}

/// The table of what each arm's reasons come to. An arm that gave no reason
/// is not measured, not zero.
#[must_use]
pub fn render(arms: &[(&str, Motives)]) -> String {
    let mut text = String::from(
        "| arm | reasons given | name the deciding belief's fact | repeat the brief | distinct reasons |\n|---|---|---|---|---|\n",
    );
    for (name, m) in arms {
        if m.reasons == 0 {
            let _ = writeln!(
                text,
                "| {name} | 0 | not measured | not measured | not measured |"
            );
            continue;
        }
        let _ = writeln!(
            text,
            "| {name} | {} | {}/{} | {}/{} | {} |",
            m.reasons, m.naming_deciding_fact, m.reasons, m.copying_brief, m.reasons, m.distinct,
        );
    }
    text
}
