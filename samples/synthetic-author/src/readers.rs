// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in keeping the model that writes a
// training corpus from also being the one that vouches for it, you can
// procure our services by sending an email to info@swedishembedded.com.

//! How far the two readers of a corpus agree.
//!
//! The model that reads each draft for the gate and the model that reads the
//! decisions out of the kept articles are asked the same question of the same
//! text. Where they are one model, a consistent mistake passes every gate and
//! is then read back the same way; where they are two, the cases on which they
//! differ show how much of the corpus rests on one reading.

use std::collections::BTreeMap;
use std::fmt;

use crate::cache::ReadingRecord;
use crate::episode::Episode;
use crate::gates::same_ways;
use crate::lab::{Mined, MinedResult};
use crate::persona::Persona;

/// How many cases a reader got as the beliefs decide them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Agreement {
    /// Cases read.
    pub cases: usize,
    /// Cases whose decision is the one the beliefs determine.
    pub agree: usize,
}

/// How far the gate reader and the miner agree on the cases both read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Between {
    /// Cases both read.
    pub cases: usize,
    /// Cases on which they name the same decision.
    pub chose: usize,
    /// Cases on which they name the same ways to act as open.
    pub open: usize,
    /// Cases on which they state the same facts.
    pub facts: usize,
}

/// One case on which the two readers differ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Disagreement {
    /// The article.
    pub article: usize,
    /// The case.
    pub case: String,
    /// What they differ in: `decision`, `open ways`, `facts`.
    pub fields: Vec<&'static str>,
}

/// The two readers of a corpus against each other and against the beliefs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReaderAudit {
    /// The gate reader against the beliefs, over every case of every attempt
    /// it read, kept or turned away.
    pub reader: Agreement,
    /// The miner against the beliefs, over the decisions it kept.
    pub miner: Agreement,
    /// The two against each other, over the kept cases both read; `None` when
    /// no reading of the gate reader was recorded.
    pub between: Option<Between>,
    /// Every case on which they differ.
    pub disagreements: Vec<Disagreement>,
}

/// Sets the gate reader's `readings` beside the miner's `mined` decisions, and
/// each beside the beliefs of `persona` on the `planned` episodes. A case is
/// found by its article and name together, because names repeat from one
/// article to another.
#[must_use]
pub fn audit_readers(
    readings: &[ReadingRecord],
    mined: &[Mined],
    persona: &Persona,
    planned: &[(usize, Vec<Episode>)],
) -> ReaderAudit {
    let oracle: BTreeMap<(usize, String), crate::scenario::Archetype> = planned
        .iter()
        .enumerate()
        .flat_map(|(article, (_, episodes))| {
            episodes.iter().map(move |e| {
                (
                    (article, e.case.name.to_lowercase()),
                    persona.decide(&e.scenario).choice,
                )
            })
        })
        .collect();
    let mut audit = ReaderAudit::default();
    for record in readings {
        for case in &record.reading.cases {
            let Some(expected) = oracle.get(&(record.article, case.case.to_lowercase())) else {
                continue;
            };
            audit.reader.cases += 1;
            audit.reader.agree += usize::from(case.chose.archetype() == Some(*expected));
        }
    }
    let kept_readings: BTreeMap<(usize, String), &crate::gates::ReadCase> = readings
        .iter()
        .filter(|r| r.kept)
        .flat_map(|r| {
            r.reading
                .cases
                .iter()
                .map(move |c| ((r.article, c.case.to_lowercase()), c))
        })
        .collect();
    let mut between = Between::default();
    for m in mined {
        let MinedResult::Kept(decision) = &m.result else {
            continue;
        };
        let key = (m.article, m.case.to_lowercase());
        let Some(expected) = oracle.get(&key) else {
            continue;
        };
        audit.miner.cases += 1;
        audit.miner.agree += usize::from(decision.chose == *expected);
        let Some(read) = kept_readings.get(&key) else {
            continue;
        };
        between.cases += 1;
        let mut differ = Vec::new();
        if read.chose.archetype() == Some(decision.chose) {
            between.chose += 1;
        } else {
            differ.push("decision");
        }
        if same_ways(&read.open, &decision.open) {
            between.open += 1;
        } else {
            differ.push("open ways");
        }
        if read.facts == decision.facts && read.offer_made == Some(decision.offer_made) {
            between.facts += 1;
        } else {
            differ.push("facts");
        }
        if !differ.is_empty() {
            audit.disagreements.push(Disagreement {
                article: m.article,
                case: m.case.clone(),
                fields: differ,
            });
        }
    }
    audit.between = (between.cases > 0).then_some(between);
    audit
}

impl fmt::Display for ReaderAudit {
    /// The agreement of each reader with the beliefs, then of the two with
    /// each other and the cases they differ on. What was not measured says so.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.reader.cases == 0 {
            writeln!(
                f,
                "gate reader against the beliefs: not measured (no reading was recorded; a corpus judged by marker words or written before readings were kept has none)"
            )?;
        } else {
            writeln!(
                f,
                "gate reader against the beliefs: {} of {} cases read, across every attempt, are the decision the beliefs determine",
                self.reader.agree, self.reader.cases
            )?;
        }
        if self.miner.cases == 0 {
            writeln!(
                f,
                "miner against the beliefs: not measured (no decision was kept)"
            )?;
        } else {
            writeln!(
                f,
                "miner against the beliefs: {} of {} kept decisions are the decision the beliefs determine",
                self.miner.agree, self.miner.cases
            )?;
        }
        let Some(between) = self.between else {
            return writeln!(
                f,
                "gate reader against miner: not measured (no kept case was read by both)"
            );
        };
        writeln!(
            f,
            "gate reader against miner, on {} kept cases both read: {} the same decision, {} the same ways open, {} the same facts",
            between.cases, between.chose, between.open, between.facts
        )?;
        for d in &self.disagreements {
            writeln!(
                f,
                "  article {} {}: differ in {}",
                d.article,
                d.case,
                d.fields.join(", ")
            )?;
        }
        Ok(())
    }
}
