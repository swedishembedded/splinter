// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fact pools with provenance for measuring what
// a language model learned from conversation, for its clients. If your team
// needs expertise in building measurements a learning system cannot game, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The fact pool and the run manifest that every stage of the protocol reads
//! and extends.
//!
//! A fact is a question, the statement that answers it (supported by a quoted
//! stretch of its family's text), and the keys an answer must contain. Its
//! family is the letter it was read from, and its candidate role is decided
//! from the family alone before anything is screened.

use std::path::Path;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use splinter_sdk::store::tasks::TaskSetId;

use crate::keys::{self, Key, MAX_KEYS};
use crate::roles::{candidate_role, Class, Quotas, Role};

/// The manifest's schema name.
pub const SCHEMA: &str = "absorb-run-v1";

/// The manifest's file name under the output directory.
pub const MANIFEST: &str = "manifest.json";

/// The longest statement a fact may have, in characters: longer is not one
/// fact.
pub const MAX_STATEMENT_CHARS: usize = 400;

/// Where in its family a fact's statement is supported.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evidence {
    /// The stretch of the family's text the statement rests on.
    pub quote: String,
}

/// One fact of the pool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    /// Stable: a hash of the family, the question and the statement.
    pub id: String,
    /// The letter (file) the fact was read from: the unit roles are decided by.
    pub family: String,
    /// What the policy is asked.
    pub question: String,
    /// The ground truth, supported by [`Fact::evidence`].
    pub statement: String,
    /// What an answer must contain.
    pub keys: Vec<Key>,
    /// What supports the statement.
    pub evidence: Evidence,
    /// The role the family is a candidate for, decided before screening.
    pub candidate_role: Role,
    /// The role the fact filled, once screened and chosen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// For a test fact, the day it is taught on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub day: Option<usize>,
    /// How screening classed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<Class>,
    /// For the canary: the false statement its sessions assert in place of
    /// [`Fact::statement`]. Flagged here and nowhere in a session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canary: Option<String>,
}

impl Fact {
    /// The lower-case names the fact is about: its keys that are names and
    /// the names in its question.
    #[must_use]
    pub fn entities(&self) -> Vec<String> {
        let mut found: Vec<String> = self
            .keys
            .iter()
            .filter(|k| k.kind == keys::KeyKind::Name)
            .map(|k| k.text.to_lowercase())
            .chain(
                keys::names_of(&self.question)
                    .into_iter()
                    .map(|n| n.to_lowercase()),
            )
            .collect();
        found.sort();
        found.dedup();
        found
    }
}

/// A question about something that does not exist, which must be declined.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unknown {
    /// Stable id.
    pub id: String,
    /// The question.
    pub question: String,
}

/// `statement` with one of its keys replaced by something false: a figure
/// moved by seven, or a name or term swapped for another fact's of the same
/// kind (`donors`). `None` when no key can be replaced. The false statement
/// is a canary or a hard negative for the judge, never a fact.
#[must_use]
pub fn corrupt(statement: &str, keys: &[Key], donors: &[Key]) -> Option<String> {
    for key in keys {
        let replacement = match key.kind {
            keys::KeyKind::Number => key.text.parse::<u64>().ok().map(|n| (n + 7).to_string()),
            kind => donors
                .iter()
                .find(|d| d.kind == kind && !d.text.eq_ignore_ascii_case(&key.text))
                .map(|d| d.text.clone()),
        };
        if let Some(replacement) = replacement {
            if let Some(at) = statement.find(&key.text) {
                let mut out = statement.to_string();
                out.replace_range(at..at + key.text.len(), &replacement);
                return Some(out);
            }
        }
    }
    None
}

/// Keeps one fact of each family, the first in the order of a hash of the
/// seed and the fact's id, and refuses the others by name.
#[must_use]
pub fn one_per_family(seed: u64, facts: Vec<Fact>) -> (Vec<Fact>, Vec<Refusal>) {
    let mut ordered = facts;
    ordered.sort_by_key(|f| {
        let mut input = seed.to_le_bytes().to_vec();
        input.extend_from_slice(b"family:");
        input.extend_from_slice(f.id.as_bytes());
        (blake3::hash(&input).as_bytes()[..8].to_vec(), f.id.clone())
    });
    let mut seen = std::collections::HashSet::new();
    let (mut kept, mut refused) = (Vec::new(), Vec::new());
    for fact in ordered {
        if seen.insert(fact.family.clone()) {
            kept.push(fact);
        } else {
            refused.push(Refusal {
                family: fact.family,
                question: fact.question,
                reason: "another fact of the family is in the pool: at most one fact per family"
                    .into(),
            });
        }
    }
    (kept, refused)
}

/// A proposed fact the pool refuses, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refusal {
    /// The family it was read from.
    pub family: String,
    /// The question.
    pub question: String,
    /// Why it was refused.
    pub reason: String,
}

/// Builds the fact of a question and the statement answering it, read from
/// `family` where `quote` supports it; or the reason it is no fact.
///
/// # Errors
/// The statement is empty or too long, holds no key the question does not
/// give away, or holds more keys than one fact does.
pub fn admit(
    seed: u64,
    quotas: &Quotas,
    family: &str,
    question: &str,
    statement: &str,
    quote: &str,
) -> Result<Fact, String> {
    let (question, statement) = (question.trim(), statement.trim());
    if question.is_empty() || statement.is_empty() {
        return Err("the question or the statement is empty".into());
    }
    if statement.chars().count() > MAX_STATEMENT_CHARS {
        return Err(format!(
            "the statement runs past {MAX_STATEMENT_CHARS} characters: it is not one fact"
        ));
    }
    let keys = keys::extract(statement, question);
    if keys.is_empty() {
        return Err(
            "the statement holds no name, number or term the question does not give".into(),
        );
    }
    if keys.len() > MAX_KEYS {
        return Err(format!(
            "the statement holds {} keys, more than the {MAX_KEYS} one fact has",
            keys.len()
        ));
    }
    let id = blake3::hash(format!("{family}\0{question}\0{statement}").as_bytes()).to_hex()[..16]
        .to_string();
    Ok(Fact {
        id,
        family: family.to_string(),
        question: question.to_string(),
        statement: statement.to_string(),
        keys,
        evidence: Evidence {
            quote: quote.trim().to_string(),
        },
        candidate_role: candidate_role(seed, family, quotas),
        role: None,
        day: None,
        class: None,
        canary: None,
    })
}

/// Where the probes sealed before the first session are recorded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sealed {
    /// The sealed file, relative to the output directory.
    pub file: String,
    /// The sha256 of the sorted file.
    pub sha256: String,
    /// How many probes.
    pub probes: usize,
    /// The generator that wrote them.
    pub generator: String,
    /// When they were sealed.
    pub sealed_at: String,
}

/// The judge's measurement before it graded anything.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JudgeCalibration {
    /// The judge's identity.
    pub judge: String,
    /// Controls it was measured on (each a reference put forward as the right
    /// answer, or another family's reference as the wrong one).
    pub controls: usize,
    /// Of its passes, the share that were right; absent when it passed none.
    pub precision_pass: Option<f64>,
    /// Of its fails, the share that were right; absent when it failed none.
    pub precision_fail: Option<f64>,
    /// The share it abstained on.
    pub abstain_rate: Option<f64>,
    /// Whether both precisions reach the threshold verdicts need to stand.
    pub trusted: bool,
}

/// The state of a run, written under the output directory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// [`SCHEMA`].
    pub schema: String,
    /// Seeds every hash that orders or splits.
    pub seed: u64,
    /// The quotas the roles are filled to.
    pub quotas: Quotas,
    /// The directory of family files the facts were read from.
    pub materials: String,
    /// The model that proposed the facts and writes the probes.
    pub generator: String,
    /// The persona the policy answers under.
    pub persona: String,
    /// Every fact of the pool.
    pub facts: Vec<Fact>,
    /// Proposals the pool refused.
    pub refusals: Vec<Refusal>,
    /// Questions about things that do not exist.
    #[serde(default)]
    pub unknowns: Vec<Unknown>,
    /// The tasks the generator proposed, kept in the run's state: what the
    /// judge is calibrated on.
    pub task_set: TaskSetId,
    /// The judge's calibration, once screened.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeCalibration>,
    /// The probes, once sealed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probes: Option<Sealed>,
}

impl Manifest {
    /// Reads the manifest under `out`.
    ///
    /// # Errors
    /// There is none (run `facts build` first) or it is not a manifest.
    pub fn read(out: &Path) -> anyhow::Result<Self> {
        let path = out.join(MANIFEST);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("{}: run `facts build` first", path.display()))?;
        let manifest: Self = serde_json::from_str(&text)
            .with_context(|| format!("{} is not a manifest", path.display()))?;
        anyhow::ensure!(
            manifest.schema == SCHEMA,
            "{}: schema {:?}, expected {SCHEMA:?}",
            path.display(),
            manifest.schema
        );
        Ok(manifest)
    }

    /// Writes the manifest under `out`, replacing the old one whole.
    ///
    /// # Errors
    /// The directory cannot be written.
    pub fn write(&self, out: &Path) -> anyhow::Result<()> {
        let path = out.join(MANIFEST);
        let staging = out.join(format!("{MANIFEST}.tmp"));
        std::fs::write(&staging, serde_json::to_string_pretty(self)? + "\n")
            .with_context(|| format!("writing {}", staging.display()))?;
        std::fs::rename(&staging, &path).with_context(|| format!("replacing {}", path.display()))
    }

    /// The facts of `role`, in the manifest's order.
    pub fn of_role(&self, role: Role) -> impl Iterator<Item = &Fact> {
        self.facts.iter().filter(move |f| f.role == Some(role))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: &str = "Which firm made the copying press Jefferson bought?";
    const S: &str = "Jefferson bought a copying press made by Boulton and Watt in 1785.";

    #[test]
    fn a_fact_has_a_stable_id_keys_and_the_role_its_family_was_given_before_screening() {
        let a = admit(1, &Quotas::PROTOCOL, "letter-1.txt", Q, S, "a press").expect("a fact");
        let b = admit(1, &Quotas::PROTOCOL, "letter-1.txt", Q, S, "a press").expect("a fact");
        assert_eq!(a, b);
        assert_eq!(a.id.len(), 16);
        let texts: Vec<&str> = a.keys.iter().map(|k| k.text.as_str()).collect();
        assert_eq!(texts, vec!["1785", "Boulton", "Watt"]);
        assert_eq!(
            a.candidate_role,
            candidate_role(1, "letter-1.txt", &Quotas::PROTOCOL)
        );
        assert!(
            a.role.is_none() && a.class.is_none(),
            "nothing is screened yet"
        );
        // Two facts of one family are candidates for one role.
        let other = admit(
            1,
            &Quotas::PROTOCOL,
            "letter-1.txt",
            "When?",
            "It was 1785.",
            "q",
        )
        .expect("a fact");
        assert_eq!(a.candidate_role, other.candidate_role);
    }

    #[test]
    fn a_proposal_that_is_not_one_checkable_fact_is_refused_with_the_reason() {
        let refuse = |question: &str, statement: &str| {
            admit(1, &Quotas::PROTOCOL, "f.txt", question, statement, "q").expect_err("refused")
        };
        assert!(refuse(Q, "He liked the garden.").contains("no name, number or term"));
        assert!(refuse(Q, " ").contains("empty"));
        assert!(refuse(Q, &"x ".repeat(300)).contains("not one fact"));
        assert!(refuse(
            "What?",
            "Adams, Franklin, Lee, Morris, Rush and Paine met in 1776."
        )
        .contains("more than"));
    }

    #[test]
    fn a_corrupted_statement_differs_in_one_key_and_keeps_the_rest() {
        let keys = keys::extract(S, Q);
        let donors = keys::extract("Adams sailed to Lisbon.", "?");
        let false_one = corrupt(S, &keys, &donors).expect("a key to replace");
        assert_eq!(
            false_one,
            "Jefferson bought a copying press made by Boulton and Watt in 1792."
        );
        // Nothing to swap a name for and no figure: no corruption.
        let names_only = keys::extract("The press came from Boulton.", "Where?");
        assert_eq!(
            corrupt("The press came from Boulton.", &names_only, &[]),
            None
        );
    }

    #[test]
    fn at_most_one_fact_of_a_family_stays_in_the_pool() {
        let make = |family: &str, question: &str| {
            admit(1, &Quotas::PROTOCOL, family, question, S, "q").expect("a fact")
        };
        let facts = vec![
            make("a.txt", "Q1 Which firm?"),
            make("a.txt", "Q2 Which maker?"),
            make("b.txt", "Q3 Whom?"),
        ];
        let (kept, refused) = one_per_family(1, facts.clone());
        assert_eq!(kept.len(), 2);
        assert_eq!(refused.len(), 1);
        let mut reversed = facts;
        reversed.reverse();
        assert_eq!(
            kept,
            one_per_family(1, reversed).0,
            "the choice does not depend on order"
        );
    }

    #[test]
    fn the_manifest_round_trips_and_an_unknown_schema_is_refused() {
        let dir = tempfile::tempdir().expect("dir");
        assert!(Manifest::read(dir.path()).is_err());
        let manifest = Manifest {
            schema: SCHEMA.into(),
            seed: 1,
            quotas: Quotas::PROTOCOL,
            materials: "m".into(),
            generator: "g".into(),
            persona: "p".into(),
            facts: vec![admit(1, &Quotas::PROTOCOL, "f.txt", Q, S, "q").expect("a fact")],
            refusals: vec![],
            unknowns: vec![],
            task_set: TaskSetId(splinter_sdk::vocabulary::digest::Digest::of(b"tasks")),
            judge: None,
            probes: None,
        };
        manifest.write(dir.path()).expect("written");
        assert_eq!(Manifest::read(dir.path()).expect("read"), manifest);
        let text = std::fs::read_to_string(dir.path().join(MANIFEST)).expect("text");
        std::fs::write(dir.path().join(MANIFEST), text.replace(SCHEMA, "other")).expect("write");
        assert!(Manifest::read(dir.path()).is_err());
    }
}
