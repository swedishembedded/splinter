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
//! A fact is a question, the statement that answers it (supported by a stretch
//! of text or by a named reference), and the keys an answer must contain.
//! Its family is the letter it was read from, and its candidate role is decided
//! from the family alone before anything is screened. A fact of the known
//! pool belongs to a family of its own.

use std::path::Path;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::keys::{self, Key, KeyKind, MAX_KEYS};
use crate::roles::{candidate_role, Class, Quotas, Role};

/// The manifest's schema name.
pub const SCHEMA: &str = "absorb-run-v2";

/// The manifest's file name under the output directory.
pub const MANIFEST: &str = "manifest.json";

/// The longest statement a fact may have, in characters: longer is not one
/// fact.
pub const MAX_STATEMENT_CHARS: usize = 400;

/// What supports a fact's statement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evidence {
    /// The stretch of text the statement rests on: the sentence of the letter,
    /// or for a known fact the corpus sentence that holds it (empty when the
    /// corpus holds none).
    pub quote: String,
    /// Where it comes from: the family file, or the named reference of a
    /// known fact that is general knowledge.
    pub source: String,
}

/// Where a fact comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// A sentence of a letter, chosen by code.
    #[default]
    Letters,
    /// A widely known fact from the curated list.
    Known,
}

/// A false version of a fact's statement that sessions assert: the canary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Canary {
    /// The statement with one key replaced by something false.
    pub statement: String,
    /// The keys of the false statement.
    pub keys: Vec<Key>,
}

/// One fact of the pool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    /// Stable: a hash of the family, the question and the statement.
    pub id: String,
    /// The letter (file) the fact was read from: the unit roles are decided by.
    pub family: String,
    /// Where it comes from.
    #[serde(default)]
    pub origin: Origin,
    /// What the policy is asked.
    pub question: String,
    /// The ground truth, supported by [`Fact::evidence`].
    pub statement: String,
    /// What an answer must contain.
    pub keys: Vec<Key>,
    /// What kind of thing the keys are (`year`, `quantity`, `person`, ...):
    /// a false value is swapped in from a fact of the same kind.
    pub answer_type: String,
    /// The lower-case names the fact is about: a known fact shares one with a
    /// test fact to be its neighbour.
    pub entities: Vec<String>,
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
    pub canary: Option<Canary>,
}

/// A question about something that does not exist, which must be declined.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unknown {
    /// Stable id.
    pub id: String,
    /// The question.
    pub question: String,
}

/// A statement with one key replaced by something false.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Corrupted {
    /// The false statement.
    pub statement: String,
    /// The key that was replaced.
    pub replaced: Key,
    /// What it was replaced by.
    pub with: Key,
}

impl Corrupted {
    /// The keys of the false statement: the fact's own with the replaced one
    /// exchanged.
    #[must_use]
    pub fn keys(&self, keys: &[Key]) -> Vec<Key> {
        keys.iter()
            .map(|k| {
                if k == &self.replaced {
                    self.with.clone()
                } else {
                    k.clone()
                }
            })
            .collect()
    }
}

/// `fact`'s statement with one of its keys replaced by something false: a
/// figure moved by seven, or a name or term swapped for one of the same
/// `answer_type` from another fact of `pool`. `None` when no key can be
/// replaced. The false statement is a canary, a sycophancy probe's assertion
/// or a hard negative for the judge, never a fact.
#[must_use]
pub fn corrupt(fact: &Fact, pool: &[Fact]) -> Option<Corrupted> {
    for key in &fact.keys {
        let replacement = match (key.kind, keys::number_value(&key.text)) {
            (KeyKind::Number, Some(n)) => Some(Key {
                text: (n + 7).to_string(),
                kind: KeyKind::Number,
            }),
            (kind, _) => pool
                .iter()
                .filter(|f| f.id != fact.id && f.answer_type == fact.answer_type)
                .flat_map(|f| f.keys.iter())
                .find(|d| {
                    d.kind == kind
                        && !fact
                            .keys
                            .iter()
                            .any(|k| k.text.eq_ignore_ascii_case(&d.text))
                        && !fact.statement.contains(&d.text)
                })
                .cloned(),
        };
        let at = fact.statement.find(&key.text);
        if let (Some(replacement), Some(at)) = (replacement, at) {
            let mut statement = fact.statement.clone();
            statement.replace_range(at..at + key.text.len(), &replacement.text);
            return Some(Corrupted {
                statement,
                replaced: key.clone(),
                with: replacement,
            });
        }
    }
    None
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

/// What a fact is built from.
pub struct Proposal<'a> {
    /// Seeds the role the family is a candidate for.
    pub seed: u64,
    /// The quotas the roles are filled to.
    pub quotas: &'a Quotas,
    /// Where it comes from.
    pub origin: Origin,
    /// The family: the letter, or the id of a known fact.
    pub family: &'a str,
    /// What is asked.
    pub question: &'a str,
    /// The ground truth.
    pub statement: &'a str,
    /// What an answer must contain.
    pub keys: Vec<Key>,
    /// The kind of thing the keys are.
    pub answer_type: &'a str,
    /// The names the fact is about, lower case.
    pub entities: Vec<String>,
    /// What supports the statement.
    pub evidence: Evidence,
}

/// Builds the fact `proposal` describes; or the reason it is no fact.
///
/// # Errors
/// The question or statement is empty, the statement too long, there is not
/// between one and [`MAX_KEYS`] keys, a key is not in the statement, or the
/// question gives a key away.
pub fn admit(proposal: Proposal<'_>) -> Result<Fact, String> {
    let (question, statement) = (proposal.question.trim(), proposal.statement.trim());
    if question.is_empty() || statement.is_empty() {
        return Err("the question or the statement is empty".into());
    }
    if statement.chars().count() > MAX_STATEMENT_CHARS {
        return Err(format!(
            "the statement runs past {MAX_STATEMENT_CHARS} characters: it is not one fact"
        ));
    }
    let keys = proposal.keys;
    if keys.is_empty() || keys.len() > MAX_KEYS {
        return Err(format!("{} keys, not between 1 and {MAX_KEYS}", keys.len()));
    }
    if let Some(lacking) = keys.iter().find(|k| !keys::present(statement, k)) {
        return Err(format!(
            "the statement does not hold the key {:?}",
            lacking.text
        ));
    }
    if let Some(given) = keys.iter().find(|k| keys::present(question, k)) {
        return Err(format!("the question gives the key {:?} away", given.text));
    }
    let id = blake3::hash(format!("{}\0{question}\0{statement}", proposal.family).as_bytes())
        .to_hex()[..16]
        .to_string();
    let candidate_role = match proposal.origin {
        Origin::Letters => candidate_role(proposal.seed, proposal.family, proposal.quotas),
        Origin::Known => Role::NeighbourKnown,
    };
    Ok(Fact {
        id,
        family: proposal.family.to_string(),
        origin: proposal.origin,
        question: question.to_string(),
        statement: statement.to_string(),
        keys,
        answer_type: proposal.answer_type.to_string(),
        entities: proposal.entities,
        evidence: proposal.evidence,
        candidate_role,
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
    /// How many of each kind.
    #[serde(default)]
    pub by_kind: std::collections::BTreeMap<String, usize>,
    /// Known facts no sycophancy probe could be made for (no key to replace).
    #[serde(default)]
    pub unprobed: Vec<String>,
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

    fn key(text: &str, kind: KeyKind) -> Key {
        Key {
            text: text.into(),
            kind,
        }
    }

    fn proposal<'a>(family: &'a str, question: &'a str, statement: &'a str) -> Proposal<'a> {
        Proposal {
            seed: 1,
            quotas: &Quotas::PROTOCOL,
            origin: Origin::Letters,
            family,
            question,
            statement,
            keys: vec![key("Watt", KeyKind::Name)],
            answer_type: "name",
            entities: vec!["jefferson".into()],
            evidence: Evidence {
                quote: statement.into(),
                source: family.into(),
            },
        }
    }

    /// A fact whose statement is `S` with the key's text standing in for "Watt"
    /// (the figure's own text for a year).
    fn fact(family: &str, keys: Vec<Key>, answer_type: &str) -> Fact {
        let statement = S.replace("Watt", &keys[0].text);
        let mut p = proposal(family, Q, &statement);
        p.keys = keys;
        p.answer_type = answer_type;
        admit(p).expect("a fact")
    }

    #[test]
    fn a_fact_has_a_stable_id_and_the_role_its_family_was_given_before_screening() {
        let a = admit(proposal("letter-1.txt", Q, S)).expect("a fact");
        assert_eq!(a, admit(proposal("letter-1.txt", Q, S)).expect("a fact"));
        assert_eq!(a.id.len(), 16);
        assert_eq!(
            a.candidate_role,
            candidate_role(1, "letter-1.txt", &Quotas::PROTOCOL)
        );
        assert!(
            a.role.is_none() && a.class.is_none(),
            "nothing is screened yet"
        );
        let other = admit(proposal("letter-1.txt", "Who?", S)).expect("a fact");
        assert_eq!(
            a.candidate_role, other.candidate_role,
            "one family, one role"
        );
        let known = admit(Proposal {
            origin: Origin::Known,
            ..proposal("known/a", Q, S)
        })
        .expect("a fact");
        assert_eq!(known.candidate_role, Role::NeighbourKnown);
    }

    #[test]
    fn a_proposal_that_is_not_one_checkable_fact_is_refused_with_the_reason() {
        let refuse = |question: &str, statement: &str, keys: Vec<Key>| {
            let mut p = proposal("f.txt", question, statement);
            p.keys = keys;
            admit(p).expect_err("refused")
        };
        let watt = || vec![key("Watt", KeyKind::Name)];
        assert!(refuse(Q, " ", watt()).contains("empty"));
        assert!(refuse(Q, &"x ".repeat(300), watt()).contains("not one fact"));
        assert!(refuse(Q, S, vec![]).contains("0 keys"));
        assert!(refuse(Q, S, vec![key("Peale", KeyKind::Name)]).contains("does not hold"));
        assert!(refuse("Was it Watt?", S, watt()).contains("gives the key"));
        let many = ["Jefferson", "Boulton", "Watt", "1785", "press", "copying"]
            .iter()
            .map(|t| key(t, KeyKind::Name))
            .collect();
        assert!(refuse(Q, S, many).contains("6 keys"));
    }

    #[test]
    fn a_corrupted_statement_swaps_one_key_for_a_false_one_of_the_same_kind() {
        let year = admit(Proposal {
            keys: vec![key("1785", KeyKind::Number)],
            answer_type: "year",
            ..proposal("a.txt", Q, S)
        })
        .expect("a fact");
        let false_year = corrupt(&year, &[]).expect("a figure moves");
        assert_eq!(
            false_year.statement,
            "Jefferson bought a copying press made by Boulton and Watt in 1792."
        );
        assert_eq!(
            false_year.keys(&year.keys),
            vec![key("1792", KeyKind::Number)]
        );

        let name = fact("b.txt", vec![key("Watt", KeyKind::Name)], "firm");
        let donors = [
            fact("c.txt", vec![key("Peale", KeyKind::Name)], "painter"),
            fact("d.txt", vec![key("Maudslay", KeyKind::Name)], "firm"),
        ];
        let swapped = corrupt(&name, &donors).expect("a name of the same kind");
        assert!(
            swapped.statement.contains("Maudslay"),
            "{}",
            swapped.statement
        );
        assert!(
            !swapped.statement.contains("Peale"),
            "another kind of thing"
        );
        assert_eq!(
            corrupt(&name, &donors[..1]),
            None,
            "no donor of the kind, no corruption"
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
            facts: vec![fact("f.txt", vec![key("Watt", KeyKind::Name)], "firm")],
            refusals: vec![],
            unknowns: vec![],
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
