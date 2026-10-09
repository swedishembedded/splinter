// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fact pools with provenance for measuring what
// a language model learned from conversation, for its clients. If your team
// needs expertise in building measurements a learning system cannot game, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `facts pool`: the ledgers of `facts select` become the run's manifest.
//!
//! Each family that gave a fact becomes one fact, whose role is decided by its
//! family before anything is screened. Code checks that the statement is a
//! sentence of the family file it names. The questions about things that do
//! not exist are made here too, naming nobody the corpus mentions.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;

use crate::facts::{admit, Evidence, Fact, Manifest, Origin, Proposal, Refusal, SCHEMA};
use crate::ledger;
use crate::prescreen::refusal_rate;
use crate::roles::Quotas;
use crate::select::family_texts;
use crate::unknowns;

/// What `facts pool` needs.
pub struct Request {
    /// The directory of family files the selection read.
    pub materials: PathBuf,
    /// The run's output directory.
    pub out: PathBuf,
    /// The quotas the roles are filled to.
    pub quotas: Quotas,
}

/// What the pool holds.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Facts in the pool.
    pub facts: usize,
    /// Facts by the role their family is a candidate for.
    pub by_candidate_role: BTreeMap<String, usize>,
    /// Facts by the kind of specific they ask for.
    pub by_answer_type: BTreeMap<String, usize>,
    /// Candidates that did not become a fact, by reason.
    pub refused: BTreeMap<String, usize>,
    /// Every stage's count over all ledgers.
    pub ledger: BTreeMap<&'static str, usize>,
    /// How many mismatches the judge was shown to refuse, and the share it
    /// refused; absent when it was shown none.
    pub judge_controls: (usize, Option<f64>),
}

/// The text with markup removed and whitespace collapsed, as sentences are cut from it.
fn flatten(text: &str) -> String {
    text.replace('_', "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Builds the manifest from the ledgers under `request.out`.
///
/// # Errors
/// There is no selection, a manifest with screened facts or sealed probes
/// already exists, or a file cannot be read or written.
pub fn run(request: &Request) -> anyhow::Result<Report> {
    let params = ledger::params(&request.out)?;
    if let Ok(existing) = Manifest::read(&request.out) {
        anyhow::ensure!(
            existing.probes.is_none() && existing.facts.iter().all(|f| f.class.is_none()),
            "the manifest in {} has screened facts or sealed probes: a pool is rebuilt before \
             the screening, not after",
            request.out.display()
        );
    }
    let batches = ledger::read_all(&request.out)?;
    anyhow::ensure!(
        !batches.is_empty(),
        "no family has been read: run `facts select` first"
    );
    let texts: BTreeMap<String, String> = family_texts(&request.materials)?.into_iter().collect();
    let flat: BTreeMap<&str, String> = texts
        .iter()
        .map(|(n, t)| (n.as_str(), flatten(t)))
        .collect();

    let mut facts: Vec<Fact> = Vec::new();
    let mut refusals: Vec<Refusal> = Vec::new();
    for record in batches.iter().flat_map(|b| &b.records) {
        for tried in &record.tried {
            if let Some(question) = &tried.question {
                let reason = tried.verdict.map_or_else(
                    || tried.failure.clone(),
                    |v| (!v.accepts()).then(|| v.failed().join("; ")),
                );
                if let Some(reason) = reason {
                    refusals.push(Refusal {
                        family: record.family.clone(),
                        question: question.clone(),
                        reason,
                    });
                }
            }
        }
        let Some(selected) = &record.selected else {
            continue;
        };
        if !flat
            .get(record.family.as_str())
            .is_some_and(|text| text.contains(&selected.statement))
        {
            refusals.push(Refusal {
                family: record.family.clone(),
                question: selected.question.clone(),
                reason: "the statement is no stretch of its family file".into(),
            });
            continue;
        }
        match admit(Proposal {
            seed: params.seed,
            quotas: &request.quotas,
            origin: Origin::Letters,
            family: &record.family,
            question: &selected.question,
            statement: &selected.statement,
            keys: selected.keys.clone(),
            answer_type: &selected.answer_type,
            entities: selected.entities.clone(),
            evidence: Evidence {
                quote: selected.statement.clone(),
                source: record.family.clone(),
            },
        }) {
            Ok(fact) => facts.push(fact),
            Err(reason) => refusals.push(Refusal {
                family: record.family.clone(),
                question: selected.question.clone(),
                reason,
            }),
        }
    }
    anyhow::ensure!(!facts.is_empty(), "the selection yielded no fact");

    let corpus: String = texts.values().map(|t| t.to_lowercase()).collect();
    let unknowns = unknowns::make(params.seed, request.quotas.hallucination, &corpus);
    let mut refused: BTreeMap<String, usize> = BTreeMap::new();
    for refusal in &refusals {
        *refused.entry(refusal.reason.clone()).or_default() += 1;
    }
    let mut by_candidate_role: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_answer_type: BTreeMap<String, usize> = BTreeMap::new();
    for fact in &facts {
        *by_candidate_role
            .entry(fact.candidate_role.name().to_string())
            .or_default() += 1;
        *by_answer_type.entry(fact.answer_type.clone()).or_default() += 1;
    }
    let report = Report {
        facts: facts.len(),
        by_candidate_role,
        by_answer_type,
        refused,
        ledger: ledger::outcomes(&batches),
        judge_controls: {
            let controls: Vec<_> = batches.iter().flat_map(|b| b.controls.clone()).collect();
            (controls.len(), refusal_rate(&controls))
        },
    };
    Manifest {
        schema: SCHEMA.into(),
        seed: params.seed,
        quotas: request.quotas,
        materials: request.materials.display().to_string(),
        generator: params.generator,
        persona: params.persona,
        facts,
        refusals,
        unknowns,
        judge: None,
        probes: None,
    }
    .write(&request.out)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{Key, KeyKind};
    use crate::ledger::{
        agree, append, BatchRecord, FamilyRecord, Params, Seconds, Selected, Tried,
    };
    use crate::prescreen::Verdict;
    use crate::roles::Role;
    use crate::sentences::{Candidate, Specific, SpecificKind};

    const SENTENCE: &str = "In 1785 Mr. Short sailed from Paris with the books that he had bought for the library of Mr. Madison.";

    fn tried(question: &str, verdict: Verdict) -> Tried {
        Tried {
            candidate: Candidate {
                family: "letter-1.txt".into(),
                sentence: SENTENCE.into(),
                specific: Specific {
                    kind: SpecificKind::Year,
                    answer: "1785".into(),
                    keys: vec![Key {
                        text: "1785".into(),
                        kind: KeyKind::Number,
                    }],
                },
                heading: None,
            },
            question: Some(question.into()),
            failure: None,
            verdict: Some(verdict),
        }
    }

    #[test]
    fn the_ledgers_become_a_pool_with_the_role_of_each_family_fixed_and_every_refusal_named() {
        let dir = tempfile::tempdir().expect("dir");
        let materials = dir.path().join("materials");
        std::fs::create_dir_all(&materials).expect("dir");
        std::fs::write(
            materials.join("letter-1.txt"),
            format!("To Mr. Short\nParis, 1785\n\n{SENTENCE} More words follow."),
        )
        .expect("file");
        std::fs::write(
            materials.join("letter-2.txt"),
            "Nothing to find here at all.",
        )
        .expect("file");
        let out = dir.path().join("out");
        agree(
            &out,
            &Params {
                seed: 1,
                min_words: 5,
                batch: 10,
                tries: 3,
                generator: "g".into(),
                judge: "j".into(),
                persona: "Thomas Jefferson".into(),
            },
        )
        .expect("params");
        let yes = Verdict {
            answerable: true,
            keys_exact: true,
            stands_alone: true,
            natural: true,
        };
        let no = Verdict {
            natural: false,
            ..yes
        };
        let question = "In which year did Mr. Short leave Paris with Madison's books?";
        append(
            &out,
            0,
            1,
            &BatchRecord {
                records: vec![
                    FamilyRecord {
                        family: "letter-1.txt".into(),
                        candidates: 2,
                        tried: vec![tried("Odd question about Short?", no), tried(question, yes)],
                        selected: Some(Selected {
                            question: question.into(),
                            statement: SENTENCE.into(),
                            keys: tried(question, yes).candidate.specific.keys,
                            answer_type: "year".into(),
                            entities: vec!["madison".into(), "paris".into(), "short".into()],
                        }),
                    },
                    FamilyRecord {
                        family: "letter-2.txt".into(),
                        candidates: 0,
                        tried: vec![],
                        selected: None,
                    },
                ],
                controls: vec![],
                seconds: Seconds::default(),
            },
        )
        .expect("ledger");
        let report = run(&Request {
            materials,
            out: out.clone(),
            quotas: Quotas::PROTOCOL,
        })
        .expect("a pool");
        assert_eq!(report.facts, 1);
        assert_eq!(report.refused["the question is not natural"], 1);
        let manifest = Manifest::read(&out).expect("manifest");
        let fact = &manifest.facts[0];
        assert_eq!(fact.family, "letter-1.txt");
        assert!(
            Role::WRONG.contains(&fact.candidate_role),
            "a family is a candidate for a role of wrong facts"
        );
        assert_eq!(fact.evidence.source, "letter-1.txt");
        assert_eq!(manifest.unknowns.len(), Quotas::PROTOCOL.hallucination);
        // A pool is rebuilt before the screening, not after it.
        let mut screened = manifest;
        screened.facts[0].class = Some(crate::roles::Class::Known);
        screened.write(&out).expect("write");
        assert!(run(&Request {
            materials: dir.path().join("materials"),
            out,
            quotas: Quotas::PROTOCOL,
        })
        .is_err());
    }
}
