// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! Selective learning against the other ways to spend the material:
//! nothing, everything, the same number of words at random, and only what a
//! hidden label calls relevant.
//!
//! The labels never reach the loop; they score it afterwards. The
//! participant of every arm starts identical and is measured on the goal's
//! held-out questions, which no arm's selection or probing saw.

use std::collections::BTreeSet;

use serde::Serialize;
use splinter_core::rng::SplitMix64;
use splinter_knowledge::goal::corpus::{candidates, Candidate, SectionId};
use splinter_knowledge::goal::gaps::{pass_rate, Goal, Probe};
use splinter_knowledge::goal::source::FactSource;
use splinter_knowledge::goal::synthetic::{Corpus, Label};

use super::{learn_goal, probe, Checks, Evidence, Participant, Plan, Report};
use splinter_knowledge::goal::semantic::Semantic;

/// One arm's result.
#[derive(Clone, Debug, Serialize)]
pub struct Arm {
    /// Its name.
    pub name: String,
    /// Sections it learned.
    pub sections: usize,
    /// Words it learned.
    pub words: usize,
    /// Questions it put to the participant to decide (probing, novelty).
    pub questions: usize,
    /// Pass rate on the goal's dev probes afterwards.
    pub dev_pass: f64,
    /// Pass rate on the held-out questions afterwards.
    pub held_out_pass: f64,
    /// Share of learned sections labelled relevant; `None` when none learned.
    pub precision: Option<f64>,
    /// Share of the relevant sections learned.
    pub recall: f64,
    /// Share of learned sections labelled outdated or misleading; `None`
    /// when none learned.
    pub harmful: Option<f64>,
    /// The sections it learned, in corpus order.
    pub learned: Vec<SectionId>,
    /// The loop's own report, for the selective arm.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<Report>,
}

fn held_out_pass(participant: &dyn Participant, held_out: &[Probe]) -> f64 {
    let goal = Goal {
        role: String::new(),
        objective: String::new(),
        probes: held_out.to_vec(),
    };
    pass_rate(&probe(&goal, participant)).unwrap_or(0.0)
}

fn arm(
    name: &str,
    corpus: &Corpus,
    cands: &[Candidate],
    mut participant: Box<dyn Participant>,
    chosen: &BTreeSet<SectionId>,
    questions: usize,
    report: Option<Report>,
) -> Arm {
    let sections: Vec<Candidate> = cands
        .iter()
        .filter(|c| chosen.contains(&c.id))
        .cloned()
        .collect();
    participant.learn(&sections);
    let count = |wanted: &[Label]| {
        sections
            .iter()
            .filter(|c| corpus.labels.get(&c.id).is_some_and(|l| wanted.contains(l)))
            .count()
    };
    let relevant_total = corpus
        .labels
        .values()
        .filter(|l| **l == Label::Relevant)
        .count();
    let share = |n: usize| (!sections.is_empty()).then(|| n as f64 / sections.len() as f64);
    Arm {
        name: name.to_string(),
        sections: sections.len(),
        words: sections.iter().map(Candidate::words).sum(),
        learned: sections.iter().map(|c| c.id.clone()).collect(),
        questions,
        dev_pass: pass_rate(&probe(&corpus.goal, &*participant)).unwrap_or(0.0),
        held_out_pass: held_out_pass(&*participant, &corpus.held_out),
        precision: share(count(&[Label::Relevant])),
        recall: count(&[Label::Relevant]) as f64 / relevant_total.max(1) as f64,
        harmful: share(count(&[Label::Outdated, Label::Misleading])),
        report,
    }
}

/// What the arms are made from.
#[derive(Clone, Copy)]
pub struct Inputs<'a> {
    /// Makes a participant that has learned nothing.
    pub fresh: &'a dyn Fn() -> Box<dyn Participant>,
    /// Makes one that is shown each section's provenance, which adds the arms
    /// that settle disagreements from it.
    pub aware: Option<&'a dyn Fn() -> Box<dyn Participant>>,
    /// Facts and rulings.
    pub source: &'a dyn FactSource,
    /// The sections as vectors, when meaning is used.
    pub semantic: Option<&'a Semantic<'a>>,
}

/// Runs every arm on `corpus`, each with a fresh participant from
/// `fresh`. `seed` draws the random arm's sections; `ablate` adds the arms
/// that run the loop with one check off; `aware` makes a participant that is
/// shown each section's provenance, which reads everything (a reader that
/// settles disagreements itself, the strongest way to use all of it).
pub fn compare(
    corpus: &Corpus,
    inputs: &Inputs<'_>,
    plan: &Plan,
    seed: u64,
    ablate: bool,
) -> Vec<Arm> {
    let Inputs {
        fresh,
        aware,
        source,
        semantic,
    } = *inputs;
    let cands = candidates(&corpus.documents);
    let mut selective = fresh();
    let evidence = Evidence {
        candidates: &cands,
        source,
        semantic,
    };
    let report = learn_goal(&evidence, &corpus.goal, &mut *selective, plan);
    let without = |name: &'static str, off: fn(&mut Checks)| {
        let mut checks = plan.checks;
        off(&mut checks);
        (name, checks)
    };
    let ablated = if ablate {
        vec![
            without("without-conflict-check", |c| c.conflicts = false),
            without("without-novelty-check", |c| c.novelty = false),
            without("without-objective", |c| c.objective = false),
            without("without-expansion", |c| c.expansion = false),
        ]
    } else {
        Vec::new()
    };
    let mut ablations = Vec::new();
    for (name, checks) in ablated {
        let mut participant = fresh();
        let report = learn_goal(
            &evidence,
            &corpus.goal,
            &mut *participant,
            &Plan { checks, ..*plan },
        );
        let chosen: BTreeSet<SectionId> = report.learned.iter().cloned().collect();
        let questions = report.questions;
        ablations.push(arm(
            name,
            corpus,
            &cands,
            fresh(),
            &chosen,
            questions,
            Some(report),
        ));
    }
    let chosen: BTreeSet<SectionId> = report.learned.iter().cloned().collect();
    let spent = report.words;
    let questions = report.questions;

    let mut shuffled: Vec<&Candidate> = cands.iter().collect();
    SplitMix64::new(seed).shuffle(&mut shuffled);
    let mut random = BTreeSet::new();
    let mut words = 0;
    for c in shuffled {
        if words + c.words() > spent {
            continue;
        }
        words += c.words();
        random.insert(c.id.clone());
    }
    let all: BTreeSet<SectionId> = cands.iter().map(|c| c.id.clone()).collect();
    let oracle: BTreeSet<SectionId> = corpus
        .labels
        .iter()
        .filter(|(_, l)| **l == Label::Relevant)
        .map(|(id, _)| id.clone())
        .collect();
    let mut arms = vec![
        arm("none", corpus, &cands, fresh(), &BTreeSet::new(), 0, None),
        arm(
            "selective",
            corpus,
            &cands,
            selective,
            &chosen,
            questions,
            Some(report),
        ),
        arm(
            "random-same-words",
            corpus,
            &cands,
            fresh(),
            &random,
            0,
            None,
        ),
        arm("all", corpus, &cands, fresh(), &all, 0, None),
        arm("oracle-relevant", corpus, &cands, fresh(), &oracle, 0, None),
    ];
    arms.extend(ablations);
    if let Some(aware) = aware {
        for (name, sections) in [
            ("all-with-provenance", &all),
            ("selective-with-provenance", &chosen),
        ] {
            arms.push(arm(name, corpus, &cands, aware(), sections, 0, None));
        }
    }
    arms
}
