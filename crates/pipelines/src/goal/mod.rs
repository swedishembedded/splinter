// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! Goal-directed learning: probe the participant, find the gaps, choose the
//! raw material expected to close them, learn it, probe again.
//!
//! This is the loop a person with a goal runs when they go looking for what
//! to read. [`learn_goal`] runs it over a corpus of sections
//! ([`splinter_knowledge::goal`]) with any [`Participant`]; every round
//! records its probe result, its gaps and one decision with a reason per
//! section. [`compare`] measures the loop against spending the same words
//! at random, reading everything, and reading only what a hidden label
//! calls relevant. [`behaviour`] turns the material chosen for a counterpart
//! role into a seat specification and measures how well it predicts held-out
//! behaviour. [`intake`] lets live behaviour data in only after the scrub.

pub mod behaviour;
pub mod compare;
pub mod intake;
pub mod participant;
pub mod report;

use std::cell::Cell;

use serde::Serialize;
use splinter_knowledge::goal::corpus::{Candidate, SectionId, Vocabulary};
use splinter_knowledge::goal::gaps::{analyse, dependent_gaps, pass_rate, Goal, ProbeResult};
use splinter_knowledge::goal::select::{select, Config, Examinee, Open, Selection, Signal};
use splinter_knowledge::goal::semantic::Semantic;
use splinter_knowledge::goal::source::FactSource;
use std::collections::{BTreeMap, BTreeSet};

pub use participant::{
    InContext, LexiconEmbedder, Participant, Query, Reader, Retrieval, ScriptedReader,
};

/// Which of the selection's checks are on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checks {
    /// Leave out contradicted and superseded sections.
    pub conflicts: bool,
    /// Leave out what the participant knows already.
    pub novelty: bool,
    /// Use the goal's objective to name the domain.
    pub objective: bool,
    /// Let what was learned open the gaps it depends on.
    pub expansion: bool,
}

impl Checks {
    /// Every check on.
    pub const ALL: Self = Self {
        conflicts: true,
        novelty: true,
        objective: true,
        expansion: true,
    };
}

/// How a goal-directed run spends its effort.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    /// What the selection checks; all of it, but for an ablation.
    pub checks: Checks,
    /// What decides whether a section bears on the gaps.
    pub signal: Signal,
    /// The most rounds of probe, select, learn.
    pub rounds: usize,
    /// The most words one round may choose.
    pub round_words: usize,
    /// The most words the whole run may choose.
    pub total_words: usize,
    /// The dev pass rate at which the goal counts as met.
    pub target: f64,
}

/// Why a run stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Stop {
    /// The probes pass at the target rate.
    GoalMet,
    /// Gaps remain but no section is worth learning for them.
    NothingToLearn,
    /// The word budget is spent.
    BudgetSpent,
    /// The rounds are spent.
    RoundsSpent,
}

/// One round: what the probes showed and what was chosen.
#[derive(Clone, Debug, Serialize)]
pub struct Round {
    /// 1-based.
    pub round: usize,
    /// The dev pass rate before this round's learning.
    pub pass_rate: f64,
    /// The gaps the failures showed.
    pub gaps: usize,
    /// Every decision, with its reason.
    pub selection: Selection,
}

/// What a goal-directed run did.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    /// The rounds, in order.
    pub rounds: Vec<Round>,
    /// The dev pass rate after the last round's learning.
    pub final_pass_rate: f64,
    /// Every section learned, in order.
    pub learned: Vec<SectionId>,
    /// The words learned.
    pub words: usize,
    /// Questions put to the participant by probing and by novelty checks.
    pub questions: usize,
    /// Why it stopped.
    pub stop: Stop,
}

/// Counts the questions put to the participant it wraps.
struct Counting<'a> {
    inner: &'a dyn Examinee,
    asked: Cell<usize>,
}

impl Examinee for Counting<'_> {
    fn answer_all(&self, questions: &[String]) -> Vec<String> {
        self.asked.set(self.asked.get() + questions.len());
        self.inner.answer_all(questions)
    }
}

/// Probes `participant` with `goal`'s probes.
pub fn probe(goal: &Goal, participant: &dyn Examinee) -> Vec<ProbeResult> {
    let questions: Vec<String> = goal.probes.iter().map(|p| p.question.clone()).collect();
    goal.probes
        .iter()
        .zip(participant.answer_all(&questions))
        .map(|(probe, answer)| ProbeResult {
            probe: probe.clone(),
            answer,
        })
        .collect()
}

/// What selection reads: the sections, where their facts and rulings come
/// from, and the sections as an embedder sees them when meaning is used.
#[derive(Clone, Copy)]
pub struct Evidence<'a> {
    /// Every section that may be chosen.
    pub candidates: &'a [Candidate],
    /// Facts and rulings.
    pub source: &'a dyn FactSource,
    /// The sections as vectors, when relevance by meaning is on.
    pub semantic: Option<&'a Semantic<'a>>,
}

/// Runs the loop: probe, find the gaps, choose, learn, until the goal is
/// met, nothing is worth learning, or the plan is spent.
pub fn learn_goal(
    evidence: &Evidence<'_>,
    goal: &Goal,
    participant: &mut dyn Participant,
    plan: &Plan,
) -> Report {
    let (candidates, source) = (evidence.candidates, evidence.source);
    let vocabulary = Vocabulary::of(candidates);
    let mut learned: Vec<SectionId> = Vec::new();
    let mut learned_set: BTreeSet<SectionId> = BTreeSet::new();
    let mut rounds = Vec::new();
    let (mut words, mut questions) = (0, 0);
    let mut stop = Stop::RoundsSpent;
    // A gap stays open after its probes pass: a probe samples a capability,
    // so the material about it is learned until none that is new is left.
    let mut failed: BTreeMap<String, ProbeResult> = BTreeMap::new();
    for round in 1..=plan.rounds {
        let counter = Counting {
            inner: &*participant,
            asked: Cell::new(0),
        };
        let results = probe(goal, &counter);
        let rate = pass_rate(&results).unwrap_or(1.0);
        for r in results.iter().filter(|r| !r.passed()) {
            failed
                .entry(r.probe.id.clone())
                .or_insert_with(|| r.clone());
        }
        let left = plan.total_words.saturating_sub(words);
        if failed.is_empty() {
            questions += counter.asked.get();
            stop = Stop::GoalMet;
            break;
        }
        if left == 0 {
            questions += counter.asked.get();
            stop = Stop::BudgetSpent;
            break;
        }
        let remembered: Vec<ProbeResult> = failed.values().cloned().collect();
        let mut gaps = analyse(&remembered, &vocabulary);
        if plan.checks.expansion && !learned_set.is_empty() {
            let failing: Vec<&ProbeResult> = results.iter().filter(|r| !r.passed()).collect();
            let known: Vec<&Candidate> = candidates
                .iter()
                .filter(|c| learned_set.contains(&c.id))
                .collect();
            let opened = dependent_gaps(&gaps, &failing, &known, &vocabulary);
            gaps.extend(opened);
        }
        let config = Config {
            check_conflicts: plan.checks.conflicts,
            signal: plan.signal,
            check_novelty: plan.checks.novelty,
            ..Config::with_budget(plan.round_words.min(left))
        };
        let selection = select(
            candidates,
            &learned_set,
            &Open {
                gaps: &gaps,
                semantic: evidence.semantic,
                objective: if plan.checks.objective {
                    &goal.objective
                } else {
                    ""
                },
            },
            &vocabulary,
            &counter,
            source,
            &config,
        );
        questions += counter.asked.get();
        let chosen: Vec<Candidate> = candidates
            .iter()
            .filter(|c| selection.chosen.contains(&c.id))
            .cloned()
            .collect();
        let nothing = chosen.is_empty();
        rounds.push(Round {
            round,
            pass_rate: rate,
            gaps: gaps.len(),
            selection,
        });
        if nothing {
            stop = if rate >= plan.target {
                Stop::GoalMet
            } else {
                Stop::NothingToLearn
            };
            break;
        }
        participant.learn(&chosen);
        for c in &chosen {
            words += c.words();
            learned.push(c.id.clone());
            learned_set.insert(c.id.clone());
        }
    }
    let counter = Counting {
        inner: &*participant,
        asked: Cell::new(0),
    };
    let final_pass_rate = pass_rate(&probe(goal, &counter)).unwrap_or(1.0);
    questions += counter.asked.get();
    Report {
        rounds,
        final_pass_rate,
        learned,
        words,
        questions,
        stop,
    }
}
