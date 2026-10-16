// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements paired, family-clustered comparisons of a
// trained code-editing agent against its untrained base, for its clients. If
// your team needs expertise in showing that fine-tuning an agent beat its
// base on held-out repository tasks, you can procure our services by
// sending an email to info@swedishembedded.com.

//! `experiences compare`: two experience sets of the same tasks, the
//! candidate's against the baseline's.
//!
//! Experiences are paired by task id. An attempt is classified by what the
//! store holds about it, and only a graded one carries evidence:
//!
//! * **exposed**: the exposure audit failed it, so its trajectory shows the
//!   solver reached for what only the grader holds;
//! * **harness-failed**: no verifier decided it, or it is recorded as an
//!   environment failure - it says nothing about the model;
//! * **graded**: the store's decision rule ([`decide`]) passed or failed it.
//!
//! Exposed and harness-failed attempts are in no denominator and are
//! counted. A task's verdict in an arm is what a strict majority of its
//! graded attempts say; with none, or a tie, it has no verdict and the task
//! is unpaired. A task is in one family, its first declared concept, and a
//! family is one unit of evidence ([`by_cluster`]): the sign test is the one
//! the release gate applies, over those units.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use splinter_core::annotation::Annotation;
use splinter_core::digest::Digest;
use splinter_core::experience::Experience;
use splinter_core::outcome::FailureClass;
use splinter_core::rng::Seed;
use splinter_eval::competence::{competence, outcome_of, Attempt, Competence};
use splinter_eval::paired::{by_cluster, PairedOutcome};
use splinter_eval::significance::{SignTest, Significance};
use splinter_model::stats::BrainSignificance;
use splinter_store::decision::decide;
use splinter_store::experiences::SetId;

use crate::verify::exposure_failure;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What one arm's attempts came to.
#[derive(Clone, Debug, Serialize)]
pub struct ArmReport {
    /// Attempts in the arm's set.
    pub attempts: usize,
    /// Of those, the ones graded: in the denominators.
    pub graded: usize,
    /// Attempts whose trajectory shows exposure to the grader's material.
    pub exposed: usize,
    /// Attempts no verifier decided, or recorded as environment failures.
    pub harness_failed: usize,
    /// Quality, cost and autonomy over the graded attempts, intervals
    /// resampled over families.
    pub competence: Competence,
}

/// One family's tasks that both arms were graded on.
#[derive(Clone, Debug, Serialize)]
pub struct FamilyLine {
    /// The family: the first concept its tasks declare.
    pub family: String,
    /// Tasks in it with a verdict in both arms.
    pub paired: usize,
    /// Of those, the ones the candidate got right.
    pub candidate_right: usize,
    /// Of those, the ones the baseline got right.
    pub baseline_right: usize,
    /// The candidate's fraction right; `None` with no paired task.
    pub candidate_fraction: Option<f64>,
    /// The baseline's fraction right; `None` with no paired task.
    pub baseline_fraction: Option<f64>,
    /// Paired tasks only the candidate got right.
    pub wins: usize,
    /// Paired tasks only the baseline got right.
    pub losses: usize,
    /// Paired tasks both got right or both got wrong.
    pub ties: usize,
}

/// How the families came out: a family is a win when the candidate won more
/// of its tasks than it lost, a loss when fewer, a tie when equal.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ClusterTally {
    /// Families the candidate won.
    pub wins: usize,
    /// Families the baseline won.
    pub losses: usize,
    /// Families neither won.
    pub ties: usize,
}

/// What `experiences compare` reports.
#[derive(Clone, Debug, Serialize)]
pub struct SetComparison {
    /// The candidate's experience set.
    pub candidate_set: SetId,
    /// The baseline's experience set.
    pub baseline_set: SetId,
    /// The seed the intervals were resampled with.
    pub interval_seed: Seed,
    /// Tasks either arm holds an attempt at.
    pub tasks: usize,
    /// Tasks with a verdict in both arms.
    pub paired: usize,
    /// Tasks without: excluded from every number below but the arm reports.
    pub unpaired: usize,
    /// The candidate arm.
    pub candidate: ArmReport,
    /// The baseline arm.
    pub baseline: ArmReport,
    /// Each family with a paired task, in name order.
    pub families: Vec<FamilyLine>,
    /// The families as units of evidence.
    pub clusters: ClusterTally,
    /// The one-sided sign test over those units: the chance of at least this
    /// many candidate wins if the candidate were no better.
    pub sign_test: SignTest,
    /// The share of paired tasks on which both arms reached the same
    /// verdict; `None` with no paired task.
    pub agreement: Option<f64>,
}

/// How one attempt ended.
enum Ending {
    Graded(bool),
    Exposed,
    HarnessFailed,
}

/// One arm's attempts at one task, classified.
struct Classified {
    family: String,
    endings: Vec<Ending>,
}

/// An arm: its attempts by task, and the measured ones for competence.
struct Arm {
    by_task: BTreeMap<Digest, Classified>,
    measured: Vec<Attempt>,
    attempts: usize,
}

/// Compares the experience sets `candidate` and `baseline`; see the module documentation.
pub fn compare(
    ctx: &Context,
    candidate: &SetId,
    baseline: &SetId,
    interval_seed: Seed,
) -> Result<SetComparison, OrchestratorError> {
    let candidate_arm = read_arm(ctx, candidate)?;
    let baseline_arm = read_arm(ctx, baseline)?;

    let tasks: BTreeSet<&Digest> = candidate_arm
        .by_task
        .keys()
        .chain(baseline_arm.by_task.keys())
        .collect();
    let mut outcomes = Vec::with_capacity(tasks.len());
    for task in &tasks {
        let verdict = |arm: &Arm| arm.by_task.get(*task).and_then(|c| majority(&c.endings));
        let family = candidate_arm
            .by_task
            .get(*task)
            .or_else(|| baseline_arm.by_task.get(*task))
            .map(|c| c.family.clone());
        outcomes.push(PairedOutcome {
            item: task.to_string(),
            candidate: verdict(&candidate_arm),
            baseline: verdict(&baseline_arm),
            cluster: family,
        });
    }

    let units = by_cluster(&outcomes);
    let clusters = ClusterTally {
        wins: units.iter().filter(|u| **u == (true, false)).count(),
        losses: units.iter().filter(|u| **u == (false, true)).count(),
        ties: units.iter().filter(|u| u.0 == u.1).count(),
    };
    let sign_test = BrainSignificance.sign_test(&units);
    let paired: Vec<(bool, bool)> = outcomes.iter().filter_map(PairedOutcome::paired).collect();
    let agreement = (!paired.is_empty())
        .then(|| paired.iter().filter(|(c, b)| c == b).count() as f64 / paired.len() as f64);

    Ok(SetComparison {
        candidate_set: candidate.clone(),
        baseline_set: baseline.clone(),
        interval_seed,
        tasks: tasks.len(),
        paired: paired.len(),
        unpaired: tasks.len() - paired.len(),
        candidate: report_of(candidate_arm, interval_seed),
        baseline: report_of(baseline_arm, interval_seed),
        families: family_lines(&outcomes),
        clusters,
        sign_test,
        agreement,
    })
}

fn read_arm(ctx: &Context, set: &SetId) -> Result<Arm, OrchestratorError> {
    let store = ctx.experiences();
    let members = store.get_set(set)?.members;
    let mut arm = Arm {
        by_task: BTreeMap::new(),
        measured: Vec::new(),
        attempts: members.len(),
    };
    for id in members {
        let experience = store.get(&id)?;
        let notes = store.annotations(&id)?.annotations;
        let family = family_of(&experience);
        let (ending, measured) = classify(&experience, &notes)?;
        if let Some(outcome) = measured {
            arm.measured.push(Attempt {
                family: family.clone(),
                outcome,
            });
        }
        arm.by_task
            .entry(experience.task.id.clone())
            .or_insert_with(|| Classified {
                family,
                endings: Vec::new(),
            })
            .endings
            .push(ending);
    }
    Ok(arm)
}

/// The family of `experience`: its task's first declared concept; a task
/// that declares none is a family of its own.
fn family_of(experience: &Experience) -> String {
    experience
        .concepts
        .first()
        .cloned()
        .unwrap_or_else(|| experience.task.id.to_string())
}

/// How `experience` ended, and its measured outcome when it is in the
/// competence denominators.
fn classify(
    experience: &Experience,
    notes: &[Annotation],
) -> Result<(Ending, Option<splinter_core::outcome::TaskOutcome>), OrchestratorError> {
    if notes.iter().any(|note| exposure_failure(note).is_some()) {
        return Ok((Ending::Exposed, None));
    }
    let outcome = outcome_of(experience, notes)?;
    // The competence report counts an environment failure by itself, so the
    // attempt stays in it.
    if outcome.failure == Some(FailureClass::Environment) {
        return Ok((Ending::HarnessFailed, Some(outcome)));
    }
    Ok(match decide(notes) {
        Some(decision) => (Ending::Graded(decision.passed), Some(outcome)),
        None => (Ending::HarnessFailed, None),
    })
}

/// What a strict majority of the graded endings say; `None` with no graded
/// attempt or no majority.
fn majority(endings: &[Ending]) -> Option<bool> {
    let graded: Vec<bool> = endings
        .iter()
        .filter_map(|e| match e {
            Ending::Graded(passed) => Some(*passed),
            _ => None,
        })
        .collect();
    let passed = graded.iter().filter(|p| **p).count();
    let failed = graded.len() - passed;
    (passed != failed).then_some(passed > failed)
}

fn report_of(arm: Arm, seed: Seed) -> ArmReport {
    let count = |wanted: fn(&Ending) -> bool| {
        arm.by_task
            .values()
            .flat_map(|c| &c.endings)
            .filter(|e| wanted(e))
            .count()
    };
    ArmReport {
        attempts: arm.attempts,
        graded: count(|e| matches!(e, Ending::Graded(_))),
        exposed: count(|e| matches!(e, Ending::Exposed)),
        harness_failed: count(|e| matches!(e, Ending::HarnessFailed)),
        competence: competence(&arm.measured, seed.value()),
    }
}

fn family_lines(outcomes: &[PairedOutcome]) -> Vec<FamilyLine> {
    let mut lines: BTreeMap<&str, FamilyLine> = BTreeMap::new();
    for outcome in outcomes {
        let (Some((candidate, baseline)), Some(family)) =
            (outcome.paired(), outcome.cluster.as_deref())
        else {
            continue;
        };
        let line = lines.entry(family).or_insert_with(|| FamilyLine {
            family: family.to_string(),
            paired: 0,
            candidate_right: 0,
            baseline_right: 0,
            candidate_fraction: None,
            baseline_fraction: None,
            wins: 0,
            losses: 0,
            ties: 0,
        });
        line.paired += 1;
        line.candidate_right += usize::from(candidate);
        line.baseline_right += usize::from(baseline);
        match (candidate, baseline) {
            (true, false) => line.wins += 1,
            (false, true) => line.losses += 1,
            _ => line.ties += 1,
        }
    }
    lines
        .into_values()
        .map(|mut line| {
            let n = line.paired as f64;
            line.candidate_fraction = Some(line.candidate_right as f64 / n);
            line.baseline_fraction = Some(line.baseline_right as f64 / n);
            line
        })
        .collect()
}
