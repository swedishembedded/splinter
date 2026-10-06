// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the choice of one fine-tuning checkpoint on
// held-out generation, for its clients. If your team needs expertise in
// selecting a model by what it writes rather than by its loss, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Choosing the step of a run on what it writes.
//!
//! The loss on a monitoring set says how likely the held-out text is, not
//! whether the model that wrote an answer got it right, spoke as the person
//! or made up a specific. A run kept with the adapter of every evaluation
//! ([`crate::train::Tuning::keep_evaluations`]) can be asked the dev suite
//! ([`crate::exam_set::Role::Dev`]) at each of them: the families of the dev
//! suite are reserved apart from the final test and from everything trained
//! on, so the choice made on them does not touch the exam that judges it.
//!
//! The rule is fixed here, before any run is chosen by it:
//!
//! 1. A checkpoint whose answers invent specifics in more than
//!    [`MAX_INVENTED_SHARE`] of the answers checked is out. If every one is
//!    out, the one that invents least stands, and the choice says the gate
//!    was not passed.
//! 2. Of those left, the best share of answers the judge calls right, and
//!    every one within a standard error of it, is in the running.
//! 3. Of those, the one that gives the writer's own text of the dev families
//!    the lowest loss; the monitoring loss, then the earlier step, decide a
//!    tie.
//!
//! Correctness decides, voice breaks the tie correctness cannot, and a
//! checkpoint that invents is never chosen for its voice.

use std::path::PathBuf;

use serde::Serialize;
use splinter_agent::CancelToken;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use crate::exam_set::ExamSet;
use crate::powered::{run, ArmChoice, PoweredRequest};
use crate::train::{load_candidate, Candidate, Checkpoint};
use splinter_core::model_ref::ModelRef;

/// The most answers that may state what the source does not hold, as a share
/// of those checked, in a checkpoint that is chosen.
pub const MAX_INVENTED_SHARE: f64 = 0.08;

/// What one checkpoint did on the dev suite.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Scored {
    /// Steps trained when its adapter was taken.
    pub step: u32,
    /// The monitoring loss at that step.
    pub monitor_loss: f32,
    /// Tasks the judge decided on the greedy answer.
    pub judged: usize,
    /// Of those, the ones it called right.
    pub right: usize,
    /// Answers checked against the source.
    pub checked: usize,
    /// Of those, the ones that state what the source does not hold.
    pub invented: usize,
    /// Mean cross-entropy per token of the writer's own text of the dev
    /// families; `None` when it could not be scored.
    pub voice_loss: Option<f32>,
}

impl Scored {
    fn right_share(&self) -> f64 {
        if self.judged == 0 {
            0.0
        } else {
            self.right as f64 / self.judged as f64
        }
    }

    fn invented_share(&self) -> f64 {
        if self.checked == 0 {
            0.0
        } else {
            self.invented as f64 / self.checked as f64
        }
    }
}

/// The checkpoint chosen and how.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Chosen {
    /// Its step.
    pub step: u32,
    /// Whether it passed the gate on invented specifics; `false` when none
    /// did and it is the one that invents least.
    pub gate_passed: bool,
    /// How many checkpoints were in the running at the voice tie-break.
    pub in_the_running: usize,
}

/// The checkpoint of `scored` the rule in the module documentation chooses;
/// `None` when there is none to choose from.
#[must_use]
pub fn choose(scored: &[Scored]) -> Option<Chosen> {
    let eligible: Vec<&Scored> = scored
        .iter()
        .filter(|s| s.invented_share() <= MAX_INVENTED_SHARE)
        .collect();
    if eligible.is_empty() {
        let least = scored.iter().min_by(|a, b| {
            a.invented_share()
                .total_cmp(&b.invented_share())
                .then(b.right_share().total_cmp(&a.right_share()))
                .then(a.step.cmp(&b.step))
        })?;
        return Some(Chosen {
            step: least.step,
            gate_passed: false,
            in_the_running: 1,
        });
    }
    let best = eligible
        .iter()
        .max_by(|a, b| a.right_share().total_cmp(&b.right_share()))?;
    let p = best.right_share();
    let se = (p * (1.0 - p) / best.judged.max(1) as f64).sqrt();
    let running: Vec<&Scored> = eligible
        .iter()
        .copied()
        .filter(|s| s.right_share() >= p - se)
        .collect();
    let winner = running.iter().min_by(|a, b| {
        let voice = |s: &Scored| s.voice_loss.unwrap_or(f32::INFINITY);
        voice(a)
            .total_cmp(&voice(b))
            .then(a.monitor_loss.total_cmp(&b.monitor_loss))
            .then(a.step.cmp(&b.step))
    })?;
    Some(Chosen {
        step: winner.step,
        gate_passed: true,
        in_the_running: running.len(),
    })
}

/// What choosing a checkpoint is asked.
pub struct SelectRequest<'a> {
    /// The candidate, by id or unique prefix; it kept its evaluations.
    pub candidate: &'a str,
    /// The dev suite.
    pub exam: &'a ExamSet,
    /// The judge; `None` is the judge role's.
    pub judge: Option<&'a ModelRef>,
    /// Stops the choice.
    pub cancel: &'a CancelToken,
}

/// What choosing a checkpoint came to.
#[derive(Clone, Debug, Serialize)]
pub struct Selected {
    /// The candidate whose checkpoints were scored.
    pub candidate: String,
    /// The dev suite.
    pub exam: String,
    /// Every checkpoint, by step.
    pub scored: Vec<Scored>,
    /// The one chosen.
    pub chosen: Chosen,
}

/// Puts each kept evaluation of `request.candidate` to the dev suite and
/// chooses one by [`choose`].
pub fn select(ctx: &Context, request: &SelectRequest<'_>) -> Result<Selected, OrchestratorError> {
    let candidate = load_candidate(ctx, request.candidate)?;
    if candidate.checkpoints.is_empty() {
        return Err(OrchestratorError::Refused(format!(
            "candidate {} kept no evaluations to choose among: train with --keep-evaluations",
            candidate.candidate
        )));
    }
    let mut scored = Vec::with_capacity(candidate.checkpoints.len());
    for checkpoint in &candidate.checkpoints {
        scored.push(score(ctx, request, &candidate, checkpoint)?);
    }
    let chosen = choose(&scored).ok_or_else(|| {
        OrchestratorError::Refused("no checkpoint could be scored on the dev suite".into())
    })?;
    Ok(Selected {
        candidate: candidate.candidate,
        exam: request.exam.id.clone(),
        scored,
        chosen,
    })
}

fn score(
    ctx: &Context,
    request: &SelectRequest<'_>,
    candidate: &Candidate,
    checkpoint: &Checkpoint,
) -> Result<Scored, OrchestratorError> {
    let adapter: PathBuf = candidate.checkpoint_adapter(ctx, checkpoint)?;
    let report = run(
        ctx,
        &PoweredRequest {
            exam: request.exam,
            candidate: &candidate.candidate,
            base: None,
            judge: request.judge,
            goal: None,
            resamples: 1,
            pilot_families: None,
            voice: true,
            arms: ArmChoice::Deployed,
            adapter: Some(&adapter),
            cancel: request.cancel,
        },
    )?;
    let arm = report
        .arms
        .first()
        .ok_or_else(|| OrchestratorError::Refused("the dev suite was asked of no arm".into()))?;
    Ok(Scored {
        step: checkpoint.step,
        monitor_loss: checkpoint.monitor_loss,
        judged: arm.judged,
        right: arm.right,
        checked: arm.checked,
        invented: arm.invented,
        voice_loss: report.voice.first().and_then(|v| v.loss),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(step: u32, right: usize, invented: usize, voice: f32) -> Scored {
        Scored {
            step,
            monitor_loss: 1.0 + step as f32 / 100.0,
            judged: 100,
            right,
            checked: 100,
            invented,
            voice_loss: Some(voice),
        }
    }

    /// Correctness decides; voice only breaks a tie within a standard error.
    #[test]
    fn the_best_answers_win_and_voice_decides_among_the_equal() {
        // 60 right is 5 points clear of 55, more than a standard error
        // (about five points) of the best: the voice of the 55 does not count.
        let clear = [at(30, 60, 2, 1.30), at(60, 50, 2, 1.00)];
        assert_eq!(choose(&clear).map(|c| c.step), Some(30));
        // 60 and 58 are within an error of each other: the better voice wins.
        let close = [at(30, 60, 2, 1.30), at(60, 58, 2, 1.10), at(90, 40, 0, 0.9)];
        let chosen = choose(&close).unwrap();
        assert_eq!((chosen.step, chosen.in_the_running), (60, 2));
    }

    /// A checkpoint that invents is never chosen for being right or for its
    /// voice; when every one invents, the one that invents least stands and
    /// the choice says so.
    #[test]
    fn a_checkpoint_that_invents_is_out_and_the_gate_failing_is_said() {
        let some = [at(30, 70, 20, 1.0), at(60, 55, 3, 1.4)];
        let chosen = choose(&some).unwrap();
        assert_eq!((chosen.step, chosen.gate_passed), (60, true));
        let all = [at(30, 70, 20, 1.0), at(60, 55, 12, 1.4)];
        let chosen = choose(&all).unwrap();
        assert_eq!((chosen.step, chosen.gate_passed), (60, false));
        assert_eq!(choose(&[]), None);
    }

    /// With nothing to separate them the lower monitoring loss, then the
    /// earlier step, decides.
    #[test]
    fn a_tie_goes_to_the_lower_monitoring_loss_then_the_earlier_step() {
        let mut a = at(30, 50, 0, 1.2);
        let mut b = at(60, 50, 0, 1.2);
        a.monitor_loss = 1.1;
        b.monitor_loss = 1.0;
        assert_eq!(choose(&[a.clone(), b.clone()]).map(|c| c.step), Some(60));
        b.monitor_loss = 1.1;
        assert_eq!(choose(&[a, b]).map(|c| c.step), Some(30));
    }
}
