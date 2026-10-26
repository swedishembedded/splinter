// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! One training round on one skill.
//!
//! Tasks are written from the skill's sections, each kept only if it
//! exercises the skill and repeats no probe; the policy as it stands attempts
//! them k times; what no attempt solved a teacher solves open-book; the
//! tasks worth training on (failed sometimes, with a verified answer) become
//! the round's dataset, which is trained on beside the datasets of the rounds
//! accepted before, from the policy the run began with. The candidate is
//! measured on the dev probes of the target skill and of every skill measured
//! before, and is accepted only if the target improved and no other skill
//! lost more than the retention bound. An accepted candidate becomes the
//! policy for the next round; a rejected one is dropped, and the policy is
//! exactly what it was.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_eval::frontier::Distribution;
use splinter_knowledge::skills::Skill;

use super::competence::SkillCompetence;
use super::cost::StageCost;
use super::probes::{score, set_of, write_until, Ask, ProbeScore};
use super::Acquisition;
use crate::curriculum::frontier::{select_frontier, tally};
use crate::curriculum::repository_teacher::{
    teach_repository, RepositoryTeacher, TeachRepositoryRequest,
};
use crate::curriculum::teacher::{teach, TeachRequest};
use crate::datasets::{build, BuildRequest, ViewName, VoiceBuild, DEFAULT_MIN_STRENGTH};
use crate::release::arm;
use crate::solving::{solve_tasks, SolveRequest};
use crate::train::{
    train_full_recorded_as, train_recorded_as, Candidate, FullCandidate, FullRequest, TrainRequest,
    DEFAULT_LORA_RANK,
};
use crate::verify::{verify_set, Grading};
use splinter_agent::work::WorkProtocol;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::runs::Recorder;

/// A skill's probe score before and after a round.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Change {
    /// Before the round.
    pub before: ProbeScore,
    /// With the candidate.
    pub after: ProbeScore,
}

impl Change {
    fn rate(score: ProbeScore) -> Option<f64> {
        (score.graded > 0).then(|| score.passes as f64 / score.graded as f64)
    }

    /// The change in pass rate; `None` when either side measured nothing.
    #[must_use]
    pub fn delta(&self) -> Option<f64> {
        Some(Self::rate(self.after)? - Self::rate(self.before)?)
    }
}

/// How a round ended.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    /// The candidate is the policy now.
    Accepted,
    /// The candidate was dropped, and why.
    Rejected {
        /// Why.
        reasons: Vec<String>,
    },
    /// Nothing was trained, and why; the round still counts against the
    /// skill.
    Skipped {
        /// Why.
        reason: String,
    },
}

/// What one round did.
#[derive(Clone, Debug, Serialize)]
pub struct Round {
    /// Which round, from 1.
    pub round: usize,
    /// The skill trained.
    pub skill: String,
    /// The benefit it was selected for.
    pub benefit: f64,
    /// The model that attempted the round's tasks and was measured as the
    /// round's starting point: the policy the run began with, then the last
    /// accepted candidate.
    pub incumbent: String,
    /// The model that wrote the round's training tasks.
    pub generator: String,
    /// Tasks the generator proposed, admitted or not.
    pub proposed: usize,
    /// Tasks admitted: they exercise the skill and repeat no probe or
    /// earlier task.
    pub written: usize,
    /// Tasks written that were refused, by reason.
    pub refused: BTreeMap<String, usize>,
    /// Attempts the incumbent made at the admitted tasks.
    pub attempts: usize,
    /// Of those, attempts with a decided verdict.
    pub graded: usize,
    /// Of those, verified passes.
    pub verified_passes: usize,
    /// What the frontier made of the tasks.
    pub frontier: Option<Distribution>,
    /// Tasks worth training on: failed sometimes with a verified answer, or
    /// answered by the teacher where the incumbent never solved them.
    pub worth_training: usize,
    /// Records in the round's dataset.
    pub records: Option<usize>,
    /// What the incumbent's attempts (solving and verifying) cost.
    pub solve: Option<StageCost>,
    /// What training the candidate cost; no device time is reported for it.
    pub train: Option<StageCost>,
    /// The candidate, when one was trained.
    pub candidate: Option<String>,
    /// The directory of the full checkpoint the round trained. It is
    /// deleted when the round is rejected or a later round supersedes it,
    /// unless the request keeps candidates.
    pub checkpoint: Option<String>,
    /// The target skill before and after.
    pub target: Option<Change>,
    /// Every other skill measured, before and after.
    pub others: BTreeMap<String, Change>,
    /// How it ended.
    pub outcome: Outcome,
    /// The competence graph after it.
    pub competence: Digest,
}

/// What a round trained.
enum Trained {
    /// A LoRA adapter on the start policy.
    Adapter(Box<Candidate>),
    /// Every weight, as a checkpoint directory.
    Full(Box<FullCandidate>),
}

impl Trained {
    fn id(&self) -> &str {
        match self {
            Self::Adapter(c) => &c.candidate,
            Self::Full(f) => &f.id,
        }
    }

    /// The candidate as a model to attempt and measure.
    fn model(&self, ctx: &splinter_orchestrator::context::Context) -> ModelRef {
        match self {
            Self::Adapter(c) => arm(ctx.config(), Some(&c.adapter)),
            Self::Full(f) => f.model.clone(),
        }
    }
}

/// The training tasks a round asks the generator for before it attempts
/// them: a 4B generator admits none or one of eight in a request once its
/// skill's probes are taken, and a round with fewer than two records cannot
/// train.
pub const ROUND_TASKS_WANTED: usize = 12;

/// The most requests one round's tasks take.
pub const MAX_ROUND_WRITES: usize = 6;

/// The goal a generator is given for training tasks of `skill`.
fn goal_of(objective: &str, skill: &Skill) -> String {
    let requirements: Vec<&str> = skill
        .requirements
        .iter()
        .map(|r| r.statement.as_str())
        .collect();
    format!(
        "{objective} (training). Skill: {}. It must be shown to: {}",
        skill.title,
        requirements.join(" ")
    )
}

impl Acquisition<'_> {
    /// Runs round `round` on `skill` (selected for `benefit`).
    pub(super) fn round(
        &mut self,
        run: &mut Recorder<'_>,
        round: usize,
        skill: &Skill,
        benefit: f64,
    ) -> Result<Round, OrchestratorError> {
        let ctx = self.ctx;
        let cancel = run.cancel_token();
        let mut report = Round {
            round,
            skill: skill.id.clone(),
            benefit,
            incumbent: self.incumbent.to_string(),
            generator: self.author().to_string(),
            proposed: 0,
            written: 0,
            refused: BTreeMap::new(),
            attempts: 0,
            graded: 0,
            verified_passes: 0,
            frontier: None,
            worth_training: 0,
            records: None,
            solve: None,
            train: None,
            candidate: None,
            checkpoint: None,
            target: None,
            others: BTreeMap::new(),
            outcome: Outcome::Skipped {
                reason: String::new(),
            },
            competence: self.competence_address.clone(),
        };
        let skip = |report: &mut Round, reason: &str| {
            report.outcome = Outcome::Skipped {
                reason: reason.into(),
            };
        };

        let generator = self.author().clone();
        // A training task must not repeat a probe of its skill, so the
        // generator is shown them rather than refused its repeats afterwards.
        let probe_instructions = self.probe_instructions(&skill.id)?;
        let written = write_until(
            ctx,
            skill,
            &Ask {
                kinds: &self.kinds,
                generator: &generator,
                goal: goal_of(&self.request.objective, skill),
                deadline: self.deadline,
                cancel: cancel.clone(),
            },
            &probe_instructions,
            ROUND_TASKS_WANTED,
            MAX_ROUND_WRITES,
            &mut self.seen,
        )?;
        report.written = written.tasks.len();
        report.proposed = report.written + written.refused.values().sum::<usize>();
        report.refused = written.refused;
        if written.tasks.is_empty() {
            skip(
                &mut report,
                "no task was written that exercises the skill and repeats no probe",
            );
            return self.close(run, report, skill);
        }
        let digests: Vec<Digest> = written.tasks.iter().map(|e| e.task.clone()).collect();
        let tasks = set_of(ctx, &format!("round {round}: {}", skill.id), &digests)?;

        let agentic = self.request.pass_at_k;
        let solving = Instant::now();
        let solved = solve_tasks(
            ctx,
            &SolveRequest {
                task_set: &tasks,
                solver: &self.incumbent,
                attempts: agentic.k,
                sampling: agentic.sampling_choice(),
                teacher: false,
                system: None,
                work: WorkProtocol::default(),
                deadline: self.deadline,
                harness: None,
                resume: None,
                cancel: cancel.clone(),
            },
        )?;
        verify_set(ctx, &solved.experience_set, Grading::ActiveJudge, &cancel)?;
        report.solve = Some(StageCost::of_attempts(
            ctx,
            solving,
            &solved.experience_set,
        )?);
        for (count, _) in tally(ctx, &solved.experience_set)?.into_values() {
            report.attempts += count.attempts;
            report.graded += count.graded;
            report.verified_passes += count.passes;
        }
        let taught = if self.request.reference_teacher {
            teach_repository(
                ctx,
                &TeachRepositoryRequest {
                    task_set: &tasks,
                    attempts: Some(&solved.experience_set),
                    teacher: &RepositoryTeacher::Reference,
                    work: WorkProtocol::default(),
                    deadline: self.deadline,
                    cancel: cancel.clone(),
                },
            )?
        } else {
            teach(
                ctx,
                &TeachRequest {
                    task_set: &tasks,
                    attempts: Some(&solved.experience_set),
                    teacher: &self.teacher,
                    system: None,
                    deadline: self.deadline,
                    cancel: cancel.clone(),
                },
            )?
        };
        let kept = select_frontier(ctx, &tasks, &solved, &taught)?;
        report.frontier = Some(kept.distribution);
        report.worth_training = kept.distribution.kept();
        if kept.distribution.kept() == 0 {
            skip(
                &mut report,
                "no task is worth training on: none is failed sometimes with a verified answer",
            );
            return self.close(run, report, skill);
        }

        let built = match build(
            ctx,
            &BuildRequest {
                sets: vec![kept.frontier_experience_set.clone()],
                view: ViewName::SftFinal,
                strip: None,
                min_strength: Some(DEFAULT_MIN_STRENGTH),
                system_prompt: None,
                export_only: false,
                limit: None,
                voice: VoiceBuild::default(),
                max_block: None,
                pack: false,
                split_over_block: false,
                control: Default::default(),
                serialise_calls: false,
                only_actions: None,
            },
        ) {
            Ok(built) => built,
            Err(OrchestratorError::View(splinter_data::ViewError::Empty { .. })) => {
                skip(
                    &mut report,
                    "no verified answer could be made a training record",
                );
                return self.close(run, report, skill);
            }
            Err(e) => return Err(e),
        };
        report.records = Some(built.records);
        if built.records < splinter_data::holdout::MIN_SAMPLES {
            skip(
                &mut report,
                &format!(
                    "{} record(s): training holds records out for scoring and needs at least {}",
                    built.records,
                    splinter_data::holdout::MIN_SAMPLES
                ),
            );
            return self.close(run, report, skill);
        }
        self.assert_no_probe_trained_on(&kept.frontier_task_set)?;

        let training = Instant::now();
        let candidate = self.train_candidate(run, round, &built.dataset.to_string())?;
        report.train = Some(StageCost::wall_only(training));
        report.candidate = Some(candidate.id().to_string());
        if let Trained::Full(full) = &candidate {
            report.checkpoint = Some(full.checkpoint.dir.display().to_string());
        }
        let tuned = candidate.model(self.ctx);

        // The target and every skill measured before, on their dev probes.
        let after = match self.measure(&tuned, round, &cancel) {
            Ok(after) => after,
            Err(e) => {
                self.dispose(candidate);
                return Err(e);
            }
        };
        let changes: BTreeMap<String, Change> = after
            .iter()
            .filter_map(|(id, after)| {
                self.competence.get(id).map(|before| {
                    (
                        id.clone(),
                        Change {
                            before: ProbeScore {
                                graded: before.graded,
                                passes: before.passes,
                            },
                            after: *after,
                        },
                    )
                })
            })
            .collect();
        report.target = changes.get(&skill.id).copied();
        report.others = changes
            .iter()
            .filter(|(id, _)| **id != skill.id)
            .map(|(id, c)| (id.clone(), *c))
            .collect();

        let mut reasons = Vec::new();
        match report.target.and_then(|c| c.delta()) {
            Some(delta) if delta > 0.0 => {}
            Some(delta) => reasons.push(format!(
                "{} did not improve on its dev probes (change {delta:+.3})",
                skill.id
            )),
            None => reasons.push(format!(
                "{} could not be measured with the candidate",
                skill.id
            )),
        }
        for (id, change) in &report.others {
            if let Some(delta) = change.delta().filter(|d| -d > self.request.retention_drop) {
                reasons.push(format!(
                    "{id} fell by {:.3}, more than the retention bound {}",
                    -delta, self.request.retention_drop
                ));
            }
        }
        if reasons.is_empty() {
            report.outcome = Outcome::Accepted;
            self.incumbent = tuned;
            self.datasets.push(built.dataset.to_string());
            self.adopt(candidate);
            // The competence of everything the candidate was measured on.
            for (id, scored) in after {
                let rounds =
                    self.competence.get(&id).map_or(0, |c| c.rounds) + usize::from(id == skill.id);
                if let Some(update) =
                    SkillCompetence::measured(&id, scored.passes, scored.graded, rounds)
                {
                    self.update_competence(update, round)?;
                }
            }
        } else {
            report.outcome = Outcome::Rejected { reasons };
            self.dispose(candidate);
        }
        self.close(run, report, skill)
    }

    /// Trains the round's candidate on `dataset`: an adapter retrained from
    /// the start policy over every accepted round's dataset, or, with
    /// `full_fine_tune`, every weight of the incumbent on `dataset` beside a
    /// replay of the accepted rounds'.
    fn train_candidate(
        &self,
        run: &mut Recorder<'_>,
        round: usize,
        dataset: &str,
    ) -> Result<Trained, OrchestratorError> {
        let name = format!("seed-round-{round}");
        let request = self.request;
        if request.full_fine_tune {
            return train_full_recorded_as(
                self.ctx,
                run,
                &name,
                &FullRequest {
                    datasets: vec![dataset.to_string()],
                    replay_datasets: self.datasets.clone(),
                    replay_fraction: request.replay_fraction,
                    from: self.incumbent.clone(),
                    steps: request.steps,
                    tuning: request.tuning,
                    train_precision: request.train_precision.clone(),
                },
                self.trainer,
            )
            .map(|f| Trained::Full(Box::new(f)));
        }
        let mut datasets = self.datasets.clone();
        datasets.push(dataset.to_string());
        train_recorded_as(
            self.ctx,
            run,
            &name,
            &TrainRequest {
                datasets,
                rehearsal: None,
                from: ModelRef::policy_default(),
                replay_fraction: request.replay_fraction,
                replay_policy: Default::default(),
                steps: request.steps,
                rank: request.rank.unwrap_or(DEFAULT_LORA_RANK),
                beta: None,
                tuning: request.tuning,
            },
            self.trainer,
        )
        .map(|c| Trained::Adapter(Box::new(c)))
    }

    /// The target and every skill measured before, on their dev probes, as
    /// `candidate` answers them.
    fn measure(
        &self,
        candidate: &ModelRef,
        round: usize,
        cancel: &splinter_agent::CancelToken,
    ) -> Result<BTreeMap<String, ProbeScore>, OrchestratorError> {
        let mut after = BTreeMap::new();
        for measured in &self.competence.skills {
            let Some(probes) = self.probes.get(&measured.skill) else {
                continue;
            };
            let (scored, _) = score(
                self.ctx,
                candidate,
                &probes.dev,
                &format!("round {round}: dev probes of {}", measured.skill),
                cancel,
            )?;
            after.insert(measured.skill.clone(), scored);
        }
        Ok(after)
    }

    /// The instructions of `skill`'s probes, dev and final, in address order.
    fn probe_instructions(&self, skill: &str) -> Result<Vec<String>, OrchestratorError> {
        let Some(probes) = self.probes.get(skill) else {
            return Ok(Vec::new());
        };
        probes
            .dev
            .iter()
            .chain(&probes.finals)
            .map(|id| Ok(self.ctx.tasks().get(id)?.instruction))
            .collect()
    }

    /// Makes an accepted candidate the one the run stands on. A full
    /// checkpoint it supersedes is not needed any more.
    fn adopt(&mut self, candidate: Trained) {
        match candidate {
            Trained::Adapter(adapter) => self.last_candidate = Some(*adapter),
            Trained::Full(full) => {
                if let Some(superseded) = self.checkpoint.replace(*full) {
                    self.dispose(Trained::Full(Box::new(superseded)));
                }
            }
        }
    }

    /// Removes a full checkpoint nothing stands on, unless candidates are
    /// kept; an adapter is a small artifact and stays.
    fn dispose(&self, candidate: Trained) {
        if let Trained::Full(full) = candidate {
            if !self.request.keep_candidates {
                full.discard();
            }
        }
    }

    /// Counts the round against `skill` when nothing was accepted, records
    /// it, and returns it.
    fn close(
        &mut self,
        run: &mut Recorder<'_>,
        mut report: Round,
        skill: &Skill,
    ) -> Result<Round, OrchestratorError> {
        if !matches!(report.outcome, Outcome::Accepted) {
            if let Some(mut spent) = self.competence.get(&skill.id).cloned() {
                spent.rounds += 1;
                self.update_competence(spent, report.round)?;
            }
        }
        report.competence = self.competence_address.clone();
        run.stage(&format!("round-{}", report.round), &report)?;
        Ok(report)
    }

    /// Refuses to go on when a task of the round's training set is a probe:
    /// the checks that kept probes out of the tasks failed.
    fn assert_no_probe_trained_on(
        &self,
        training: &splinter_store::tasks::TaskSetId,
    ) -> Result<(), OrchestratorError> {
        for entry in self.ctx.tasks().get_set(training)?.members {
            if self.probe_ids.contains(&entry.task) {
                return Err(OrchestratorError::Refused(format!(
                    "task {} is a probe and would have been trained on",
                    entry.task
                )));
            }
        }
        Ok(())
    }
}
