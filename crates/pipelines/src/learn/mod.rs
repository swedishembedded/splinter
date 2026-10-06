// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that acquire a capability
// from a document or a tool and prove it with evidence, for its clients. If
// your team needs expertise in continual learning or agent evaluation, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `learn`: sources -> tasks -> solve -> verify -> teach -> frontier ->
//! variants -> critique and retry -> select -> dataset -> train -> release, as one
//! recorded run whose every stage is also a command of its own.
//!
//! Each stage hands the next the content-addressed set it wrote, and its
//! summary is recorded (and reported) as it finishes, so a stage can be
//! inspected, or rerun alone, from what the run records. The policy -
//! `policy:default` - is resolved once, when the run starts, and the
//! release it resolved to is the run's first recorded stage: the whole run
//! generates, solves, teaches, critiques, retries and trains from that
//! release, however the alias moves meanwhile. The tasks stage also
//! generates from the sections of every concept queued for new tasks
//! ([`crate::curriculum::queue`]). Each task is solved k times closed-book
//! and graded by its kind's verifiers, never by a judge; each task no
//! graded attempt solved is solved once more by the teacher - the policy,
//! or the model `--teacher` names - open-book, with its grounding material
//! shown, and graded the same way ([`crate::curriculum::teacher`]). Only
//! the tasks worth training on go on: those the student fails at least
//! sometimes and that have a verified answer, its own or the teacher's
//! ([`crate::curriculum::frontier`]; `--no-frontier` solves each task once
//! and keeps them all). The generator model then writes differently worded
//! questions about each task kept ([`crate::variants`]): the same facts,
//! asked in other words, that the release gate measures the candidate on
//! and that are never trained on. Failed attempts are critiqued and retried; the
//! passing attempts, verified teacher answers and revisions are
//! deduplicated and capped per concept, kind and strength
//! ([`crate::curriculum::quota`]), and the `sft-final` view over what is
//! selected is the dataset - the student's records: the instruction alone,
//! whatever the solver was shown. A run that learns to think like a person
//! trains beside it on the writer's own text (the `voice` view): a share of
//! the training tokens ([`LearnRequest::voice`]) built by code from the
//! sources, chunks of the writer's words word for word as the answer to a
//! request that names what the passage is, spread over the whole of the
//! sources and held out with the family of the letter it prints. Such a run also rehearses the base
//! model's own answers to general tasks (the `rehearse` stage,
//! [`crate::rehearsal`]): a share of the training draws
//! ([`LearnRequest::rehearsal`]) and of the monitoring set, so the persona
//! does not cost the model its general behaviour, which the release gate's
//! anchor suite holds it to. The training's step budget is a ceiling
//! of passes over the examples ([`crate::train::sizing`]); the run monitors
//! a share of its training families as it goes and carries its best
//! evaluation's adapter. The candidate trained on it continues
//! that release, and goes through the release gate, which grades it
//! closed-book and releases it only if every check passes (`--no-release`
//! stops at the candidate).
//!
//! The pipeline stops early, and says why, when a stage leaves the next
//! nothing to work on or the budget is spent; `--dry-run` resolves the
//! plan and writes nothing.

mod dataset_stage;
mod exam_stages;
mod report;
mod stages;

pub use report::{LearnPlan, LearnReport, Learned, Planned, PolicyUsed};
use stages::{Learn, LearnState};

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::author::kind_authors;
use crate::curriculum::frontier::PassAtK;
use crate::curriculum::quota::Quotas;
use crate::raft::PassageShare;

use crate::sources::SourceTarget;
use crate::tasks::{check_kinds, DEFAULT_LEARN_KINDS};
pub use crate::train::{auto_records_per_step, auto_steps, steps_for, MAX_AUTO_STEPS, MAX_PASSES};
use crate::train::{Trainer, Tuning, DEFAULT_LORA_RANK};

use splinter_core::model_ref::ModelRef;
use splinter_core::role::{Role, RoleOverrides};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::roles;
use splinter_orchestrator::runs::record;

/// The stages, in order, as runs and reports name them.
pub const STAGES: [&str; 20] = [
    "policy",
    "sources",
    "plan",
    "reserve",
    "exam-set",
    "tasks",
    "solve",
    "verify",
    "teach",
    "author",
    "frontier",
    "variants",
    "critique",
    "select",
    "dataset",
    "rehearse",
    "train",
    "checkpoint",
    "exam",
    "release",
];

/// The share of the training tokens that is the writer's own text when a
/// run learns to think like a person and names no share: most of what the
/// adapter reads is the writer's words, and the dialogues, which are what
/// the policy is asked as, keep the rest.
pub const DEFAULT_VOICE_SHARE: f64 = 0.7;

/// The share of the training draws that are the base model's own answers
/// to general tasks when a run learns to think like a person and names no
/// share: [`DEFAULT_REHEARSAL_SHARE`]. A run without a persona rehearses
/// nothing unless asked.
pub use crate::rehearsal::DEFAULT_REHEARSAL_SHARE;

/// How many records go beside `examples` so that they are `share` of all
/// the examples: `examples * share / (1 - share)`, rounded up; none for a
/// share of zero. What sizes the writer's own text and the rehearsal set
/// against the dialogue answers.
#[must_use]
pub fn records_at_share(examples: usize, share: f64) -> usize {
    if share <= 0.0 {
        return 0;
    }
    (examples as f64 * share / (1.0 - share)).ceil() as usize
}

/// How many tokens go beside `tokens` so that they are `share` of all the
/// tokens: `tokens * share / (1 - share)`, rounded up; none for a share of
/// zero. What sizes the writer's own text against the dialogue answers.
#[must_use]
pub fn tokens_at_share(tokens: usize, share: f64) -> usize {
    records_at_share(tokens, share)
}

/// The exam a run reserves up front.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ExamPlan {
    /// The families reserved for the final test: `None` is
    /// [`crate::reserve::DEFAULT_EXAM_FAMILIES`] for a run with a persona and
    /// none otherwise, `Some(0)` none at all.
    pub families: Option<usize>,
    /// Tasks of each family of the final test;
    /// [`crate::exam_set::DEFAULT_TASKS_PER_FAMILY`] when `None`.
    pub tasks_per_family: Option<usize>,
    /// The families reserved for the dev suite a checkpoint is chosen on:
    /// `None` is [`crate::reserve::DEFAULT_DEV_FAMILIES`] when a final test is
    /// reserved, `Some(0)` none.
    pub dev_families: Option<usize>,
    /// Tasks of each dev family;
    /// [`crate::exam_set::DEFAULT_DEV_TASKS_PER_FAMILY`] when `None`.
    pub dev_tasks_per_family: Option<usize>,
    /// Answers per task per arm; [`crate::powered::DEFAULT_RESAMPLES`] when
    /// `None`.
    pub resamples: Option<usize>,
}

/// One `learn`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LearnRequest {
    /// What to learn from, as the command line names it.
    pub sources: Vec<String>,
    /// What the learner is after.
    pub goal: Option<String>,
    /// Who the policy is to become, when it is to think like a person: the
    /// training records open with a system prompt that says so, and the
    /// policy is asked under it afterwards. A plan names one from the goal
    /// when this does not.
    pub persona: Option<String>,
    /// Give a share of the training records retrieved passages of the sources
    /// in their prompt ([`crate::raft`]), so that the policy learns to use
    /// context where it holds the answer and to answer without it where it
    /// does not. Needs the embedding model.
    pub passages: Option<PassageShare>,
    /// The share of the training tokens that is the writer's own text
    /// (the `voice` view) when the policy learns to think like a person, in
    /// `[0, 1)`: `None` is [`DEFAULT_VOICE_SHARE`] for a run with a persona
    /// and none without, `Some(0.0)` none at all.
    pub voice: Option<f64>,
    /// Ask the writer's passages by what the generator says they are about
    /// ([`crate::describe`]) instead of by their heading and opening.
    pub describe_voice: bool,
    /// The share of the training draws that are the base model's own
    /// answers to general tasks ([`crate::rehearsal`]), in `[0, 1)`: `None`
    /// is [`DEFAULT_REHEARSAL_SHARE`] for a run with a persona and none
    /// without, `Some(0.0)` none at all.
    pub rehearsal: Option<f64>,
    /// Task kinds; empty is [`DEFAULT_LEARN_KINDS`] unless `plan` is set.
    pub kinds: Vec<String>,
    /// Let a planner model survey the sources and choose the task kinds,
    /// and whether to distil. Refused together with `kinds`: one decides.
    pub plan: bool,
    /// Wall-clock time the whole pipeline may take.
    pub budget: Option<Duration>,
    /// Resolve the plan and write nothing.
    pub dry_run: bool,
    /// Stop at the candidate: do not run the release gate.
    pub no_release: bool,
    /// Solve each task once and keep every task, instead of measuring
    /// pass@k and keeping the frontier.
    pub no_frontier: bool,
    /// Skip the student's attempts: the teacher answers every task open-book,
    /// and the training set is its verified answers. For a student that
    /// cannot answer a task closed-book at all, where its attempts cost the
    /// most and teach nothing. Implies no frontier.
    pub distill: bool,
    /// The step budget of the training; [`auto_steps`] of the examples
    /// when `None`.
    pub steps: Option<u32>,
    /// LoRA rank of the adapter; [`DEFAULT_LORA_RANK`] when `None`.
    pub rank: Option<u32>,
    /// The base's precision, the learning rate and how the training is
    /// watched.
    pub tuning: Tuning,
    /// The models this run names for roles (teacher, generator, planner);
    /// a role not named is played as [`splinter_core::role`] says.
    pub roles: RoleOverrides,
    /// k and the sampling of the frontier's pass@k.
    pub pass_at_k: PassAtK,
    /// The diversity quotas on the training set.
    pub quotas: Quotas,
    /// The exam reserved before anything is generated.
    pub exam: ExamPlan,
    /// Keep the adapter of every evaluation of the training and choose the
    /// step by what each writes on the reserved dev suite
    /// ([`crate::checkpoints`]) instead of by the monitoring loss.
    pub select_on_dev: bool,
}

/// Runs `request`, training with `trainer`.
pub fn learn(
    ctx: &Context,
    request: &LearnRequest,
    trainer: &dyn Trainer,
) -> Result<Learned, OrchestratorError> {
    if request.plan && !request.kinds.is_empty() {
        return Err(OrchestratorError::Refused(
            "name the task kinds or ask for a plan, not both: one of them decides".into(),
        ));
    }
    let kinds = if request.plan {
        Vec::new()
    } else if request.kinds.is_empty() {
        DEFAULT_LEARN_KINDS.iter().map(|k| k.to_string()).collect()
    } else {
        check_kinds(&request.kinds)?
    };
    if request.sources.is_empty() {
        return Err(OrchestratorError::Refused(
            "name at least one source to learn from".into(),
        ));
    }
    let measures_frontier = !request.no_frontier && !request.distill;
    if measures_frontier {
        request.pass_at_k.validate()?;
    }
    request.quotas.validate()?;
    if let Some(share) = &request.passages {
        share.validate()?;
    }
    if let Some(share) = request.voice {
        if !(0.0..1.0).contains(&share) {
            return Err(OrchestratorError::Refused(format!(
                "the voice share {share} is not in [0, 1): the writer's own text cannot be every \
                 token, since the dialogues are what the policy is asked as"
            )));
        }
    }
    if let Some(share) = request.rehearsal {
        if !(0.0..1.0).contains(&share) {
            return Err(OrchestratorError::Refused(format!(
                "the rehearsal share {share} is not in [0, 1): the base's own answers cannot be \
                 every draw, since the new records are what the run is for"
            )));
        }
    }
    let targets = request
        .sources
        .iter()
        .map(|s| SourceTarget::from_learn_arg(s))
        .collect::<Result<Vec<_>, _>>()?;
    let policy = ModelRef::policy_default();
    let assignments = roles::assignments(ctx.config(), &request.roles)?;
    let teacher = assignments.get(Role::Teacher).clone();
    let generator = assignments.get(Role::Generator).clone();
    let judge = assignments.get(Role::Judge).clone();
    // The command's budget, else the configured default: a run on a corpus
    // too large to read has no end without one.
    let budget = match (request.budget, &ctx.config().default_budget) {
        (Some(named), _) => Some(named),
        (None, Some(text)) => Some(parse_budget(text).map_err(|e| {
            OrchestratorError::Refused(format!("SPLINTER_BUDGET is not usable: {e}"))
        })?),
        (None, None) => None,
    };
    let planner = request.plan.then(|| assignments.get(Role::Planner).clone());
    // Whether the named kinds include one whose reference is the writer's own
    // words, which the author stage puts forward as the answer.
    let authors = kinds.iter().any(|name| kind_authors(name));
    let mut roles_used = BTreeMap::new();
    for (role, model) in [
        (Role::Policy, &policy),
        (Role::Critic, assignments.get(Role::Critic)),
        (Role::Teacher, &teacher),
        (Role::Generator, &generator),
        (Role::Judge, &judge),
    ]
    .into_iter()
    .chain(planner.as_ref().map(|p| (Role::Planner, p)))
    {
        roles_used.insert(role, ctx.selection(model)?.identity());
    }
    if request.dry_run {
        return Ok(Learned::Planned(Box::new(LearnPlan {
            state: ctx.root().path().to_path_buf(),
            sources: targets,
            kinds,
            goal: request.goal.clone(),
            budget_secs: budget.map(|b| b.as_secs()),
            policy: ctx.selection(&policy)?.identity(),
            teacher: ctx.selection(&teacher)?.identity(),
            generator: ctx.selection(&generator)?.identity(),
            planner: planner
                .as_ref()
                .map(|p| ctx.selection(p).map(|s| s.identity()))
                .transpose()?,
            stages: STAGES
                .into_iter()
                .filter(|stage| !(request.no_release && *stage == "release"))
                .filter(|stage| !(!request.plan && *stage == "plan"))
                .filter(|stage| {
                    !(["reserve", "exam-set"].contains(stage)
                        && request
                            .exam
                            .families
                            .unwrap_or(if request.persona.is_some() {
                                crate::reserve::DEFAULT_EXAM_FAMILIES
                            } else {
                                0
                            })
                            == 0)
                })
                .filter(|stage| !(!request.select_on_dev && *stage == "checkpoint"))
                .filter(|stage| !(!measures_frontier && *stage == "frontier"))
                .filter(|stage| !(!request.plan && !authors && *stage == "author"))
                .filter(|stage| {
                    !(*stage == "rehearse"
                        && !request.plan
                        && request.rehearsal.unwrap_or(if request.persona.is_some() {
                            DEFAULT_REHEARSAL_SHARE
                        } else {
                            0.0
                        }) <= 0.0)
                })
                .filter(|stage| {
                    !(request.distill && ["solve", "verify", "critique"].contains(stage))
                })
                .collect(),
            dry_run: true,
        })));
    }
    let learn = Learn {
        targets: &targets,
        kinds: &kinds,
        planner: planner.as_ref(),
        goal: request.goal.as_deref(),
        persona: request.persona.as_deref(),
        voice: request.voice,
        describe_voice: request.describe_voice,
        rehearsal: request.rehearsal,
        passages: request.passages,
        deadline: budget.map(|b| Instant::now() + b),
        trainer,
        policy,
        teacher: &teacher,
        generator: &generator,
        judge: &judge,
        roles_used,
        no_release: request.no_release,
        steps: request.steps,
        rank: request.rank.unwrap_or(DEFAULT_LORA_RANK),
        tuning: request.tuning,
        quotas: request.quotas,
        exam: request.exam,
        select_on_dev: request.select_on_dev,
    };
    let frontier = measures_frontier.then_some(request.pass_at_k);
    let recorded = record(ctx, "learn", request, |run| {
        let mut state = LearnState::new(learn, request.distill, frontier);
        let deadline = state.deadline();
        let stopped = stages::pipeline().run(ctx, run, &mut state, deadline)?;
        let mut report = state.report;
        report.stopped = stopped;
        Ok(report)
    })?;
    Ok(Learned::Ran(Box::new(recorded)))
}

/// A duration as `learn --budget` takes it: whole units of `s`, `m` and
/// `h`, alone or combined largest first (`30m`, `1h30m`), or bare
/// seconds.
pub fn parse_budget(text: &str) -> Result<Duration, OrchestratorError> {
    let refuse = |why: &str| {
        OrchestratorError::Refused(format!(
            "{text:?} is not a duration ({why}): e.g. 30m, 2h, 1h30m; units h, m and s"
        ))
    };
    if text.is_empty() {
        return Err(refuse("it is empty"));
    }
    if let Ok(seconds) = text.parse::<u64>() {
        return positive(Duration::from_secs(seconds)).ok_or_else(|| refuse("it is zero"));
    }
    let mut total: u64 = 0;
    let mut number = String::new();
    let mut last_unit = u64::MAX;
    for c in text.chars() {
        if c.is_ascii_digit() {
            number.push(c);
            continue;
        }
        let unit = match c {
            'h' => 3600,
            'm' => 60,
            's' => 1,
            _ => return Err(refuse("units are h, m and s")),
        };
        if number.is_empty() {
            return Err(refuse("a unit needs a number before it"));
        }
        if unit >= last_unit {
            return Err(refuse("units go largest first, each once"));
        }
        let value: u64 = number
            .parse()
            .map_err(|_| refuse("the number is too large"))?;
        total = value
            .checked_mul(unit)
            .and_then(|v| total.checked_add(v))
            .ok_or_else(|| refuse("it is too long"))?;
        number.clear();
        last_unit = unit;
    }
    if !number.is_empty() {
        return Err(refuse("a number needs a unit after it"));
    }
    positive(Duration::from_secs(total)).ok_or_else(|| refuse("it is zero"))
}

fn positive(duration: Duration) -> Option<Duration> {
    (!duration.is_zero()).then_some(duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The list a report and a dry run show is the pipeline that runs.
    #[test]
    fn the_stage_list_is_the_pipeline() {
        assert_eq!(stages::pipeline().names(), STAGES);
    }

    /// The writer's text, or the rehearsal, is as many records as makes it
    /// the share asked.
    #[test]
    fn the_records_at_a_share_make_it_the_share_asked() {
        assert_eq!(records_at_share(214, 0.5), 214);
        assert_eq!(records_at_share(100, 0.25), 34);
        assert_eq!(records_at_share(100, 0.0), 0);
        assert_eq!(records_at_share(0, 0.5), 0);
    }

    #[test]
    fn a_budget_is_whole_units_largest_first() {
        let secs = |text: &str| parse_budget(text).ok().map(|d| d.as_secs());
        assert_eq!(secs("90"), Some(90));
        assert_eq!(secs("90s"), Some(90));
        assert_eq!(secs("30m"), Some(1800));
        assert_eq!(secs("1h30m"), Some(5400));
        assert_eq!(secs("2h5s"), Some(7205));
        for refused in ["", "0", "0s", "m", "1.5h", "30m1h", "5x", "10mm", "3m3m"] {
            assert!(parse_budget(refused).is_err(), "{refused}");
        }
    }
}
