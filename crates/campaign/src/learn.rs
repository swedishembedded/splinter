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
//! whatever the solver was shown. The candidate trained on it continues
//! that release, and goes through the release gate, which grades it
//! closed-book and releases it only if every check passes (`--no-release`
//! stops at the candidate).
//!
//! The pipeline stops early, and says why, when a stage leaves the next
//! nothing to work on or the budget is spent; `--dry-run` resolves the
//! plan and writes nothing.

use std::time::{Duration, Instant};

use serde::Serialize;
use splinter_lab::holdout::MIN_SAMPLES;
use splinter_record::experiences::SetId;
use splinter_record::source::SourceId;

use splinter_knowledge::survey::{survey, Survey};

use crate::context::Context;
use crate::critique::{critique_set, CritiqueRequest, Critiqued, DEFAULT_RETRIES};
use crate::curriculum::frontier::{select_frontier, Frontier, PassAtK};
use crate::curriculum::queue;
use crate::curriculum::quota::{select_training_set, Quotas, Selected};
use crate::curriculum::teacher::{teach, Taught, TeachRequest};
use crate::datasets::{build, BuildRequest, Built, ViewName, DEFAULT_MIN_STRENGTH};
use crate::error::CampaignError;
use crate::model_ref::{ModelRef, POLICY_DEFAULT};
use crate::plan::{plan as make_plan, Plan};
use crate::release::{release, ReleaseId, ReleaseRequest, Released};
use crate::runs::{record, Recorded, Recorder};
use crate::solving::{solve_tasks, SamplingChoice, SolveRequest, Solved};
use crate::sources::{self, SourceSummary, SourceTarget};
use crate::tasks::{check_kinds, generate, Generation, TasksGenerated, DEFAULT_LEARN_KINDS};
use crate::train::{
    train, Candidate, TrainRequest, Trainer, Tuning, DEFAULT_LEARNING_RATE, DEFAULT_LORA_RANK,
    DEFAULT_REPLAY_FRACTION, DEFAULT_STEPS,
};
use crate::variants::{
    generate_variants, VariantsGenerated, VariantsRequest, DEFAULT_VARIANTS_PER_TASK,
};
use crate::verify::{verify_set, Verified};

/// The stages, in order, as runs and reports name them.
pub const STAGES: [&str; 14] = [
    "policy", "sources", "plan", "tasks", "solve", "verify", "teach", "frontier", "variants",
    "critique", "select", "dataset", "train", "release",
];

/// How many passes a `learn` run makes over what it has learned when its
/// steps are not given.
pub const EPOCHS: u32 = 2;

/// The most steps a `learn` run takes when they are not given.
pub const MAX_AUTO_STEPS: u32 = 2000;

/// The steps of a run over `records` records when none are given: about
/// [`EPOCHS`] passes, never fewer than [`DEFAULT_STEPS`] and never more than
/// [`MAX_AUTO_STEPS`].
#[must_use]
pub fn auto_steps(records: usize) -> u32 {
    u32::try_from(records)
        .unwrap_or(u32::MAX)
        .saturating_mul(EPOCHS)
        .clamp(DEFAULT_STEPS, MAX_AUTO_STEPS)
}

/// One `learn`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LearnRequest {
    /// What to learn from, as the command line names it.
    pub sources: Vec<String>,
    /// What the learner is after.
    pub goal: Option<String>,
    /// Task kinds; empty is [`DEFAULT_LEARN_KINDS`] unless `plan` is set.
    pub kinds: Vec<String>,
    /// Let a planner model survey the sources and choose the task kinds,
    /// and whether to distil. Refused together with `kinds`: one decides.
    pub plan: bool,
    /// The model that plans; `None`: the generator, else the policy.
    pub planner: Option<ModelRef>,
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
    /// Optimizer steps of the training; [`DEFAULT_STEPS`] when `None`.
    pub steps: Option<u32>,
    /// LoRA rank of the adapter; [`DEFAULT_LORA_RANK`] when `None`.
    pub rank: Option<u32>,
    /// The base's precision and the learning rate of the training.
    pub tuning: Tuning,
    /// The model that teaches what the policy never solves; `None`: the
    /// policy itself.
    pub teacher: Option<ModelRef>,
    /// The model that writes the tasks; `None`: the policy itself.
    pub generator: Option<ModelRef>,
    /// k and the sampling of the frontier's pass@k.
    pub pass_at_k: PassAtK,
    /// The diversity quotas on the training set.
    pub quotas: Quotas,
}

/// What a dry run reports: the plan, and nothing written.
#[derive(Clone, Debug, Serialize)]
pub struct LearnPlan {
    /// The state root the run would write under.
    pub state: std::path::PathBuf,
    /// The sources, as they would be captured.
    pub sources: Vec<SourceTarget>,
    /// The task kinds.
    pub kinds: Vec<String>,
    /// The goal.
    pub goal: Option<String>,
    /// The budget, in seconds.
    pub budget_secs: Option<u64>,
    /// The model that solves, critiques and retries.
    pub policy: String,
    /// The model that writes the tasks.
    pub generator: String,
    /// The model that teaches what the policy never solves.
    pub teacher: String,
    /// The model that surveys the sources and plans, when a plan is asked
    /// for.
    pub planner: Option<String>,
    /// The stages, in order.
    pub stages: Vec<&'static str>,
    /// Always `true`: nothing was written.
    pub dry_run: bool,
}

/// What the plan stage found and chose.
#[derive(Clone, Debug, Serialize)]
pub struct Planned {
    /// What the sources hold.
    pub survey: Survey,
    /// How the planner chose to learn from them.
    pub plan: Plan,
}

/// The policy a run works with, as resolved when it started.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PolicyUsed {
    /// The alias.
    pub alias: String,
    /// The release it pointed at; `None` when it pointed at none, and the
    /// run works from the base.
    pub release: Option<ReleaseId>,
}

/// What a `learn` run reports, stage by stage.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LearnReport {
    /// The policy the whole run used.
    pub policy: PolicyUsed,
    /// The sources learned from.
    pub sources: Vec<SourceSummary>,
    /// The plan stage: the survey and the planner's choice; `None` when no
    /// plan was asked for.
    pub plan: Option<Planned>,
    /// The tasks stage.
    pub tasks: Option<TasksGenerated>,
    /// The solve stage.
    pub solve: Option<Solved>,
    /// The verify stage.
    pub verify: Option<Verified>,
    /// The teach stage: the teacher's graded solves of the tasks never
    /// solved.
    pub teach: Option<Taught>,
    /// The frontier stage: pass@k, and the tasks kept.
    pub frontier: Option<Frontier>,
    /// The variants stage: the tasks kept, asked in other words.
    pub variants: Option<VariantsGenerated>,
    /// The critique stage.
    pub critique: Option<Critiqued>,
    /// The select stage: the training set under the quotas.
    pub select: Option<Selected>,
    /// The dataset stage.
    pub dataset: Option<Built>,
    /// The candidate trained.
    pub candidate: Option<Candidate>,
    /// The release gate on it, and the release when it passed.
    pub release: Option<Released>,
    /// Why the pipeline stopped before training or releasing, if it did.
    pub stopped: Option<String>,
}

impl LearnReport {
    /// Whether the run got as far as it was asked: a candidate trained,
    /// and released unless the release was not asked for.
    #[must_use]
    pub fn finished(&self, release_asked: bool) -> bool {
        self.candidate.is_some()
            && (!release_asked || self.release.as_ref().is_some_and(|r| r.release.is_some()))
    }
}

/// What `learn` did: the plan of a dry run, or the recorded run.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum Learned {
    /// A dry run's plan.
    Planned(Box<LearnPlan>),
    /// A run and its report.
    Ran(Box<Recorded<LearnReport>>),
}

/// Runs `request`, training with `trainer`.
pub fn learn(
    ctx: &Context,
    request: &LearnRequest,
    trainer: &dyn Trainer,
) -> Result<Learned, CampaignError> {
    if request.plan && !request.kinds.is_empty() {
        return Err(CampaignError::Refused(
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
        return Err(CampaignError::Refused(
            "name at least one source to learn from".into(),
        ));
    }
    let measures_frontier = !request.no_frontier && !request.distill;
    if measures_frontier {
        request.pass_at_k.validate()?;
    }
    request.quotas.validate()?;
    let targets = request
        .sources
        .iter()
        .map(|s| SourceTarget::from_learn_arg(s))
        .collect::<Result<Vec<_>, _>>()?;
    let policy = ModelRef::policy_default();
    // The roles a command names win; then the configured assistant; then
    // the policy itself.
    let assistant = match &ctx.config().assistant_model {
        Some(text) => Some(text.parse::<ModelRef>().map_err(|e| {
            CampaignError::Refused(format!(
                "the assistant model {text:?} is not a model reference: {e}"
            ))
        })?),
        None => None,
    };
    let role = |named: &Option<ModelRef>| {
        named
            .clone()
            .or_else(|| assistant.clone())
            .unwrap_or_else(|| policy.clone())
    };
    let teacher = role(&request.teacher);
    let generator = role(&request.generator);
    let planner = request.plan.then(|| match (&request.planner, &assistant) {
        (Some(named), _) => named.clone(),
        (None, Some(assistant)) => assistant.clone(),
        (None, None) => generator.clone(),
    });
    if request.dry_run {
        return Ok(Learned::Planned(Box::new(LearnPlan {
            state: ctx.root().path().to_path_buf(),
            sources: targets,
            kinds,
            goal: request.goal.clone(),
            budget_secs: request.budget.map(|b| b.as_secs()),
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
                .filter(|stage| !(!measures_frontier && *stage == "frontier"))
                .filter(|stage| {
                    !(request.distill && ["solve", "verify", "critique"].contains(stage))
                })
                .collect(),
            dry_run: true,
        })));
    }
    let pipeline = Pipeline {
        ctx,
        targets: &targets,
        kinds: &kinds,
        planner: planner.as_ref(),
        goal: request.goal.as_deref(),
        deadline: request.budget.map(|b| Instant::now() + b),
        trainer,
        teacher: &teacher,
        generator: &generator,
        no_release: request.no_release,
        distill: request.distill,
        steps: request.steps,
        rank: request.rank.unwrap_or(DEFAULT_LORA_RANK),
        tuning: Tuning {
            bf16_base: request.tuning.bf16_base || ctx.config().bf16_base,
            learning_rate: request.tuning.learning_rate.or(Some(DEFAULT_LEARNING_RATE)),
        },
        frontier: measures_frontier.then_some(request.pass_at_k),
        quotas: request.quotas,
    };
    let recorded = record(ctx, "learn", request, |run| {
        let mut report = LearnReport::default();
        pipeline.run(run, &mut report)?;
        Ok(report)
    })?;
    Ok(Learned::Ran(Box::new(recorded)))
}

/// One learn run's inputs, resolved.
struct Pipeline<'a> {
    ctx: &'a Context,
    targets: &'a [SourceTarget],
    kinds: &'a [String],
    /// The model that plans the kinds; `None` keeps `kinds`.
    planner: Option<&'a ModelRef>,
    goal: Option<&'a str>,
    deadline: Option<Instant>,
    trainer: &'a dyn Trainer,
    teacher: &'a ModelRef,
    generator: &'a ModelRef,
    no_release: bool,
    /// The teacher answers every task and the student makes no attempt.
    distill: bool,
    /// Optimizer steps (`None`: [`auto_steps`] of the dataset), LoRA rank
    /// and tuning of the training.
    steps: Option<u32>,
    rank: u32,
    tuning: Tuning,
    /// pass@k's parameters; `None` keeps every task.
    frontier: Option<PassAtK>,
    quotas: Quotas,
}

impl Pipeline<'_> {
    /// The stages, filling `report` as they finish; returns early (with
    /// `report.stopped` set) when one leaves the next nothing to do.
    fn run(&self, run: &mut Recorder<'_>, report: &mut LearnReport) -> Result<(), CampaignError> {
        let Self {
            ctx,
            targets,
            kinds,
            planner,
            goal,
            deadline,
            trainer,
            teacher,
            generator,
            no_release,
            distill,
            steps,
            rank,
            tuning,
            frontier,
            quotas,
        } = *self;
        let policy = ModelRef::policy_default();
        report.policy = PolicyUsed {
            alias: POLICY_DEFAULT.into(),
            release: ctx.policy_pin(POLICY_DEFAULT)?.map(|pin| pin.release),
        };
        run.stage("policy", &report.policy)?;
        let spent = |stage: &str| {
            deadline
                .is_some_and(|d| Instant::now() >= d)
                .then(|| format!("the budget was spent before the {stage} stage"))
        };

        let mut source_ids: Vec<SourceId> = Vec::new();
        for target in targets {
            let added = sources::add(ctx, target)?;
            source_ids.push(added.source.id.clone());
            report.sources.push(added.source);
        }
        run.stage("sources", &report.sources)?;

        // The planner chooses the kinds, and whether to distil, from what
        // the sources hold; otherwise they are as the request named them.
        let planned_kinds: Vec<String>;
        let mut distill = distill;
        let kinds: &[String] = match planner {
            Some(planner) => {
                let surveyed = survey(&ctx.sources(), &source_ids)?;
                let chosen = make_plan(ctx, &surveyed, goal, planner, &run.cancel_token())?;
                let planned = Planned {
                    survey: surveyed,
                    plan: chosen,
                };
                run.stage("plan", &planned)?;
                distill = distill || planned.plan.distill;
                planned_kinds = planned.plan.kinds.clone();
                report.plan = Some(planned);
                &planned_kinds
            }
            None => kinds,
        };
        let frontier = if distill { None } else { frontier };

        run.check_cancelled()?;
        let queued = queue::pending(ctx)?;
        let sections: Vec<_> = queued
            .iter()
            .flat_map(|q| q.sections.iter().cloned())
            .collect();
        let generated = generate(
            ctx,
            &Generation {
                sources: &source_ids,
                sections: &sections,
                kinds,
                generator,
                goal,
                deadline,
                cancel: run.cancel_token(),
            },
        )?;
        run.stage("tasks", &generated)?;
        if generated.stopped.is_none() {
            queue::complete(ctx, &queued)?;
        }
        let task_set = generated.task_set.clone();
        let tasks = generated.tasks;
        report.tasks = Some(generated);
        if tasks == 0 {
            report.stopped = Some("no task was admitted, so there is nothing to solve".into());
            return Ok(());
        }

        if let Some(why) = spent("solve") {
            report.stopped = Some(why);
            return Ok(());
        }
        // The student's own attempts: skipped when distilling.
        let mut attempts: Option<SetId> = None;
        let mut solved: Option<Solved> = None;
        let mut failed = 0;
        if !distill {
            let attempted = solve_tasks(
                ctx,
                &SolveRequest {
                    task_set: &task_set,
                    solver: &policy,
                    attempts: frontier.map_or(1, |p| p.k),
                    sampling: frontier.map_or(SamplingChoice::Own, |p| p.sampling_choice()),
                    teacher: false,
                    deadline,
                    cancel: run.cancel_token(),
                },
            )?;
            run.stage("solve", &attempted)?;
            let graded = attempted.experience_set.clone();
            report.solve = Some(attempted.clone());
            solved = Some(attempted);

            run.check_cancelled()?;
            let verified = verify_set(ctx, &graded, None, &run.cancel_token())?;
            attempts = Some(graded);
            run.stage("verify", &verified)?;
            failed = verified.failed;
            report.verify = Some(verified);
        }

        if let Some(why) = spent("teach") {
            report.stopped = Some(why);
            return Ok(());
        }
        let taught = teach(
            ctx,
            &TeachRequest {
                task_set: &task_set,
                attempts: attempts.as_ref(),
                teacher,
                deadline,
                cancel: run.cancel_token(),
            },
        )?;
        run.stage("teach", &taught)?;
        let mut sets: Vec<SetId> = Vec::new();
        let kept_tasks;
        if let (Some(_), Some(solved)) = (frontier, solved.as_ref()) {
            let kept = select_frontier(ctx, &task_set, solved, &taught)?;
            run.stage("frontier", &kept)?;
            attempts = Some(kept.frontier_experience_set.clone());
            kept_tasks = kept.frontier_task_set.clone();
            let d = kept.distribution;
            report.teach = Some(taught);
            report.frontier = Some(kept);
            if d.kept() == 0 {
                report.stopped = Some(format!(
                    "no task is worth training on: {} always solved, {} never solved with no \
                     verified answer, {} unmeasured of {}",
                    d.always,
                    d.never,
                    d.unmeasured,
                    d.tasks()
                ));
                return Ok(());
            }
            // Every task kept failed at least one attempt.
            failed = d.kept();
        } else {
            // Every task is kept, and the teacher's verified answers with it.
            kept_tasks = task_set.clone();
            sets.push(taught.solve.experience_set.clone());
            report.teach = Some(taught);
        }

        if let Some(why) = spent("variants") {
            report.stopped = Some(why);
            return Ok(());
        }
        let varied = generate_variants(
            ctx,
            &VariantsRequest {
                task_set: &kept_tasks,
                generator,
                per_task: DEFAULT_VARIANTS_PER_TASK,
                deadline,
                cancel: run.cancel_token(),
            },
        )?;
        run.stage("variants", &varied)?;
        report.variants = Some(varied);

        if let Some(attempts) = &attempts {
            sets.insert(0, attempts.clone());
        }
        if let (true, Some(attempts)) = (failed > 0, attempts.as_ref()) {
            if let Some(why) = spent("critique") {
                report.stopped = Some(why);
                return Ok(());
            }
            let critiqued = critique_set(
                ctx,
                &CritiqueRequest {
                    set: attempts,
                    critic: &policy,
                    solver: &policy,
                    retries: DEFAULT_RETRIES,
                    deadline,
                    cancel: run.cancel_token(),
                },
            )?;
            run.stage("critique", &critiqued)?;
            sets.push(critiqued.revisions.clone());
            report.critique = Some(critiqued);
        }

        run.check_cancelled()?;
        let selected = select_training_set(ctx, &sets, DEFAULT_MIN_STRENGTH, &quotas)?;
        run.stage("select", &selected)?;
        let training_set = selected.experience_set.clone();
        report.select = Some(selected);

        run.check_cancelled()?;
        let built = match build(
            ctx,
            &BuildRequest {
                sets: vec![training_set],
                view: ViewName::SftFinal,
                strip: None,
                min_strength: None,
                export_only: false,
            },
        ) {
            Ok(built) => built,
            Err(CampaignError::View(splinter_views::ViewError::Empty)) => {
                report.stopped = Some(
                    "no experience passed verification, so there is nothing to train on".into(),
                );
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        run.stage("dataset", &built)?;
        let dataset = built.dataset.to_string();
        let records = built.records;
        report.dataset = Some(built);
        if records < MIN_SAMPLES {
            report.stopped = Some(format!(
                "{records} record(s) passed; training holds records out for scoring and needs \
                 at least {MIN_SAMPLES}"
            ));
            return Ok(());
        }

        if let Some(why) = spent("train") {
            report.stopped = Some(why);
            return Ok(());
        }
        let candidate = train(
            ctx,
            &TrainRequest {
                datasets: vec![dataset],
                from: policy,
                replay_fraction: DEFAULT_REPLAY_FRACTION,
                steps: steps.unwrap_or_else(|| auto_steps(records)),
                rank,
                beta: None,
                tuning,
            },
            trainer,
            &run.cancel_token(),
        )?;
        run.stage("train", &candidate)?;
        let id = candidate.candidate.clone();
        report.candidate = Some(candidate);
        if no_release {
            return Ok(());
        }
        run.check_cancelled()?;
        match release(ctx, &ReleaseRequest::new(id), &run.cancel_token()) {
            Ok(released) => {
                run.stage("release", &released)?;
                report.release = Some(released);
            }
            // A candidate the gate may not judge (its champion moved on) is
            // still trained; the run says why it was not released.
            Err(e) if e.is_refusal() => report.stopped = Some(e.to_string()),
            Err(e) => return Err(e),
        }
        Ok(())
    }
}

/// A duration as `learn --budget` takes it: whole units of `s`, `m` and
/// `h`, alone or combined largest first (`30m`, `1h30m`), or bare
/// seconds.
pub fn parse_budget(text: &str) -> Result<Duration, CampaignError> {
    let refuse = |why: &str| {
        CampaignError::Refused(format!(
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
