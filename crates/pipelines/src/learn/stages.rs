// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that acquire a capability
// from a document or a tool and prove it with evidence. If your team needs
// expertise in agent infrastructure or small-model training loops, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The stages of `learn`, and the pipeline that orders them.
//!
//! Each stage is a function over [`LearnState`]: what the run was asked, and
//! what the stages so far left behind. The engine
//! ([`splinter_orchestrator::pipeline`]) runs them in order and records each;
//! a stage here only does its work and says what it did, and whether the run
//! can go on.

use std::time::Instant;

use splinter_core::model_ref::{ModelRef, POLICY_DEFAULT};
use splinter_core::role::Role;
use splinter_core::source::SourceId;
use splinter_knowledge::survey::survey;
use splinter_orchestrator::pipeline::{FnStage, Pipeline, StageEnd};
use splinter_orchestrator::runs::{to_json, Recorder};
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_store::experiences::SetId;
use splinter_store::tasks::TaskSetId;
use std::collections::BTreeMap;

use super::dataset_stage::dataset_stage;
use super::exam_stages::{checkpoint_stage, exam_set_stage, exam_stage, reserve_stage};
use super::report::{LearnReport, Planned, PolicyStage, PolicyUsed};
use super::{records_at_share, DEFAULT_REHEARSAL_SHARE, DEFAULT_VOICE_SHARE};
use crate::author::{author, kind_authors, AuthorRequest, Authored};
use crate::budget::StageDeadlines;
use crate::critique::{critique_set, CritiqueRequest, DEFAULT_RETRIES};
use crate::curriculum::frontier::{select_frontier, PassAtK};
use crate::curriculum::queue;
use crate::curriculum::quota::{select_training_set, Quotas};
use crate::curriculum::teacher::{teach, TeachRequest};
use crate::datasets::DEFAULT_MIN_STRENGTH;
use crate::exam_set::ExamSet;
use crate::plan::plan as make_plan;
use crate::raft::PassageShare;
use crate::rehearsal::{rehearse, RehearseRequest, REHEARSAL_SEED};
use crate::release::{release, ReleaseRequest};
use crate::solving::{solve_tasks, SamplingChoice, SolveRequest};
use crate::sources::{self, SourceTarget};
use crate::tasks::{generate, Generation};
use crate::train::{train, Rehearse, TrainRequest, Trainer, Tuning, DEFAULT_REPLAY_FRACTION};
use crate::variants::{generate_variants, VariantsRequest, DEFAULT_VARIANTS_PER_TASK};
use crate::verify::{kind_needs_judge, verify_set, Grading, Judge};
use splinter_core::annotation::Strength;

/// What one learn run was asked, resolved.
pub(super) struct Learn<'a> {
    pub(super) targets: &'a [SourceTarget],
    pub(super) kinds: &'a [String],
    /// The model that plans the kinds; `None` keeps `kinds`.
    pub(super) planner: Option<&'a ModelRef>,
    pub(super) goal: Option<&'a str>,
    /// Who the policy becomes, as the request names it.
    pub(super) persona: Option<&'a str>,
    /// The share of the training examples that is the writer's own text, as
    /// the request names it; `None` is the default for a run with a persona.
    pub(super) voice: Option<f64>,
    /// Whether the writer's passages are asked by a description of them.
    pub(super) describe_voice: bool,
    /// The share of the training draws that are the base's own answers, as
    /// the request names it; `None` is the default for a run with a persona.
    pub(super) rehearsal: Option<f64>,
    /// The share of training records given retrieved passages, when any.
    pub(super) passages: Option<PassageShare>,
    pub(super) deadline: Option<Instant>,
    pub(super) trainer: &'a dyn Trainer,
    pub(super) policy: ModelRef,
    pub(super) teacher: &'a ModelRef,
    pub(super) generator: &'a ModelRef,
    /// The model that judges the exam: another one than the policy when an
    /// assistant is configured.
    pub(super) judge: &'a ModelRef,
    /// The identity each role in use was given.
    pub(super) roles_used: BTreeMap<Role, String>,
    pub(super) no_release: bool,
    /// The step budget (`None`: [`super::auto_steps`] of the dataset), LoRA
    /// rank and tuning of the training.
    pub(super) steps: Option<u32>,
    pub(super) rank: u32,
    pub(super) tuning: Tuning,
    pub(super) quotas: Quotas,
    /// The exam reserved up front, as the request names it.
    pub(super) exam: super::ExamPlan,
    /// Whether the training step is chosen on the dev suite.
    pub(super) select_on_dev: bool,
}

/// The run so far: what it was asked, what its stages reported, and what
/// they hand the stages after them.
pub(super) struct LearnState<'a> {
    pub(super) learn: Learn<'a>,
    pub(super) report: LearnReport,
    /// When the run began: what the stage deadlines are counted from.
    pub(super) started: Instant,
    pub(super) stage_deadlines: StageDeadlines,
    /// The teacher answers every task and the student makes no attempt.
    distill: bool,
    /// pass@k's parameters; `None` keeps every task.
    frontier: Option<PassAtK>,
    pub(super) kinds: Vec<String>,
    pub(super) source_ids: Vec<SourceId>,
    task_set: Option<TaskSetId>,
    /// The student's graded attempts, then the frontier's.
    attempts: Option<SetId>,
    /// Attempts that failed: what is worth critiquing.
    failed: usize,
    /// The experience sets the training set is selected from.
    sets: Vec<SetId>,
    /// The tasks kept, whose variants are written.
    kept_tasks: Option<TaskSetId>,
    /// The frozen exam, when the run reserved one.
    pub(super) exam_set: Option<ExamSet>,
    /// The datasets trained on: the dialogues, then the writer's own text
    /// when the run has it.
    pub(super) datasets: Vec<String>,
    /// The rehearsal dataset, when the run built one.
    rehearsal: Option<String>,
    pub(super) records: usize,
    /// What the training reads: [`examples_in`] the datasets.
    pub(super) examples: usize,
    pub(super) candidate: Option<String>,
    /// The judge is named and measured, or no task needs one.
    judge_prepared: bool,
    /// The weakest decision the training set counts: judged verdicts too
    /// when a measured judge decides the kinds nothing else can.
    pub(super) min_strength: Strength,
}

impl<'a> LearnState<'a> {
    /// The system prompt the policy is trained and asked under: the persona
    /// the request names, else the one the plan found in the goal; `None`
    /// keeps the default.
    pub(super) fn system_prompt(&self) -> Option<String> {
        self.persona().map(splinter_core::prompt::persona_prompt)
    }

    /// Who the policy becomes: the persona the request names, else the one
    /// the plan found in the goal. The sources are what that person wrote,
    /// so the generator is told they are the author.
    pub(super) fn persona(&self) -> Option<&str> {
        let planned = self
            .report
            .plan
            .as_ref()
            .and_then(|planned| planned.plan.persona.as_deref());
        self.learn.persona.or(planned)
    }

    /// The share of the training examples that is the writer's own text:
    /// what the request names, else [`DEFAULT_VOICE_SHARE`] when the policy
    /// learns to think like a person and none otherwise.
    pub(super) fn voice_share(&self) -> f64 {
        self.learn.voice.unwrap_or(if self.persona().is_some() {
            DEFAULT_VOICE_SHARE
        } else {
            0.0
        })
    }

    /// The share of the training draws that are the base's own answers:
    /// what the request names, else [`DEFAULT_REHEARSAL_SHARE`] when the
    /// policy learns to think like a person and none otherwise.
    fn rehearsal_share(&self) -> f64 {
        self.learn.rehearsal.unwrap_or(if self.persona().is_some() {
            DEFAULT_REHEARSAL_SHARE
        } else {
            0.0
        })
    }

    /// Shares the budget between the stages as the voice share says: the
    /// plan may have named the persona, and with it the writer's text.
    fn share_budget(&mut self) {
        self.stage_deadlines =
            StageDeadlines::sharing(self.started, self.learn.deadline, self.voice_share());
    }

    /// The state of a run that has not begun.
    pub(super) fn new(learn: Learn<'a>, distill: bool, frontier: Option<PassAtK>) -> Self {
        let started = Instant::now();
        let kinds = learn.kinds.to_vec();
        let mut state = Self {
            learn,
            report: LearnReport::default(),
            started,
            stage_deadlines: StageDeadlines::of(started, None),
            distill,
            frontier: if distill { None } else { frontier },
            kinds,
            source_ids: Vec::new(),
            task_set: None,
            attempts: None,
            failed: 0,
            sets: Vec::new(),
            kept_tasks: None,
            exam_set: None,
            datasets: Vec::new(),
            rehearsal: None,
            records: 0,
            examples: 0,
            candidate: None,
            judge_prepared: false,
            min_strength: DEFAULT_MIN_STRENGTH,
        };
        state.share_budget();
        state
    }

    /// The run's end: the budget, or nothing to stop for.
    pub(super) fn deadline(&self) -> Option<Instant> {
        self.learn.deadline
    }

    /// Whether some kind of the run has the writer's own passage as its
    /// reference, which the author stage puts forward as the answer.
    fn authors(&self) -> bool {
        self.kinds.iter().any(|name| kind_authors(name))
    }

    fn uses_frontier(&self) -> bool {
        self.frontier.is_some() && self.report.solve.is_some()
    }

    fn task_set(&self) -> Result<&TaskSetId, OrchestratorError> {
        self.task_set
            .as_ref()
            .ok_or_else(|| OrchestratorError::Refused("a stage ran before the tasks stage".into()))
    }
}

/// Names the judge and measures it before the first stage that grades,
/// when some task is of a kind a judge grades: only a judge's verdict
/// establishes that an answer to such a task is right, so without one
/// nothing of that kind could be decided. The judge is another model than
/// the teacher, the generator and the policy, whose work it grades, and is
/// measured on controls made from the tasks' references
/// ([`Judge::calibrated`]).
fn prepare_judge(ctx: &Context, st: &mut LearnState<'_>) -> Result<(), OrchestratorError> {
    if st.judge_prepared {
        return Ok(());
    }
    let store = ctx.tasks();
    let tasks = store
        .get_set(st.task_set()?)?
        .members
        .iter()
        .map(|entry| store.get(&entry.task))
        .collect::<Result<Vec<_>, _>>()?;
    if tasks.iter().any(|task| kind_needs_judge(&task.task.kind)) {
        let judge = st.learn.judge;
        let roles = [
            ("teacher", st.learn.teacher),
            ("generator", st.learn.generator),
            ("policy", &st.learn.policy),
        ];
        if let Some((role, _)) = roles.iter().find(|(_, model)| *model == judge) {
            return Err(OrchestratorError::Refused(format!(
                "the judge {judge} is also the {role}: a model does not grade its own work, and \
                 these tasks are decided by a judge. Name another model for the judge"
            )));
        }
        Judge::calibrated(ctx, judge, &tasks)?;
        ctx.set_judge(judge.clone());
        // Its verdicts stand only above its measured precision, and for
        // these kinds nothing else can pass an answer: they are evidence.
        st.min_strength = Strength::Judged;
    }
    st.judge_prepared = true;
    Ok(())
}

/// The stages of `learn`, in order.
pub(super) fn pipeline<'a>() -> Pipeline<'a, LearnState<'a>> {
    Pipeline::new()
        .then(FnStage::new("policy", policy_stage).ignoring_budget())
        .then(FnStage::new("sources", sources_stage).ignoring_budget())
        .then(
            FnStage::new("plan", plan_stage)
                .when(|s| s.learn.planner.is_some())
                .ignoring_budget(),
        )
        .then(
            FnStage::new("reserve", reserve_stage)
                .when(|s| s.exam_families() > 0)
                .ignoring_budget(),
        )
        .then(
            FnStage::new("exam-set", exam_set_stage)
                .when(|s| s.exam_families() > 0)
                .ignoring_budget(),
        )
        .then(FnStage::new("tasks", tasks_stage).ignoring_budget())
        .then(FnStage::new("solve", solve_stage).when(|s| !s.distill))
        .then(
            FnStage::new("verify", verify_stage)
                .when(|s| !s.distill)
                .ignoring_budget(),
        )
        .then(FnStage::new("teach", teach_stage))
        .then(
            FnStage::new("author", author_stage)
                .when(LearnState::authors)
                .ignoring_budget(),
        )
        .then(
            FnStage::new("frontier", frontier_stage)
                .when(LearnState::uses_frontier)
                .ignoring_budget(),
        )
        .then(FnStage::new("variants", variants_stage))
        .then(
            FnStage::new("critique", critique_stage).when(|s| s.failed > 0 && s.attempts.is_some()),
        )
        .then(FnStage::new("select", select_stage).ignoring_budget())
        .then(FnStage::new("dataset", dataset_stage).ignoring_budget())
        .then(FnStage::new("rehearse", rehearse_stage).when(|s| s.rehearsal_share() > 0.0))
        .then(FnStage::new("train", train_stage))
        .then(
            FnStage::new("checkpoint", checkpoint_stage)
                .when(|s| s.learn.select_on_dev)
                .ignoring_budget(),
        )
        .then(
            FnStage::new("exam", exam_stage)
                .when(|s| s.report.select.is_some())
                .ignoring_budget(),
        )
        .then(
            FnStage::new("release", release_stage)
                .when(|s| !s.learn.no_release)
                .ignoring_budget(),
        )
}

pub(super) type Done = Result<StageEnd, OrchestratorError>;

/// A stage's summary as the run records it.
pub(super) fn to_value(
    summary: &impl serde::Serialize,
) -> Result<serde_json::Value, OrchestratorError> {
    to_json("the stage summary", summary)
}

fn policy_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    st.report.policy = PolicyUsed {
        alias: POLICY_DEFAULT.into(),
        release: ctx.policy_pin(POLICY_DEFAULT)?.map(|pin| pin.release),
    };
    st.report.roles = st.learn.roles_used.clone();
    Ok(StageEnd::done(to_value(&PolicyStage {
        policy: &st.report.policy,
        roles: &st.report.roles,
    })?))
}

fn sources_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    for target in st.learn.targets {
        let added = sources::add(ctx, target)?;
        st.source_ids.push(added.source.id.clone());
        st.report.sources.push(added.source);
    }
    Ok(StageEnd::done(to_value(&st.report.sources)?))
}

/// The planner chooses the kinds, and whether to distil, from what the
/// sources hold.
fn plan_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    let Some(planner) = st.learn.planner else {
        unreachable!("the plan stage applies when a planner is named")
    };
    let surveyed = survey(&ctx.sources(), &st.source_ids)?;
    let chosen = make_plan(ctx, &surveyed, st.learn.goal, planner, &run.cancel_token())?;
    let planned = Planned {
        survey: surveyed,
        plan: chosen,
    };
    st.distill = st.distill || planned.plan.distill;
    if st.distill {
        st.frontier = None;
    }
    st.kinds = planned.plan.kinds.clone();
    let summary = to_value(&planned)?;
    st.report.plan = Some(planned);
    st.share_budget();
    Ok(StageEnd::done(summary))
}

fn tasks_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    let queued = queue::pending(ctx)?;
    let sections: Vec<_> = queued
        .iter()
        .flat_map(|q| q.sections.iter().cloned())
        .collect();
    let generated = generate(
        ctx,
        &Generation {
            sources: &st.source_ids,
            sections: &sections,
            kinds: &st.kinds,
            generator: st.learn.generator,
            goal: st.learn.goal,
            author: st.persona(),
            deadline: st.stage_deadlines.tasks,
            cancel: run.cancel_token(),
        },
    )?;
    let summary = to_value(&generated)?;
    if generated.stopped.is_none() {
        queue::complete(ctx, &queued)?;
    }
    st.task_set = Some(generated.task_set.clone());
    let tasks = generated.tasks;
    st.report.tasks = Some(generated);
    Ok(if tasks == 0 {
        StageEnd::stop(
            summary,
            "no task was admitted, so there is nothing to solve",
        )
    } else {
        StageEnd::done(summary)
    })
}

/// The student's own attempts: skipped when distilling.
fn solve_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    prepare_judge(ctx, st)?;
    // The student attempts under the prompt it is trained under.
    let system = st.system_prompt();
    let attempted = solve_tasks(
        ctx,
        &SolveRequest {
            task_set: st.task_set()?,
            solver: &st.learn.policy,
            attempts: st.frontier.map_or(1, |p| p.k),
            sampling: st
                .frontier
                .map_or(SamplingChoice::Own, |p| p.sampling_choice()),
            teacher: false,
            system: system.as_deref(),
            deadline: st.stage_deadlines.attempts,
            cancel: run.cancel_token(),
        },
    )?;
    let summary = to_value(&attempted)?;
    st.attempts = Some(attempted.experience_set.clone());
    st.report.solve = Some(attempted);
    Ok(StageEnd::done(summary))
}

fn verify_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    let Some(graded) = st.attempts.as_ref() else {
        unreachable!("the verify stage follows the solve stage")
    };
    let verified = verify_set(ctx, graded, Grading::ActiveJudge, &run.cancel_token())?;
    let summary = to_value(&verified)?;
    st.failed = verified.failed;
    st.report.verify = Some(verified);
    Ok(StageEnd::done(summary))
}

fn teach_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    prepare_judge(ctx, st)?;
    // The teacher answers as the person the records open with: its answers
    // are what the student learns under that prompt.
    let system = st.system_prompt();
    let taught = teach(
        ctx,
        &TeachRequest {
            task_set: st.task_set()?,
            attempts: st.attempts.as_ref(),
            teacher: st.learn.teacher,
            system: system.as_deref(),
            deadline: st.stage_deadlines.teach,
            cancel: run.cancel_token(),
        },
    )?;
    let summary = to_value(&taught)?;
    if !st.uses_frontier() {
        // Every task is kept, and the teacher's verified answers with it.
        st.kept_tasks = st.task_set.clone();
        st.sets.push(taught.solve.experience_set.clone());
    }
    st.report.teach = Some(taught);
    Ok(StageEnd::done(summary))
}

/// The writer's own passages as answers, for the tasks whose reference is one,
/// each kept if a judge of fit says it is a natural reply to its message. The
/// messages were written by the generator, so it is not the judge.
fn author_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    // The stage adds to what the teacher teaches: without a judge of fit it
    // does nothing and says why, and the run goes on.
    let skipped = |why: String| Authored {
        tasks: 0,
        kept: 0,
        refused: 0,
        undecided: 0,
        experience_set: None,
        skipped: Some(why),
    };
    let authored = if st.learn.judge == st.learn.generator {
        skipped(format!(
            "the judge {} also wrote the messages whose fit it would judge: name another model \
             for the judge",
            st.learn.judge
        ))
    } else {
        match author(
            ctx,
            &AuthorRequest {
                task_set: st.task_set()?,
                judge: st.learn.judge,
            },
            &run.cancel_token(),
        ) {
            Ok(authored) => authored,
            Err(OrchestratorError::Refused(why)) => skipped(why),
            Err(e) => return Err(e),
        }
    };
    if let Some(set) = &authored.experience_set {
        // Selection keeps the writer's own words for a message a teacher's
        // paraphrase of them also answers.
        st.sets.push(set.clone());
        // A fit judge's verdict is the only one these experiences carry.
        st.min_strength = Strength::Judged;
    }
    let summary = to_value(&authored)?;
    st.report.authored = Some(authored);
    Ok(StageEnd::done(summary))
}

/// Keeps the tasks worth training on: those the student fails at least
/// sometimes and that have a verified answer.
fn frontier_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    let (Some(solved), Some(taught)) = (st.report.solve.as_ref(), st.report.teach.as_ref()) else {
        unreachable!("the frontier stage follows the solve and teach stages")
    };
    let kept = select_frontier(ctx, st.task_set()?, solved, taught)?;
    let summary = to_value(&kept)?;
    st.attempts = Some(kept.frontier_experience_set.clone());
    st.kept_tasks = Some(kept.frontier_task_set.clone());
    let d = kept.distribution;
    st.report.frontier = Some(kept);
    if d.kept() == 0 {
        return Ok(StageEnd::stop(
            summary,
            format!(
                "no task is worth training on: {} always solved, {} never solved with no \
                 verified answer, {} unmeasured of {}",
                d.always,
                d.never,
                d.unmeasured,
                d.tasks()
            ),
        ));
    }
    // Every task kept failed at least one attempt.
    st.failed = d.kept();
    Ok(StageEnd::done(summary))
}

fn variants_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    let Some(kept) = st.kept_tasks.as_ref() else {
        unreachable!("the variants stage follows the teach stage")
    };
    let varied = generate_variants(
        ctx,
        &VariantsRequest {
            task_set: kept,
            generator: st.learn.generator,
            per_task: DEFAULT_VARIANTS_PER_TASK,
            deadline: st.learn.deadline,
            cancel: run.cancel_token(),
        },
    )?;
    let summary = to_value(&varied)?;
    st.report.variants = Some(varied);
    if let Some(attempts) = &st.attempts {
        st.sets.insert(0, attempts.clone());
    }
    Ok(StageEnd::done(summary))
}

fn critique_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    let Some(attempts) = st.attempts.as_ref() else {
        unreachable!("the critique stage applies when there are attempts")
    };
    let critiqued = critique_set(
        ctx,
        &CritiqueRequest {
            set: attempts,
            critic: &st.learn.policy,
            solver: &st.learn.policy,
            retries: DEFAULT_RETRIES,
            deadline: st.learn.deadline,
            cancel: run.cancel_token(),
        },
    )?;
    let summary = to_value(&critiqued)?;
    st.sets.push(critiqued.revisions.clone());
    st.report.critique = Some(critiqued);
    Ok(StageEnd::done(summary))
}

fn select_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    let selected = select_training_set(ctx, &st.sets, st.min_strength, &st.learn.quotas)?;
    let summary = to_value(&selected)?;
    st.report.select = Some(selected);
    Ok(StageEnd::done(summary))
}

/// The base model's own answers to general tasks, as many records as make
/// the rehearsal share of the examples, so a rehearsed record is drawn
/// about as often as a dialogue answer; the train stage mixes them in at
/// that share and monitors on a share of them.
fn rehearse_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    let rehearsed = rehearse(
        ctx,
        &RehearseRequest {
            records: records_at_share(st.examples, st.rehearsal_share()),
            seed: REHEARSAL_SEED,
            deadline: st.learn.deadline,
            cancel: run.cancel_token(),
        },
    )?;
    let summary = to_value(&rehearsed)?;
    st.rehearsal = Some(rehearsed.dataset.dataset.to_string());
    st.report.rehearsal = Some(rehearsed);
    Ok(StageEnd::done(summary))
}

fn train_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    if st.datasets.is_empty() {
        unreachable!("the train stage follows the dataset stage")
    }
    // The budget, what a step averages and how the run is watched follow
    // the data unless the command named them; `train` resolves them.
    let candidate = train(
        ctx,
        &TrainRequest {
            datasets: st.datasets.clone(),
            rehearsal: st.rehearsal.clone().map(|dataset| Rehearse {
                dataset,
                share: st.rehearsal_share(),
            }),
            from: st.learn.policy.clone(),
            replay_fraction: DEFAULT_REPLAY_FRACTION,
            steps: st.learn.steps,
            rank: st.learn.rank,
            beta: None,
            tuning: Tuning {
                keep_evaluations: st.learn.tuning.keep_evaluations || st.learn.select_on_dev,
                ..st.learn.tuning
            },
        },
        st.learn.trainer,
        &run.cancel_token(),
    )?;
    let summary = to_value(&candidate)?;
    st.candidate = Some(candidate.candidate.clone());
    st.report.candidate = Some(candidate);
    Ok(StageEnd::done(summary))
}

fn release_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    let Some(id) = st.candidate.clone() else {
        unreachable!("the release stage follows the train stage")
    };
    match release(ctx, &ReleaseRequest::new(id), &run.cancel_token()) {
        Ok(released) => {
            let summary = to_value(&released)?;
            st.report.release = Some(released);
            Ok(StageEnd::done(summary))
        }
        // A candidate the gate may not judge (its champion moved on) is still
        // trained; the run says why it was not released.
        Err(e) if e.is_refusal() => Ok(StageEnd::halt(e.to_string())),
        Err(e) => Err(e),
    }
}
