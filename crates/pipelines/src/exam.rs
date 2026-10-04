// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out examinations of what a model
// learned from a person's writing, for its clients. If your team needs
// expertise in measuring whether a model took on a writer's way of thinking,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The exam: what a candidate learned that its base did not, on tasks neither
//! was trained on, measured where no verifier can decide by matching text.
//!
//! The release gate grades by code alone, which is right for facts and
//! quotations and cannot tell a reply that gives a writer's advice in its own
//! words from one that does not. The exam puts the same held-out tasks to the
//! base and to the candidate closed-book and asks a judge whether each answer
//! gives what the task's reference says, and asks code whether it states a
//! number or name the task's source does not hold ([`GroundingVerifier`]).
//! The judged results are compared with a paired sign test.
//!
//! The judge is trusted only as far as it can be measured. Before it grades an
//! arm it is calibrated on controls made from the tasks' own references, which
//! no model wrote: each task's reference answering that task (right) and the
//! next task's reference answering it (wrong). No control is a model's answer,
//! so the judge is never asked to grade its own model's work, whichever model
//! it is. A
//! judge whose precision on either falls below
//! [`DEFAULT_MIN_PRECISION`] abstains on every answer, and the report makes no
//! claim from it. Answers are never stored as experiences: nothing an exam
//! produces can reach a training set.

use serde::Serialize;
use splinter_agent::solve::with_system_addendum;
use splinter_agent::CancelToken;
use splinter_core::annotation::Outcome;
use splinter_core::experience::{Experience, PrivilegedKind, Provenance, Task};
use splinter_eval::significance::SignTest;
use splinter_eval::verifiers::calibration::{
    measure, CalibratedJudge, Calibration, Measurement, DEFAULT_MIN_PRECISION,
};
use splinter_eval::verifiers::Verifier;
use splinter_model::stats::sign_test;

use crate::release::arm;
use crate::release::probe::{answer_prompted, greedy, held_out, trained_tasks};
use crate::retrieval::Retrieval;
use crate::train::load_candidate;
use crate::verify::{grounding_verifier, judge_verifier};
use splinter_core::model_ref::ModelRef;
use splinter_core::role::{Role, RoleOverrides};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::roles::assignments;

/// One arm's verdicts, a task each: `None` where nothing decided.
type Decided = Vec<Option<bool>>;

/// What an exam is given.
pub struct ExamRequest<'a> {
    /// The held-out tasks, put closed-book to both arms.
    pub tasks: &'a [Task],
    /// Tasks with a reference, from which the judge's controls are made: at
    /// least two.
    pub controls: &'a [Task],
    /// The model the candidate is measured against.
    pub base: &'a ModelRef,
    /// The model under examination.
    pub candidate: &'a ModelRef,
    /// The judge.
    pub judge: &'a ModelRef,
    /// The goal the base is also asked under, as a prompt-only baseline: the
    /// training is worth what it adds beyond telling the base what the goal
    /// is. `None` adds no such arm.
    pub prompted: Option<&'a str>,
    /// Passages to retrieve for each task and show before it, for a further
    /// arm: the candidate with retrieval, against the candidate alone. `None`
    /// adds no such arm.
    pub retrieval: Option<&'a Retrieval<'a>>,
    /// Stops the exam.
    pub cancel: CancelToken,
}

/// How far the judge could be trusted.
#[derive(Clone, Debug, Serialize)]
pub struct JudgeTrust {
    /// The judge's identity.
    pub judge: String,
    /// Whether its measured precision on both passes and fails reached the
    /// threshold, so that its verdicts count.
    pub trusted: bool,
    /// Controls it was measured on.
    pub controls: usize,
    /// The precision threshold.
    pub min_precision: f64,
    /// The measurement.
    pub calibration: Calibration,
    /// The controls it did not judge as labelled: what shows where it goes
    /// wrong when it is not trusted.
    pub misjudged: Vec<Misjudged>,
}

/// A control the judge did not judge as labelled.
#[derive(Clone, Debug, Serialize)]
pub struct Misjudged {
    /// What the control should have got: `pass` or `fail`.
    pub label: String,
    /// What the judge gave it: `pass`, `fail` or `abstain`.
    pub judged: String,
    /// The task's instruction, cut at [`SHOWN_CHARS`].
    pub instruction: String,
    /// The answer it was judged on, cut at [`SHOWN_CHARS`].
    pub answer: String,
    /// The judge's reason, when it gave one.
    pub reason: String,
}

/// The most characters of an instruction or an answer a [`Misjudged`] shows.
const SHOWN_CHARS: usize = 400;

fn outcome_word(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Pass => "pass",
        Outcome::Fail => "fail",
        Outcome::Abstain => "abstain",
    }
}

/// The controls `measurements` (one per control, in order) shows the judge
/// did not judge as labelled.
fn misjudged(
    labelled: &[(Task, Experience, Outcome)],
    measurements: &[Measurement],
) -> Vec<Misjudged> {
    let cut = |text: &str| text.chars().take(SHOWN_CHARS).collect::<String>();
    labelled
        .iter()
        .zip(measurements)
        .filter(|(_, m)| m.judged != m.label)
        .map(|((task, exp, _), m)| Misjudged {
            label: outcome_word(m.label).into(),
            judged: outcome_word(m.judged).into(),
            instruction: cut(&task.instruction),
            answer: cut(exp.final_output.as_deref().unwrap_or_default()),
            reason: m.evidence["reason"].as_str().unwrap_or_default().into(),
        })
        .collect()
}

/// One arm's results.
#[derive(Clone, Debug, Serialize)]
pub struct ArmResult {
    /// The model's identity.
    pub model: String,
    /// Tasks the model gave no answer to (a reasoning model that spent its
    /// whole reply budget thinking). Each counts as not done in `judged`.
    pub unanswered: usize,
    /// Answers decided: the judge's pass or fail when it is trusted, and a
    /// task without an answer, which code decides is not done.
    pub judged: usize,
    /// Of those, the ones it said give what the reference says.
    pub judged_right: usize,
    /// Answers the grounding check decided.
    pub checked: usize,
    /// Of those, the ones that state a number, name or quotation the source
    /// does not hold.
    pub invented: usize,
}

/// The candidate asked with retrieved passages shown before each task.
#[derive(Clone, Debug, Serialize)]
pub struct RetrievalResult {
    /// How the candidate did with them.
    pub arm: ArmResult,
    /// Tasks for which a retrieved passage overlaps the evidence the task was
    /// written from.
    pub hits: usize,
    /// Tasks asked.
    pub tasks: usize,
    /// Passages shown with each.
    pub passages: usize,
}

/// What the exam reports.
#[derive(Clone, Debug, Serialize)]
pub struct Examined {
    /// Tasks put to each arm.
    pub tasks: usize,
    /// The judge and what it was measured at.
    pub judge: JudgeTrust,
    /// The base.
    pub base: ArmResult,
    /// The base asked under the goal; `None` when no goal was given.
    pub prompted: Option<ArmResult>,
    /// The candidate.
    pub candidate: ArmResult,
    /// The judged results compared over the tasks both were judged on; `None`
    /// when the judge is not trusted or no task was judged for both.
    pub paired: Option<SignTest>,
    /// The candidate against the prompted base, by the same rule; `None`
    /// when there is no prompted arm.
    pub paired_vs_prompted: Option<SignTest>,
    /// The candidate with retrieval; `None` when none was asked for.
    pub retrieval: Option<RetrievalResult>,
    /// The candidate with retrieval against the candidate alone, by the same
    /// rule; `None` when there is no retrieval arm.
    pub paired_retrieval: Option<SignTest>,
}

/// An experience of `task` answered with `answer`, as the exam's controls
/// are: by no model.
fn answered(ctx: &Context, task: &Task, answer: &str) -> Result<Experience, OrchestratorError> {
    Ok(Experience::answered_without_a_run(
        task.clone(),
        answer,
        Provenance::new(CONTROLS_SOURCE, ctx.clock()),
    )?)
}

/// What a control's provenance names as its solver: not a model.
const CONTROLS_SOURCE: &str = "splinter/exam-controls";

/// The reference of `task`, the answer it is controlled with.
fn reference(task: &Task) -> Option<&str> {
    task.privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .map(|p| p.content.as_str())
}

/// The judge's controls: each task's reference right for it, and the next
/// task's reference wrong for it. A task with no reference, or the same one
/// as its neighbour's, makes no control.
fn controls(
    ctx: &Context,
    tasks: &[Task],
) -> Result<Vec<(Task, Experience, Outcome)>, OrchestratorError> {
    let mut labelled = Vec::with_capacity(tasks.len() * 2);
    for (n, task) in tasks.iter().enumerate() {
        let other = &tasks[(n + 1) % tasks.len()];
        let (Some(right), Some(wrong)) = (reference(task), reference(other)) else {
            continue;
        };
        if right == wrong {
            continue;
        }
        labelled.push((task.clone(), answered(ctx, task, right)?, Outcome::Pass));
        labelled.push((task.clone(), answered(ctx, task, wrong)?, Outcome::Fail));
    }
    Ok(labelled)
}

/// Runs `request`; see the module documentation.
pub fn exam(ctx: &Context, request: &ExamRequest<'_>) -> Result<Examined, OrchestratorError> {
    if request.controls.len() < 2 {
        return Err(OrchestratorError::Refused(
            "an exam calibrates its judge on at least two tasks with a reference".into(),
        ));
    }
    if request.base == request.candidate {
        return Err(OrchestratorError::Refused(
            "the exam would compare the model with itself: the base and the candidate are one"
                .into(),
        ));
    }
    // What retrieval finds is found before any model answers: it needs the
    // embedder, and the arms need the device.
    let retrieved = request
        .retrieval
        .map(|retrieval| {
            request
                .tasks
                .iter()
                .map(|task| retrieval.find(&task.instruction))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    // A reader that judged passages gives the device back before any arm.
    if request.retrieval.is_some_and(|r| r.rerank.is_some()) {
        ctx.release_bases();
    }
    let retrieval_prompts: Option<Vec<String>> = retrieved.as_ref().map(|found| {
        found
            .iter()
            .zip(request.tasks)
            .map(|(found, task)| found.prompt(&task.instruction))
            .collect()
    });
    let answers = |reference: &ModelRef,
                   goal: Option<&str>,
                   prompts: Option<&[String]>|
     -> Result<(String, Vec<Experience>), OrchestratorError> {
        let mut model = greedy(ctx, reference)?;
        if let Some(goal) = goal {
            model.provider =
                with_system_addendum(model.provider.clone(), &format!("Your purpose: {goal}"));
            model.identity = format!("{}+prompted", model.identity);
        }
        if prompts.is_some() {
            model.identity = format!("{}+retrieval", model.identity);
        }
        let given = request
            .tasks
            .iter()
            .enumerate()
            .map(|(n, task)| {
                let prompt = prompts.map(|p| p[n].as_str());
                answer_prompted(ctx, &model, task, prompt, &request.cancel)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((model.identity, given))
    };
    // Both arms answer before the judge is loaded, so the device swaps models
    // as few times as it can.
    let (base_model, base_answers) = answers(request.base, None, None)?;
    let prompted_answers = request
        .prompted
        .map(|goal| answers(request.base, Some(goal), None))
        .transpose()?;
    let (candidate_model, candidate_answers) = answers(request.candidate, None, None)?;
    let retrieval_answers = retrieval_prompts
        .as_deref()
        .map(|prompts| answers(request.candidate, None, Some(prompts)))
        .transpose()?;

    // The judge is a different model from the arms' and needs the device for
    // itself: two resident bases at once do not fit a card the size of the
    // models'.
    ctx.release_bases();
    let judge_model = ctx.model(request.judge)?;
    let judge = judge_verifier(ctx, &judge_model);
    let labelled = controls(ctx, request.controls)?;
    let (calibration, measurements) = measure(&judge, &labelled)?;
    let precise = |p: Option<f64>| p.is_some_and(|p| p >= DEFAULT_MIN_PRECISION);
    let trusted = precise(calibration.precision_pass) && precise(calibration.precision_fail);
    let judge = CalibratedJudge::new(judge, calibration.clone(), DEFAULT_MIN_PRECISION)?;
    let grounding = grounding_verifier(ctx);

    let grade = |given: &[Experience]| -> Result<(Decided, Decided), OrchestratorError> {
        let mut judged = Vec::with_capacity(given.len());
        let mut grounded = Vec::with_capacity(given.len());
        for (task, exp) in request.tasks.iter().zip(given) {
            if request.cancel.is_cancelled() {
                return Err(OrchestratorError::Cancelled);
            }
            let decided = |finding: splinter_eval::verifiers::Finding| match finding.outcome {
                Outcome::Pass => Some(true),
                Outcome::Fail => Some(false),
                Outcome::Abstain => None,
            };
            if exp.final_output.is_none() {
                // Nothing was said: not done, and nothing to hold to the source.
                judged.push(Some(false));
                grounded.push(None);
                continue;
            }
            judged.push(decided(judge.verify(task, exp)?));
            grounded.push(decided(grounding.verify(task, exp)?));
        }
        Ok((judged, grounded))
    };
    let (base_judged, base_grounded) = grade(&base_answers)?;
    let (candidate_judged, candidate_grounded) = grade(&candidate_answers)?;
    let prompted = prompted_answers
        .map(|(model, given)| {
            grade(&given).map(|(judged, grounded)| (model, given, judged, grounded))
        })
        .transpose()?;
    let with_retrieval = retrieval_answers
        .map(|(model, given)| {
            grade(&given).map(|(judged, grounded)| (model, given, judged, grounded))
        })
        .transpose()?;

    let arm = |model: String,
               given: &[Experience],
               judged: &[Option<bool>],
               grounded: &[Option<bool>]| ArmResult {
        model,
        unanswered: given.iter().filter(|e| e.final_output.is_none()).count(),
        judged: judged.iter().flatten().count(),
        judged_right: judged.iter().flatten().filter(|right| **right).count(),
        checked: grounded.iter().flatten().count(),
        invented: grounded.iter().flatten().filter(|held| !**held).count(),
    };
    let pairs_against = |other: &[Option<bool>]| -> Vec<(bool, bool)> {
        candidate_judged
            .iter()
            .zip(other)
            .filter_map(|(c, o)| Some(((*c)?, (*o)?)))
            .collect()
    };
    let pairs = pairs_against(&base_judged);
    let pairs_vs_prompted = prompted
        .as_ref()
        .map(|(_, _, judged, _)| pairs_against(judged))
        .unwrap_or_default();
    // With retrieval against without: a win is a task the passages made right.
    let pairs_retrieval: Vec<(bool, bool)> = with_retrieval
        .as_ref()
        .map(|(_, _, judged, _)| {
            judged
                .iter()
                .zip(&candidate_judged)
                .filter_map(|(w, c)| Some(((*w)?, (*c)?)))
                .collect()
        })
        .unwrap_or_default();
    let retrieval = with_retrieval
        .as_ref()
        .zip(retrieved.as_ref())
        .zip(request.retrieval)
        .map(|((arm_of, found), retrieval)| RetrievalResult {
            arm: arm(arm_of.0.clone(), &arm_of.1, &arm_of.2, &arm_of.3),
            hits: found
                .iter()
                .zip(request.tasks)
                .filter(|(found, task)| found.finds_the_evidence_of(task))
                .count(),
            tasks: request.tasks.len(),
            passages: retrieval.passages,
        });
    Ok(Examined {
        tasks: request.tasks.len(),
        judge: JudgeTrust {
            judge: judge_model.identity,
            trusted,
            controls: request.controls.len() * 2,
            min_precision: DEFAULT_MIN_PRECISION,
            calibration,
            misjudged: misjudged(&labelled, &measurements),
        },
        base: arm(base_model, &base_answers, &base_judged, &base_grounded),
        prompted: prompted
            .as_ref()
            .map(|(model, given, judged, grounded)| arm(model.clone(), given, judged, grounded)),
        candidate: arm(
            candidate_model,
            &candidate_answers,
            &candidate_judged,
            &candidate_grounded,
        ),
        paired: (trusted && !pairs.is_empty()).then(|| sign_test(&pairs)),
        paired_vs_prompted: (trusted && !pairs_vs_prompted.is_empty())
            .then(|| sign_test(&pairs_vs_prompted)),
        retrieval,
        paired_retrieval: (trusted && !pairs_retrieval.is_empty())
            .then(|| sign_test(&pairs_retrieval)),
    })
}

/// The most held-out tasks an exam of a candidate puts to each arm: chosen
/// evenly over the held-out set, so a long set is sampled and not truncated.
pub const MAX_EXAM_TASKS: usize = 48;

/// The most tasks the judge's controls are made from.
const MAX_CONTROLS: usize = 24;

/// What the exam of a candidate came to.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Exam {
    /// The exam ran.
    Ran(Box<Examined>),
    /// It could not run, and why.
    NotRun(String),
}

/// `items` thinned to at most `most`, evenly from first to last.
fn spaced<T: Clone>(items: &[T], most: usize) -> Vec<T> {
    if items.len() <= most || most < 2 {
        return items.iter().take(most).cloned().collect();
    }
    (0..most)
        .map(|i| items[i * (items.len() - 1) / (most - 1)].clone())
        .collect()
}

/// The `exam` command: the stored candidate `candidate` examined against the
/// model it continues (its parent release, else the base), by `judge` (else the judge role's model); see
/// [`examine_candidate`].
pub fn examine(
    ctx: &Context,
    candidate: &str,
    judge: Option<&ModelRef>,
    prompted: Option<&str>,
    retrieval: Option<&Retrieval<'_>>,
    cancel: &CancelToken,
) -> Result<Exam, OrchestratorError> {
    let judge = match judge {
        Some(named) => named.clone(),
        None => assignments(ctx.config(), &RoleOverrides::default())?
            .get(Role::Judge)
            .clone(),
    };
    // The candidate is measured against what it continues - the release it
    // was trained from, or the base alone - not against what the alias
    // points at now, which is the candidate itself once it is released.
    let trained = load_candidate(ctx, candidate)?;
    let continued = trained
        .parent
        .as_ref()
        .map(|id| ctx.releases().get(id))
        .transpose()?;
    let base = arm(
        ctx.config(),
        continued.as_ref().map(|r| r.adapter.as_path()),
    );
    examine_candidate(ctx, candidate, &base, &judge, prompted, retrieval, cancel)
}

/// Examines the trained candidate `candidate` against `base` on its held-out
/// tasks, the judge calibrated on controls made from the references of its
/// tasks. An exam that has nothing to run on says so
/// instead of reporting an empty result.
pub fn examine_candidate(
    ctx: &Context,
    candidate: &str,
    base: &ModelRef,
    judge: &ModelRef,
    prompted: Option<&str>,
    retrieval: Option<&Retrieval<'_>>,
    cancel: &CancelToken,
) -> Result<Exam, OrchestratorError> {
    let trained = load_candidate(ctx, candidate)?;
    let suite = held_out(ctx, "exam", &trained.datasets)?;
    if suite.tasks.is_empty() {
        return Ok(Exam::NotRun("no held-out task to put to the models".into()));
    }
    // Controls use references only, never an arm's answer, so the held-out
    // tasks serve as well as the trained-on ones.
    let mut controls: Vec<Task> = trained_tasks(ctx, &trained.datasets)?
        .into_iter()
        .chain(suite.tasks.iter().cloned())
        .filter(|task| reference(task).is_some())
        .collect();
    controls.dedup_by(|a, b| a.task.id == b.task.id);
    controls.sort_by(|a, b| a.task.kind.cmp(&b.task.kind));
    let controls = spaced(&controls, MAX_CONTROLS);
    if controls.len() < 2 {
        return Ok(Exam::NotRun(
            "fewer than two tasks with a reference to calibrate the judge on".into(),
        ));
    }
    let tasks = spaced(&suite.tasks, MAX_EXAM_TASKS);
    let candidate = arm(ctx.config(), Some(&trained.adapter));
    let examined = exam(
        ctx,
        &ExamRequest {
            tasks: &tasks,
            controls: &controls,
            base,
            candidate: &candidate,
            judge,
            prompted,
            retrieval,
            cancel: cancel.clone(),
        },
    )?;
    Ok(Exam::Ran(Box::new(examined)))
}
