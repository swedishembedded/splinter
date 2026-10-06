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
use splinter_core::experience::{Experience, Task};
use splinter_eval::paired::{by_cluster, PairedOutcome};
use splinter_eval::significance::SignTest;
use splinter_eval::verifiers::calibration::{
    measure, CalibratedJudge, Calibration, Measurement, DEFAULT_MIN_PRECISION,
};
use splinter_eval::verifiers::Verifier;
use splinter_model::stats::{bootstrap_interval, sign_test, Interval};

use crate::grouping::task_clusters;
use crate::judging::{controls, reference, spaced};
use crate::release::arm;
use crate::release::probe::{answer_prompted, greedy, held_out, trained_tasks};
use crate::retrieval::Retrieval;
use crate::train::load_candidate;
use crate::verify::{grounding_verifier, judge_verifier};
use splinter_core::model_ref::ModelRef;
use splinter_core::role::{Role, RoleOverrides};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::releases::StoredRelease;
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
    /// The system prompt the model the candidate continues is asked under:
    /// the one it was trained under; `None` is the default.
    pub base_system: Option<&'a str>,
    /// The system prompt the candidate was trained under, and so is asked
    /// under; `None` is the default. The prompted base is asked under it
    /// too: the one thing that tells it from the base is the prompt.
    pub candidate_system: Option<&'a str>,
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
pub(crate) fn misjudged(
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
    /// Each task: what retrieval found and how the candidate fared.
    pub by_task: Vec<RetrievalTask>,
}

/// One task of the retrieval arm.
#[derive(Clone, Debug, Serialize)]
pub struct RetrievalTask {
    /// The task.
    pub task: String,
    /// Whether a retrieved passage overlaps the evidence the task was
    /// written from.
    pub evidence_found: bool,
    /// The candidate's judged verdict without the passages; `None` where the
    /// judge did not decide.
    pub alone: Option<bool>,
    /// And with them.
    pub with_passages: Option<bool>,
}

/// What the exam reports.
#[derive(Clone, Debug, Serialize)]
pub struct Examined {
    /// Tasks put to each arm.
    pub tasks: usize,
    /// The families of source text they come from: the units of evidence the
    /// comparisons count.
    pub families: usize,
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
    /// A bootstrap interval of the candidate's gain over the base per family
    /// (a family won is 1, lost -1, tied 0), over the families both were
    /// judged on: what a result on a few families has to be read by. `None`
    /// when `paired` is, or there are too few families to resample.
    pub paired_interval: Option<Interval>,
    /// The candidate against the prompted base, by the same rule; `None`
    /// when there is no prompted arm.
    pub paired_vs_prompted: Option<SignTest>,
    /// The candidate with retrieval; `None` when none was asked for.
    pub retrieval: Option<RetrievalResult>,
    /// The candidate with retrieval against the candidate alone, by the same
    /// rule; `None` when there is no retrieval arm.
    pub paired_retrieval: Option<SignTest>,
    /// What the candidate's training curve warns of about the adapter
    /// examined (`Candidate::warnings`); empty for an exam of models that
    /// are not a stored candidate.
    pub training_warnings: Vec<String>,
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
                   system: Option<&str>,
                   goal: Option<&str>,
                   prompts: Option<&[String]>|
     -> Result<(String, Vec<Experience>), OrchestratorError> {
        let mut model = greedy(ctx, reference)?;
        if let Some(system) = system {
            model = model.with_system(system);
        }
        if let Some(goal) = goal {
            // Where the candidate has a prompt of its own the prompted base
            // is asked under it and needs no goal beside it; else the goal
            // is added to the default prompt.
            if system.is_none() {
                model.provider =
                    with_system_addendum(model.provider.clone(), &format!("Your purpose: {goal}"));
            }
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
    let (base_model, base_answers) = answers(request.base, request.base_system, None, None)?;
    let prompted_answers = request
        .prompted
        .map(|goal| answers(request.base, request.candidate_system, Some(goal), None))
        .transpose()?;
    let (candidate_model, candidate_answers) =
        answers(request.candidate, request.candidate_system, None, None)?;
    let retrieval_answers = retrieval_prompts
        .as_deref()
        .map(|prompts| {
            answers(
                request.candidate,
                request.candidate_system,
                None,
                Some(prompts),
            )
        })
        .transpose()?;

    // The judge is a different model from the arms' and needs the device for
    // itself: two resident bases at once do not fit a card the size of the
    // models'.
    ctx.release_bases();
    let judge_model = crate::verify::judge_model(ctx, request.judge)?;
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
    // Tasks about one family of sources are one unit of evidence: the sign
    // test counts a family once ([`by_cluster`]).
    let clusters = task_clusters(ctx, request.tasks)?;
    let pairs_of = |better: &[Option<bool>], other: &[Option<bool>]| -> Vec<(bool, bool)> {
        let outcomes: Vec<PairedOutcome> = better
            .iter()
            .zip(other)
            .zip(&clusters)
            .map(|((better, other), cluster)| PairedOutcome {
                item: String::new(),
                candidate: *better,
                baseline: *other,
                cluster: cluster.clone(),
            })
            .collect();
        by_cluster(&outcomes)
    };
    let pairs = pairs_of(&candidate_judged, &base_judged);
    let pairs_vs_prompted = prompted
        .as_ref()
        .map(|(_, _, judged, _)| pairs_of(&candidate_judged, judged))
        .unwrap_or_default();
    // With retrieval against without: a win is a task the passages made right.
    let pairs_retrieval = with_retrieval
        .as_ref()
        .map(|(_, _, judged, _)| pairs_of(judged, &candidate_judged))
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
            by_task: request
                .tasks
                .iter()
                .zip(found)
                .zip(candidate_judged.iter().zip(&arm_of.2))
                .map(|((task, found), (alone, with))| RetrievalTask {
                    task: task.task.id.to_string(),
                    evidence_found: found.finds_the_evidence_of(task),
                    alone: *alone,
                    with_passages: *with,
                })
                .collect(),
        });
    let families = clusters
        .iter()
        .flatten()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        + clusters.iter().filter(|c| c.is_none()).count();
    Ok(Examined {
        tasks: request.tasks.len(),
        families,
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
        paired_interval: (trusted && !pairs.is_empty())
            .then(|| gain_interval(&pairs))
            .flatten(),
        paired_vs_prompted: (trusted && !pairs_vs_prompted.is_empty())
            .then(|| sign_test(&pairs_vs_prompted)),
        retrieval,
        paired_retrieval: (trusted && !pairs_retrieval.is_empty())
            .then(|| sign_test(&pairs_retrieval)),
        training_warnings: Vec::new(),
    })
}

/// Resamples of the per-family gains the interval is made from.
const INTERVAL_RESAMPLES: usize = 2000;
/// The share of resampled means the interval holds.
const INTERVAL_LEVEL: f64 = 0.95;
/// The seed of the resampling: the same interval for the same gains.
const INTERVAL_SEED: u64 = 0;

/// The bootstrap interval of the candidate's gain per family over `pairs`
/// of `(candidate right, base right)`, one per family.
fn gain_interval(pairs: &[(bool, bool)]) -> Option<Interval> {
    let gains: Vec<f64> = pairs
        .iter()
        .map(|(candidate, base)| f64::from(*candidate) - f64::from(*base))
        .collect();
    bootstrap_interval(&gains, INTERVAL_RESAMPLES, INTERVAL_LEVEL, INTERVAL_SEED)
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

/// What the `exam` command asks of a stored candidate.
pub struct ExamineRequest<'a> {
    /// The candidate to examine, by id or unique prefix.
    pub candidate: &'a str,
    /// The model it is measured against; `None` is the model it continues:
    /// the release it was trained from, else the base alone - not what the
    /// alias points at now, which is the candidate itself once it is released.
    pub base: Option<&'a ModelRef>,
    /// The model that judges; `None` is the judge role's.
    pub judge: Option<&'a ModelRef>,
    /// A goal the base is also asked under, for a prompt-only baseline; see
    /// [`ExamRequest::prompted`].
    pub prompted: Option<&'a str>,
    /// Retrieval for a further arm; see [`ExamRequest::retrieval`].
    pub retrieval: Option<&'a Retrieval<'a>>,
}

/// Examines the stored candidate `request` names against what it continues,
/// on its held-out tasks, the judge calibrated on controls made from the
/// references of its tasks. An exam that has nothing to run on says so
/// instead of reporting an empty result.
pub fn examine(
    ctx: &Context,
    request: &ExamineRequest<'_>,
    cancel: &CancelToken,
) -> Result<Exam, OrchestratorError> {
    let judge = match request.judge {
        Some(named) => named.clone(),
        None => assignments(ctx.config(), &RoleOverrides::default())?
            .get(Role::Judge)
            .clone(),
    };
    let trained = load_candidate(ctx, request.candidate)?;
    let base = match request.base {
        Some(named) => named.clone(),
        None => {
            let continued = trained
                .parent
                .as_ref()
                .map(|id| ctx.releases().get(id))
                .transpose()?;
            let adapter = continued.as_ref().map(StoredRelease::adapter).transpose()?;
            arm(ctx.config(), adapter)
        }
    };
    // Each model is asked under the prompt it was trained under.
    let candidate_system = ctx.system_prompt_of(&trained.datasets)?;
    let base_system = match trained.parent.as_ref() {
        Some(parent) => ctx.system_prompt_of(&ctx.releases().get(parent)?.manifest.datasets)?,
        None => None,
    };
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
    let mut examined = exam(
        ctx,
        &ExamRequest {
            tasks: &tasks,
            controls: &controls,
            base: &base,
            candidate: &candidate,
            judge: &judge,
            prompted: request.prompted,
            retrieval: request.retrieval,
            base_system: base_system.as_deref(),
            candidate_system: candidate_system.as_deref(),
            cancel: cancel.clone(),
        },
    )?;
    examined.training_warnings = trained.warnings();
    Ok(Exam::Ran(Box::new(examined)))
}
