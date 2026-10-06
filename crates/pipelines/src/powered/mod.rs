// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out examinations of what a model
// learned from a person's writing, for its clients. If your team needs
// expertise in measuring a fine-tune with enough power to tell a gain from
// luck, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The powered exam: a candidate, its base and the base told what the
//! candidate was trained to be, put to a frozen exam ([`crate::exam_set`])
//! written from reserved families, with every verdict of every arm kept.
//!
//! The arms are the base (the default prompt), the prompted base (the prompt
//! the candidate was trained under, or the learner's goal when it has none),
//! the candidate asked under the default prompt, and the candidate asked
//! under the prompt it was trained under, as it is deployed. Each task is
//! answered once greedily - the verdict the comparisons are made on - and
//! `resamples - 1` times more as the model samples, from which each task has
//! a pass rate. A judge calibrated on controls made from the tasks' own
//! references grades every answer, blind to which arm gave it: it is shown
//! the task, the reference and the answer, decodes greedily, and each answer
//! is judged on its own, so no order among the arms can reach a verdict. Code
//! checks every answer for numbers and names the source does not hold. The
//! voice of each arm is also scored, with no judge, by the likelihood it
//! gives the writer's own text ([`voice`]).
//!
//! An exam family the candidate, or any release it continues, was trained on
//! is left out: a model that has seen the text is not examined on it.
//! Everything is kept in the report, answers included, so the analysis
//! ([`analysis`]) can be redone without a model.

pub mod analysis;
pub mod voice;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use splinter_agent::solve::{with_system_addendum, Model};
use splinter_agent::CancelToken;
use splinter_core::annotation::Outcome;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, Task};
use splinter_eval::verifiers::Verifier;

use crate::exam::{calibrate_judge, JudgeTrust};
use crate::exam_set::{ExamSet, ExamTask};
use crate::grouping::groups_of;
use crate::judging::{hard_controls, reference};
use crate::release::arm as arm_ref;
use crate::release::probe::{answer_prompted, greedy};
use crate::reserve::touched_blobs;
use crate::train::{load_candidate, Candidate};
use crate::verify::grounding_verifier;
use analysis::{compare, summarise, Answer, ArmSummary, Comparison, TaskRecord};
use splinter_core::model_ref::ModelRef;
use splinter_core::role::{Role, RoleOverrides};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::roles::assignments;

/// The base: the policy base asked under the default prompt.
pub const BASE: &str = "base";
/// The base told what the candidate was trained to be.
pub const PROMPTED: &str = "prompted";
/// The candidate asked under the default prompt.
pub const CANDIDATE: &str = "candidate";
/// The candidate asked under the prompt it was trained under.
pub const PERSONA: &str = "candidate-persona";

/// Answers per task per arm when none is named: the greedy one and two
/// sampled.
pub const DEFAULT_RESAMPLES: usize = 3;

/// What the powered exam is asked.
pub struct PoweredRequest<'a> {
    /// The frozen exam.
    pub exam: &'a ExamSet,
    /// The candidate, by id or unique prefix.
    pub candidate: &'a str,
    /// The model it is measured against; `None` is the policy base.
    pub base: Option<&'a ModelRef>,
    /// The judge; `None` is the judge role's.
    pub judge: Option<&'a ModelRef>,
    /// The goal the base is told when the candidate was trained under no
    /// prompt of its own.
    pub goal: Option<&'a str>,
    /// Answers per task per arm, the first greedy; at least one.
    pub resamples: usize,
    /// Whether to score the voice (it loads each arm on the device).
    pub voice: bool,
    /// Stops the exam.
    pub cancel: &'a CancelToken,
}

/// The judge on a harder set of controls than the ones it is calibrated on.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HardControls {
    /// Wrong answers put to it: each task answered with the reference of the
    /// task of another family most like its own reference in words.
    pub wrong_answers: usize,
    /// Of those, the ones it passed: its false passes.
    pub passed: usize,
}

/// What the powered exam found.
#[derive(Clone, Debug, Serialize)]
pub struct Powered {
    /// The exam it was put to.
    pub exam: String,
    /// The candidate.
    pub candidate: String,
    /// Tasks the exam holds.
    pub tasks_in_exam: usize,
    /// Tasks examined, once the families the candidate was trained on were
    /// left out.
    pub tasks: usize,
    /// Families examined.
    pub families: usize,
    /// Families left out for having been trained on.
    pub families_trained_on: usize,
    /// Answers per task per arm.
    pub resamples: usize,
    /// The system prompt each arm was asked under.
    pub prompts: BTreeMap<String, String>,
    /// The judge and what it was measured at.
    pub judge: JudgeTrust,
    /// The judge on harder wrong answers than the calibration's.
    pub hard_controls: Option<HardControls>,
    /// Each arm over the whole exam.
    pub arms: Vec<ArmSummary>,
    /// The comparisons, the first arm of each the one said to be better.
    pub comparisons: Vec<Comparison>,
    /// The voice of each arm, by the likelihood it gives the writer's text.
    pub voice: Vec<voice::VoiceScore>,
    /// Every task with every arm's every answer and verdict.
    pub records: Vec<TaskRecord>,
    /// What the candidate's training curve warns of.
    pub training_warnings: Vec<String>,
}

/// One arm to be asked.
struct ArmSpec {
    name: &'static str,
    reference: ModelRef,
    /// Whether the candidate's adapter is folded in.
    tuned: bool,
    /// The system prompt it is asked under; `None` is the default.
    system: Option<String>,
    /// The goal added to the default prompt, for a prompted base with no
    /// prompt of the candidate's to be told.
    goal: Option<String>,
}

/// How a report names a prompt.
fn prompt_name(system: Option<&str>, goal: Option<&str>) -> String {
    const SHOWN: usize = 60;
    match (system, goal) {
        (Some(system), _) => {
            let start: String = system.chars().take(SHOWN).collect();
            format!("persona: {start}")
        }
        (None, Some(goal)) => format!("default + purpose: {goal}"),
        (None, None) => "default".into(),
    }
}

/// The datasets that made `candidate` and everything it continues.
fn datasets_trained_on(
    ctx: &Context,
    candidate: &Candidate,
) -> Result<Vec<DatasetId>, OrchestratorError> {
    let mut datasets = candidate.datasets.clone();
    if let Some(parent) = &candidate.parent {
        for release in ctx.releases().lineage(parent)? {
            datasets.extend(release.manifest.datasets);
        }
    }
    Ok(datasets)
}

/// The text parts of the exam's sources that some dataset trained on prints,
/// in whole or in part, and which of `tasks` were written from them: what a
/// model trained on `datasets` has seen. Overlap is the rule that groups two
/// printings, so another edition of a letter trained on counts.
fn seen_text(
    ctx: &Context,
    exam: &ExamSet,
    tasks: &[Task],
    datasets: &[DatasetId],
) -> Result<(BTreeSet<Digest>, Vec<bool>), OrchestratorError> {
    let touched = touched_blobs(ctx, datasets)?;
    let group_of = groups_of(ctx, &exam.sources, touched.clone())?;
    let touched_groups: BTreeSet<&String> =
        touched.iter().filter_map(|b| group_of.get(b)).collect();
    let in_seen_group = |blob: &Digest| {
        group_of
            .get(blob)
            .is_some_and(|group| touched_groups.contains(group))
    };
    let mut parts = BTreeSet::new();
    for id in &exam.sources {
        for part in ctx.sources().get_source(id)?.parts {
            if in_seen_group(&part.content) {
                parts.insert(part.content);
            }
        }
    }
    let seen = tasks
        .iter()
        .map(|task| task.evidence.iter().any(|span| in_seen_group(&span.source)))
        .collect();
    Ok((parts, seen))
}

/// The arms to ask, from the base and the candidate.
fn arms_of(
    base: &ModelRef,
    candidate: &ModelRef,
    persona: Option<&str>,
    goal: Option<&str>,
) -> Vec<ArmSpec> {
    let mut specs = vec![ArmSpec {
        name: BASE,
        reference: base.clone(),
        tuned: false,
        system: None,
        goal: None,
    }];
    if persona.is_some() || goal.is_some() {
        specs.push(ArmSpec {
            name: PROMPTED,
            reference: base.clone(),
            tuned: false,
            system: persona.map(str::to_string),
            goal: goal.filter(|_| persona.is_none()).map(str::to_string),
        });
    }
    specs.push(ArmSpec {
        name: CANDIDATE,
        reference: candidate.clone(),
        tuned: true,
        system: None,
        goal: None,
    });
    if let Some(persona) = persona {
        specs.push(ArmSpec {
            name: PERSONA,
            reference: candidate.clone(),
            tuned: true,
            system: Some(persona.to_string()),
            goal: None,
        });
    }
    specs
}

/// Puts `request.candidate` to the exam; see the module documentation.
pub fn run(ctx: &Context, request: &PoweredRequest<'_>) -> Result<Powered, OrchestratorError> {
    if request.resamples == 0 {
        return Err(OrchestratorError::Refused(
            "an exam asks every task at least once: --resamples is at least 1".into(),
        ));
    }
    let judge_ref = match request.judge {
        Some(named) => named.clone(),
        None => assignments(ctx.config(), &RoleOverrides::default())?
            .get(Role::Judge)
            .clone(),
    };
    let trained = load_candidate(ctx, request.candidate)?;
    let base_ref = match request.base {
        Some(named) => named.clone(),
        None => arm_ref(ctx.config(), None),
    };
    let persona = ctx.system_prompt_of(&trained.datasets)?;
    let store = ctx.tasks();
    let all: Vec<Task> = request
        .exam
        .tasks
        .iter()
        .map(|t| store.get(&t.task))
        .collect::<Result<_, _>>()?;
    let (seen_parts, seen) = seen_text(
        ctx,
        request.exam,
        &all,
        &datasets_trained_on(ctx, &trained)?,
    )?;
    let families_trained_on = request
        .exam
        .tasks
        .iter()
        .zip(&seen)
        .filter(|(_, seen)| **seen)
        .map(|(t, _)| t.family.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let (tasks, examined): (Vec<Task>, Vec<&ExamTask>) = all
        .into_iter()
        .zip(&request.exam.tasks)
        .zip(&seen)
        .filter(|(_, seen)| !**seen)
        .map(|(pair, _)| pair)
        .unzip();
    if tasks.len() < 2 {
        return Err(OrchestratorError::Refused(format!(
            "{} of the exam's {} tasks are of text the candidate or a release it continues was \
             trained on, leaving too few to examine it on: reserve an exam from other text",
            request.exam.tasks.len() - tasks.len(),
            request.exam.tasks.len()
        )));
    }

    let candidate_ref = arm_ref(ctx.config(), Some(&trained.adapter));
    let specs = arms_of(&base_ref, &candidate_ref, persona.as_deref(), request.goal);
    // Every arm answers before the judge is loaded, so the device swaps
    // models as few times as it can.
    let mut answered: Vec<(&'static str, Vec<Vec<Experience>>)> = Vec::new();
    for spec in &specs {
        answered.push((spec.name, ask(ctx, spec, &tasks, request)?));
    }

    let with_reference: Vec<Task> = tasks
        .iter()
        .filter(|t| reference(t).is_some())
        .cloned()
        .collect();
    let calibrated = calibrate_judge(ctx, &judge_ref, &with_reference)?;
    let hard_controls = if calibrated.trust.trusted {
        let hard = hard_controls(ctx, &with_reference)?;
        let mut passed = 0;
        for (task, exp) in &hard {
            if request.cancel.is_cancelled() {
                return Err(OrchestratorError::Cancelled);
            }
            if calibrated.judge.verify(task, exp)?.outcome == Outcome::Pass {
                passed += 1;
            }
        }
        (!hard.is_empty()).then_some(HardControls {
            wrong_answers: hard.len(),
            passed,
        })
    } else {
        None
    };
    let grounding = grounding_verifier(ctx);
    let mut records: Vec<TaskRecord> = examined
        .iter()
        .map(|t| TaskRecord {
            task: t.task.to_string(),
            family: t.family.clone(),
            arms: BTreeMap::new(),
        })
        .collect();
    for (name, per_task) in &answered {
        for (n, answers) in per_task.iter().enumerate() {
            let mut graded = Vec::with_capacity(answers.len());
            for exp in answers {
                if request.cancel.is_cancelled() {
                    return Err(OrchestratorError::Cancelled);
                }
                graded.push(grade(&calibrated.judge, &grounding, &tasks[n], exp)?);
            }
            records[n].arms.insert((*name).to_string(), graded);
        }
    }

    let arm_names: Vec<String> = specs.iter().map(|s| s.name.to_string()).collect();
    let has = |name: &str| arm_names.iter().any(|a| a == name);
    // A judge that is not trusted abstains on every answer, so there is
    // nothing to compare and the report makes no claim.
    let comparisons = if calibrated.trust.trusted {
        [
            (PERSONA, PROMPTED),
            (PERSONA, BASE),
            (CANDIDATE, PROMPTED),
            (CANDIDATE, BASE),
            (PERSONA, CANDIDATE),
            (PROMPTED, BASE),
        ]
        .into_iter()
        .filter(|(first, second)| has(first) && has(second))
        .map(|(first, second)| compare(&records, first, second))
        .collect()
    } else {
        Vec::new()
    };
    let families: BTreeSet<&str> = examined.iter().map(|t| t.family.as_str()).collect();

    let voice = if request.voice {
        let arms: Vec<voice::VoiceArm<'_>> = specs
            .iter()
            .map(|s| voice::VoiceArm {
                name: s.name,
                adapter: s.tuned.then_some(trained.adapter.as_path()),
                system: s.system.as_deref(),
            })
            .collect();
        ctx.release_bases();
        voice::score(
            ctx,
            request.exam,
            &seen_parts,
            &arms,
            &ctx.root().path().join("tmp"),
        )?
    } else {
        Vec::new()
    };
    Ok(Powered {
        exam: request.exam.id.clone(),
        candidate: trained.candidate.clone(),
        tasks_in_exam: request.exam.tasks.len(),
        tasks: tasks.len(),
        families: families.len(),
        families_trained_on,
        resamples: request.resamples,
        prompts: specs
            .iter()
            .map(|s| {
                (
                    s.name.to_string(),
                    prompt_name(s.system.as_deref(), s.goal.as_deref()),
                )
            })
            .collect(),
        judge: calibrated.trust,
        hard_controls,
        arms: summarise(&records, &arm_names),
        comparisons,
        voice,
        records,
        training_warnings: trained.warnings(),
    })
}

/// One answer, judged and checked against the source.
fn grade(
    judge: &dyn Verifier,
    grounding: &dyn Verifier,
    task: &Task,
    exp: &Experience,
) -> Result<Answer, OrchestratorError> {
    let decided = |outcome: Outcome| match outcome {
        Outcome::Pass => Some(true),
        Outcome::Fail => Some(false),
        Outcome::Abstain => None,
    };
    let Some(text) = exp.final_output.as_deref() else {
        // Nothing was said: not done, and nothing to hold to the source.
        return Ok(Answer {
            judged: Some(false),
            grounded: None,
            chars: 0,
            text: None,
        });
    };
    Ok(Answer {
        judged: decided(judge.verify(task, exp)?.outcome),
        grounded: decided(grounding.verify(task, exp)?.outcome),
        chars: text.chars().count(),
        text: Some(text.to_string()),
    })
}

/// `spec` asked every task `request.resamples` times: the greedy answer,
/// then sampled ones.
fn ask(
    ctx: &Context,
    spec: &ArmSpec,
    tasks: &[Task],
    request: &PoweredRequest<'_>,
) -> Result<Vec<Vec<Experience>>, OrchestratorError> {
    let under = |model: Model| -> Model {
        let mut model = match &spec.system {
            Some(system) => model.with_system(system),
            None => model,
        };
        if let Some(goal) = &spec.goal {
            model.provider =
                with_system_addendum(model.provider.clone(), &format!("Your purpose: {goal}"));
        }
        model
    };
    let greedy_model = under(greedy(ctx, &spec.reference)?);
    let sampled_model = if request.resamples > 1 {
        Some(under(ctx.model(&spec.reference)?))
    } else {
        None
    };
    let mut answers = Vec::with_capacity(tasks.len());
    for task in tasks {
        let mut given = vec![answer_prompted(
            ctx,
            &greedy_model,
            task,
            None,
            request.cancel,
        )?];
        if let Some(model) = &sampled_model {
            for _ in 1..request.resamples {
                given.push(answer_prompted(ctx, model, task, None, request.cancel)?);
            }
        }
        answers.push(given);
    }
    Ok(answers)
}
