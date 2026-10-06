// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning that rehearses a model's own
// answers so new training does not cost it what it could already do, for
// its clients. If your team needs expertise in continual learning without
// catastrophic forgetting, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The rehearse stage: the base model's own answers to general tasks, as a
//! dataset a training run mixes in at a set share of its draws and
//! monitors on, so the new material does not move the model off what it
//! already does.
//!
//! A fine-tune on one person's answers erodes the base's general behaviour
//! before it forgets a fact: it answers a sum at length and gets it wrong,
//! or answers a plain question in a paragraph where a word was asked. What
//! holds a model to its base is training beside the new records on what the
//! base itself answers - self-synthesized rehearsal, with no corpus from
//! outside - and what moves it least is the base's own text, on-policy for
//! the model being trained.
//!
//! The tasks come from two places, neither a corpus: `arithmetic` and
//! `format` tasks built by code from a seed with their references
//! ([`splinter_knowledge::rehearsal::form_tasks`]), [`FORM_SHARE`] of the
//! set; and general requests the base model itself writes, one domain at a
//! time ([`splinter_knowledge::rehearsal::GENERAL_DOMAINS`]), the rest. Every
//! task passes the one admission ([`splinter_knowledge::rehearsal::Admission`]):
//! nothing the frozen anchor suite holds, nor a near copy of it, is ever in
//! the set, since the anchor is the gate's held-out check of general
//! behaviour and a candidate rehearsed on it would pass that check by
//! training on it. The base - the policy's base checkpoint with no adapter -
//! answers every task greedily under the default assistant prompt, as the
//! gate asks it; its answers are graded by their kinds' code verifiers alone
//! and the `rehearsal` view keeps every answer not decided wrong. The
//! dataset is stored like any other and `train --rehearsal` mixes it in.

use std::time::Instant;

use serde::Serialize;
use splinter_agent::prompts::{propose_prompts, PromptRequest};
use splinter_agent::{CallError, CancelToken};
use splinter_core::experience::{Environment, Task};
use splinter_core::kinds::GENERAL;
use splinter_knowledge::rehearsal::{
    form_tasks, Admission, GeneralRequest, GENERAL_BRIEF, GENERAL_DOMAINS, GENERATOR,
};
use splinter_knowledge::tasks::{DEFAULT_REPAIRS, DEFAULT_REQUEST_DEADLINE};
use splinter_model::local::GREEDY_SAMPLING;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

use crate::datasets::{build, BuildRequest, Built, ViewName};
use crate::release::probe::greedy;
use crate::release::{anchor, arm};
use crate::solving::{solve_tasks, SamplingChoice, SolveRequest, Solved};
use crate::tasks::remaining;
use crate::verify::{verify_set, Grading, Verified};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The share of a supervised run's training draws the rehearsed records
/// take when a run with a persona names none: a quarter, as the replay of
/// earlier releases takes, so the new records keep most of the steps while
/// every fourth draw holds the model to its base.
pub const DEFAULT_REHEARSAL_SHARE: f64 = 0.25;

/// The seed the form tasks are built from: the same tasks every time for
/// the same count.
pub const REHEARSAL_SEED: u64 = 0;

/// The share of a rehearsal set that is form tasks - sums and format
/// requests, which a persona fine-tune erodes first and which code can
/// grade - the rest general requests the base writes itself.
pub const FORM_SHARE: f64 = 0.5;

/// How many general requests one brief asks the base model for.
pub const PROMPTS_PER_REQUEST: usize = 8;

/// One rehearse.
#[derive(Clone, Debug, Serialize)]
pub struct RehearseRequest {
    /// How many tasks the set is to hold.
    pub records: usize,
    /// The seed the form tasks are built from.
    pub seed: u64,
    /// No model request starts after this.
    #[serde(skip)]
    pub deadline: Option<Instant>,
    /// Stops the stage.
    #[serde(skip)]
    pub cancel: CancelToken,
}

/// What the rehearse stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct Rehearsed {
    /// The base model that wrote the general requests and answered every
    /// task, as its identity names it.
    pub base: String,
    /// The anchor suite whose tasks were kept out: `(version, digest)`;
    /// `None` when none is frozen.
    pub anchor: Option<(u32, String)>,
    /// The task set.
    pub task_set: TaskSetId,
    /// Tasks in it.
    pub tasks: usize,
    /// Of those, the form tasks built by code.
    pub form_tasks: usize,
    /// Of those, the general requests the base wrote.
    pub general_tasks: usize,
    /// Proposals refused for repeating an anchor task or a near copy of
    /// one.
    pub excluded_as_anchor: usize,
    /// Proposals refused for repeating an admitted task.
    pub repeated: usize,
    /// Requests for general prompts that came back malformed or empty.
    pub failed_requests: usize,
    /// The base's answers.
    pub solved: Solved,
    /// Their grading, by the kinds' code verifiers.
    pub verified: Verified,
    /// The rehearsal dataset (`train ... --rehearsal <dataset>`).
    pub dataset: Built,
    /// Why fewer tasks than asked were written, if so.
    pub stopped: Option<String>,
}

/// Builds the rehearsal set `request` asks for; see the module
/// documentation.
pub fn rehearse(ctx: &Context, request: &RehearseRequest) -> Result<Rehearsed, OrchestratorError> {
    if request.records == 0 {
        return Err(OrchestratorError::Refused(
            "a rehearsal set holds at least one task".into(),
        ));
    }
    let frozen = anchor::current(ctx)?;
    let excluded: Vec<String> = frozen
        .iter()
        .flat_map(|f| f.suite.tasks.iter().map(|t| t.instruction.clone()))
        .collect();
    let mut admission = Admission::new(excluded.iter().map(String::as_str));
    let base = arm(ctx.config(), None);
    let writer = ctx.model(&base)?;

    let wanted_form = (request.records as f64 * FORM_SHARE).ceil() as usize;
    let mut members: Vec<(Task, Option<String>)> = Vec::new();
    // The generator's sequence is the same for a longer count, so asking
    // for more extends it past the tasks considered so far; a round that
    // admits nothing new ends the search.
    let mut considered = 0;
    let mut asked = wanted_form;
    while members.len() < wanted_form {
        let before = members.len();
        for task in form_tasks(request.seed, asked).into_iter().skip(considered) {
            considered += 1;
            if admission.admit(&task.instruction).is_ok() {
                members.push((task, Some(GENERATOR.to_string())));
                if members.len() == wanted_form {
                    break;
                }
            }
        }
        if members.len() == before {
            break;
        }
        asked *= 2;
    }
    let form = members.len();

    let mut stopped = None;
    let mut failed_requests = 0;
    let wanted_general = request.records.saturating_sub(form);
    let mut general = 0;
    'domains: while general < wanted_general {
        let before = general;
        for domain in GENERAL_DOMAINS {
            if general == wanted_general {
                break;
            }
            if request.cancel.is_cancelled() {
                return Err(OrchestratorError::Cancelled);
            }
            if request.deadline.is_some_and(|d| Instant::now() >= d) {
                stopped =
                    Some("the budget was spent before every general request was written".into());
                break 'domains;
            }
            let proposal = PromptRequest {
                brief: GENERAL_BRIEF,
                input: &GeneralRequest {
                    domain,
                    count: PROMPTS_PER_REQUEST.min(wanted_general - general),
                },
                deadline: remaining(request.deadline, DEFAULT_REQUEST_DEADLINE),
                max_output_tokens: None,
                repairs: DEFAULT_REPAIRS,
                cancel: Some(request.cancel.clone()),
            };
            let prompts = match ctx.block_on(propose_prompts(&writer, &proposal)) {
                Ok(prompts) => prompts,
                Err(CallError::Invalid { .. } | CallError::Stopped { .. }) => {
                    failed_requests += 1;
                    continue;
                }
                Err(e) => {
                    return Err(OrchestratorError::Refused(format!(
                        "asking {} for general requests: {e}",
                        writer.identity
                    )))
                }
            };
            for prompt in prompts {
                if general == wanted_general {
                    break;
                }
                if admission.admit(&prompt).is_err() {
                    continue;
                }
                let task = Task::new(
                    GENERAL,
                    Vec::new(),
                    Environment::closed_book(),
                    prompt,
                    Vec::new(),
                )?;
                members.push((task, Some(writer.identity.clone())));
                general += 1;
            }
        }
        if general == before {
            stopped = Some(format!(
                "a pass over every domain admitted no new request; {general} of {wanted_general} written"
            ));
            break;
        }
    }

    let store = ctx.tasks();
    let mut entries = Vec::with_capacity(members.len());
    for (task, generator) in &members {
        entries.push(TaskEntry {
            task: store.put(task)?,
            generator: generator.clone(),
            prompt: None,
            variant_of: None,
            subject: None,
        });
    }
    let task_set = store.put_set(&TaskSet {
        name: format!(
            "rehearsal tasks for {}: {form} by code, {general} general by the base",
            writer.identity
        ),
        members: entries,
    })?;

    // The base answers as the gate asks it: greedily, under the default
    // prompt, with no adapter.
    let greedy = greedy(ctx, &base)?;
    let solved = solve_tasks(
        ctx,
        &SolveRequest {
            task_set: &task_set,
            solver: &base,
            attempts: 1,
            sampling: SamplingChoice::Prefer(GREEDY_SAMPLING),
            teacher: false,
            system: None,
            deadline: request.deadline,
            cancel: request.cancel.clone(),
        },
    )?;
    drop(greedy);
    let verified = verify_set(
        ctx,
        &solved.experience_set,
        Grading::CodeOnly,
        &request.cancel,
    )?;
    let dataset = build(
        ctx,
        &BuildRequest {
            sets: vec![solved.experience_set.clone()],
            view: ViewName::Rehearsal,
            strip: None,
            min_strength: None,
            system_prompt: None,
            export_only: false,
            limit: None,
        },
    )?;
    Ok(Rehearsed {
        base: writer.identity.clone(),
        anchor: frozen.map(|f| (f.suite.version, f.digest.to_string())),
        task_set,
        tasks: members.len(),
        form_tasks: form,
        general_tasks: general,
        excluded_as_anchor: admission.excluded_count,
        repeated: admission.repeated_count,
        failed_requests,
        solved,
        verified,
        dataset,
        stopped,
    })
}
