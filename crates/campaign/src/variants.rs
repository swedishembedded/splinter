// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements contamination-free evaluation of fine-tuned
// models, for its clients. If your team needs expertise in model
// evaluation, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The variants stage: the tasks kept for training are asked again in
//! other words, so the release gate can measure what was learned.
//!
//! A fact-learning campaign cannot be judged on questions about other
//! facts, and the question it trained on says nothing new. What shows the
//! candidate learned a fact is the same fact asked differently. For each
//! task of a task set the generator model - the one that wrote the tasks -
//! writes up to [`DEFAULT_VARIANTS_PER_TASK`] differently worded questions
//! about the same fact, each naming the subject the original names, one
//! request per task, showing it the section the task's evidence falls in
//! ([`splinter_knowledge::tasks::variants`]). What
//! code admits is stored as a task set of its own: each member a task with
//! the original's kind, evidence and reference, its entry recording the
//! original it is a variant of.
//!
//! A variant is measured and never trained on: the set is not the one
//! `solve` works on, `dataset build` refuses an experience of a variant
//! ([`refuse_variants`]), and the gate leaves out any variant whose wording
//! the candidate trained on ([`crate::release::leakage`]).

use std::sync::Arc;

use splinter_agent::proposer::SvenProposer;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_core::experience::Experience;
use splinter_knowledge::tasks::{
    can_vary, GenerateError, GenerationPolicy, ModelTaskGenerator, DEFAULT_REQUEST_DEADLINE,
};
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};
use sven_sdk::CancelToken;

pub use splinter_knowledge::tasks::DEFAULT_VARIANTS_PER_TASK;

use crate::context::Context;
use crate::error::CampaignError;
use crate::model_ref::ModelRef;
use crate::tasks::{reason_name, remaining, RejectionNote};

/// One variants request.
pub struct VariantsRequest<'a> {
    /// The tasks to write variants of.
    pub task_set: &'a TaskSetId,
    /// The generator model.
    pub generator: &'a ModelRef,
    /// The most variants one task is given.
    pub per_task: usize,
    /// No request starts after this, and none runs past it.
    pub deadline: Option<Instant>,
    /// Stops generation.
    pub cancel: CancelToken,
}

/// What the variants stage reports.
#[derive(Clone, Debug, Default, Serialize)]
pub struct VariantsGenerated {
    /// The set of variants (`tasks show <set>`); `None` when none was
    /// admitted.
    pub variant_set: Option<TaskSetId>,
    /// Tasks asked for variants of.
    pub tasks: usize,
    /// Variants admitted.
    pub variants: usize,
    /// Tasks that cannot be varied, by reason.
    pub ineligible: BTreeMap<String, usize>,
    /// Proposals rejected, by reason.
    pub rejected: BTreeMap<String, usize>,
    /// Every rejection, with what failed.
    pub rejections: Vec<RejectionNote>,
    /// Why generation stopped before every task was covered, if it did.
    pub stopped: Option<String>,
}

/// Writes variants of the tasks of `request.task_set` and stores them as a
/// task set of their own. A task that cannot be varied is counted, not an
/// error; so is a proposal the model got wrong.
pub fn generate_variants(
    ctx: &Context,
    request: &VariantsRequest<'_>,
) -> Result<VariantsGenerated, CampaignError> {
    let originals = ctx.tasks().get_set(request.task_set)?;
    let policy = GenerationPolicy {
        deadline: remaining(request.deadline, DEFAULT_REQUEST_DEADLINE),
        ..GenerationPolicy::default()
    };
    let proposer =
        SvenProposer::new(ctx.model(request.generator)?).with_cancel(request.cancel.clone());
    let generator = ModelTaskGenerator::new(Arc::new(proposer), ctx.sources()).with_policy(policy);
    let mut out = VariantsGenerated::default();
    let mut members: Vec<TaskEntry> = Vec::new();
    for entry in &originals.members {
        if request.cancel.is_cancelled() {
            return Err(CampaignError::Cancelled);
        }
        if request.deadline.is_some_and(|d| Instant::now() >= d) {
            out.stopped = Some("the budget was spent before every task was covered".into());
            break;
        }
        let original = ctx.tasks().get(&entry.task)?;
        if let Err(reason) = can_vary(&original) {
            *out.ineligible.entry(reason.to_string()).or_default() += 1;
            continue;
        }
        // A variant must keep naming what the original is about; a task
        // whose set records no subject has nothing to hold it to.
        let Some(subject) = entry.subject.as_deref() else {
            *out.ineligible.entry("no_subject".to_string()).or_default() += 1;
            continue;
        };
        let report = match ctx.block_on(generator.variants(&original, subject, request.per_task)) {
            Ok(report) => report,
            // The recorded subject is not in the instruction.
            Err(GenerateError::NotVaryable { reason, .. }) => {
                *out.ineligible.entry(reason.to_string()).or_default() += 1;
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        out.tasks += 1;
        for (reason, count) in &report.rejected {
            *out.rejected.entry(reason_name(reason)).or_default() += count;
        }
        out.rejections
            .extend(report.rejections.into_iter().map(|r| RejectionNote {
                reason: reason_name(&r.reason),
                kind: r.kind,
                detail: r.detail,
            }));
        for generated in report.admitted {
            let id = ctx.tasks().put(&generated.task)?;
            if members.iter().all(|m| m.task != id) {
                members.push(TaskEntry {
                    task: id,
                    generator: Some(generated.generator),
                    prompt: Some(generated.prompt),
                    variant_of: Some(entry.task.clone()),
                    subject: generated.subject,
                });
            }
        }
    }
    out.variants = members.len();
    if !members.is_empty() {
        out.variant_set = Some(ctx.tasks().put_set(&TaskSet {
            name: format!("variants of {}", request.task_set),
            members,
        })?);
    }
    Ok(out)
}

/// Every stored variant, with the task it is a variant of: each variant
/// once, in the order the task sets list them.
pub(crate) fn stored_variants(ctx: &Context) -> Result<Vec<(Digest, Digest)>, CampaignError> {
    let store = ctx.tasks();
    let mut seen = BTreeSet::new();
    let mut variants = Vec::new();
    for id in store.list_sets()? {
        for entry in store.get_set(&id)?.members {
            if let Some(original) = entry.variant_of {
                if seen.insert(entry.task.clone()) {
                    variants.push((entry.task, original));
                }
            }
        }
    }
    Ok(variants)
}

/// Refuses `experiences` when one is of a variant task: a variant is
/// measured, and what is trained on is what was never measured.
pub(crate) fn refuse_variants<'a>(
    ctx: &Context,
    experiences: impl IntoIterator<Item = &'a Experience>,
) -> Result<(), CampaignError> {
    let variants: BTreeSet<Digest> = stored_variants(ctx)?.into_iter().map(|(v, _)| v).collect();
    if variants.is_empty() {
        return Ok(());
    }
    for experience in experiences {
        let task = experience.to_task().without_critiques()?;
        if variants.contains(&task.task.id) {
            return Err(CampaignError::Refused(format!(
                "task {} is a variant of another: variants are only measured, never trained on",
                task.task.id
            )));
        }
    }
    Ok(())
}
