// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `ask`: one question put to a model the way every task is solved -
//! closed-book by default, so the answer is what the model knows; with
//! `--open-book` the named source's text rides along in the question.
//!
//! Every answer is recorded ([`crate::answers`]) with the model that gave
//! it and, asked through `policy:<alias>`, the release the alias resolved
//! to, so an answer traces back to what it was learned from.

use std::time::Duration;

use serde::Serialize;
use splinter_agent::solve::{solve, SolveOptions};
use splinter_sandbox::ResolvedEnvironment;
use splinter_store::experience::{Environment, Task};
use splinter_store::source::SourceId;

use crate::answers::{AnswerId, AnswerRecord, AnswerStore, ANSWER_FORMAT};
use crate::context::Context;
use crate::error::CampaignError;
use crate::model_ref::ModelRef;
use crate::release::ReleaseId;
use crate::solving::conclusion_name;
use crate::sources;

/// How long one answer may take.
pub const DEFAULT_ASK_DEADLINE: Duration = Duration::from_secs(300);

/// The most source text an open-book question carries, in bytes.
pub const MAX_OPEN_BOOK_BYTES: usize = 64 * 1024;

/// The task kind a question is asked as.
pub const ASK_TASK_KIND: &str = "ask";

/// What `ask` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Answer {
    /// The recorded answer's id.
    pub id: AnswerId,
    /// The question, as asked.
    pub question: String,
    /// The model's answer.
    pub answer: String,
    /// The identity of the model that answered.
    pub model: String,
    /// The source shown with the question, if any.
    pub open_book: Option<SourceId>,
    /// The release a `policy:` reference resolved to; `None` otherwise,
    /// and for the base before any release.
    pub release: Option<ReleaseId>,
}

/// Asks `question` of `policy`, with the text of the source `open_book`
/// names when one is given, and records the answer.
pub fn ask(
    ctx: &Context,
    question: &str,
    open_book: Option<&str>,
    policy: &ModelRef,
) -> Result<Answer, CampaignError> {
    if question.trim().is_empty() {
        return Err(CampaignError::Refused("the question is empty".into()));
    }
    let (instruction, open_book) = match open_book {
        None => (question.to_string(), None),
        Some(id) => {
            let source = sources::resolve(ctx, id)?;
            let material = material(ctx, &source)?;
            (
                format!("{question}\n\nReference material:\n\n{material}"),
                Some(source),
            )
        }
    };
    // The pin is the context's for its lifetime, so it names the release
    // the model below serves.
    let release = match policy {
        ModelRef::Policy(alias) => ctx.policy_pin(alias)?.map(|pin| pin.release),
        _ => None,
    };
    let model = ctx.model(policy)?;
    let task = Task::new(
        ASK_TASK_KIND,
        vec![],
        Environment::closed_book(),
        instruction,
        vec![],
    )?;
    let mut options = SolveOptions::new(DEFAULT_ASK_DEADLINE);
    options.stream_idle = model.stream_idle;
    let solution = ctx.block_on(solve(
        &task,
        &ResolvedEnvironment::ClosedBook,
        model.provider.clone(),
        options,
    ))?;
    let answer = solution.final_output.ok_or_else(|| CampaignError::Model {
        model: model.identity.clone(),
        detail: format!(
            "it gave no answer (its run ended: {})",
            conclusion_name(solution.conclusion)
        ),
    })?;
    let record = AnswerRecord {
        format: ANSWER_FORMAT.into(),
        question: question.to_string(),
        answer,
        model: model.identity,
        policy: policy.to_string(),
        release,
        open_book,
        asked_at: ctx.clock().utc_now(),
    };
    let id = AnswerStore::open(ctx.root()).put(&record)?;
    Ok(Answer {
        id,
        question: record.question,
        answer: record.answer,
        model: record.model,
        open_book: record.open_book,
        release: record.release,
    })
}

/// The source's text parts, each under its name; refused past
/// [`MAX_OPEN_BOOK_BYTES`] rather than cut, since a cut would silently
/// change what the model was shown.
fn material(ctx: &Context, id: &SourceId) -> Result<String, CampaignError> {
    let store = ctx.sources();
    let source = store.get_source(id)?;
    let mut text = String::new();
    for part in source
        .parts
        .iter()
        .filter(|p| p.media_type.starts_with("text/"))
    {
        let bytes = store.read_blob(&part.content)?;
        text.push_str(&format!("--- {} ---\n", part.name));
        text.push_str(&String::from_utf8_lossy(&bytes));
        text.push('\n');
        if text.len() > MAX_OPEN_BOOK_BYTES {
            return Err(CampaignError::Refused(format!(
                "source {id} holds more than {MAX_OPEN_BOOK_BYTES} bytes of text, more than one \
                 question can carry"
            )));
        }
    }
    if text.is_empty() {
        return Err(CampaignError::Refused(format!(
            "source {id} has no text part to show"
        )));
    }
    Ok(text)
}
