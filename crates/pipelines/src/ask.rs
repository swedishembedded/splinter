// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `ask`: one question put to a model the way every task is solved -
//! closed-book by default, so the answer is what the model knows; with
//! `--open-book` the named source's text is shown before the question, as
//! a teacher is shown a task's grounding material
//! ([`splinter_agent::solve::open_book_prompt`]), each part under what the
//! source is ([`SourceIdentity::label`]): output captured from a command
//! says nothing of which program printed it.
//!
//! With retrieval ([`ask_retrieving`]) the model is shown instead the few
//! passages of the named sources that bear on the question
//! ([`splinter_knowledge::retrieve::Library`]), each under the part and
//! section it is from, and the answer records the sources they came from.
//!
//! Every answer is recorded ([`splinter_orchestrator::answers`]) with the model that gave
//! it and, asked through `policy:<alias>`, the release the alias resolved
//! to, so an answer traces back to what it was learned from.

use std::time::Duration;

use serde::Serialize;
use splinter_agent::solve::{open_book_prompt, solve, SolveOptions};
use splinter_core::experience::{Environment, Task};
use splinter_core::source::SourceId;
use splinter_knowledge::retrieve::Embedder;
use splinter_knowledge::tasks::SourceIdentity;
use splinter_sandbox::ResolvedEnvironment;

use crate::retrieval::{library_of, Rerank, Retrieval};
use crate::solving::conclusion_name;
use crate::sources;
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_orchestrator::answers::{
    AnswerId, AnswerRecord, ShownPassage, ANSWER_FORMAT, EXCERPT_CHARS,
};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

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
    /// The sources the passages shown with the question were retrieved from;
    /// empty when none were.
    pub retrieved_from: Vec<SourceId>,
    /// The passages shown with the question, nearest in meaning first; empty
    /// when none were retrieved.
    pub shown: Vec<ShownPassage>,
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
) -> Result<Answer, OrchestratorError> {
    if question.trim().is_empty() {
        return Err(OrchestratorError::Refused("the question is empty".into()));
    }
    let (instruction, open_book) = match open_book {
        None => (question.to_string(), None),
        Some(id) => {
            let source = sources::resolve(ctx, id)?;
            let material = material(ctx, &source)?;
            (open_book_prompt(question, &material), Some(source))
        }
    };
    answer_with(
        ctx,
        question,
        instruction,
        Grounding {
            open_book,
            ..Grounding::default()
        },
        policy,
    )
}

/// Asks `question` of `policy` with the (at most) `passages` passages of
/// `sources` that bear on it, nearest in meaning first, found by `embedder`,
/// and records the answer. A source with nothing to retrieve is refused.
pub fn ask_retrieving(
    ctx: &Context,
    question: &str,
    sources: &[String],
    passages_shown: usize,
    embedder: &dyn Embedder,
    rerank: Option<Rerank<'_>>,
    policy: &ModelRef,
) -> Result<Answer, OrchestratorError> {
    if question.trim().is_empty() {
        return Err(OrchestratorError::Refused("the question is empty".into()));
    }
    let (ids, library) = library_of(ctx, sources, embedder)?;
    let retrieval = Retrieval {
        library: &library,
        embedder,
        passages: passages_shown,
        rerank,
    };
    let retrieved = retrieval.find(question)?;
    // A reader that was loaded to judge passages gives the device back
    // before the model that answers needs it.
    if retrieval.rerank.is_some() {
        ctx.release_bases();
    }
    let instruction = retrieved.prompt(question);
    let shown = retrieved.passages;
    let shown = shown
        .iter()
        .filter_map(|p| {
            Some(ShownPassage {
                source: p.source.clone()?,
                part: p.part.clone(),
                section: p.section + 1,
                excerpt: p.text.chars().take(EXCERPT_CHARS).collect(),
            })
        })
        .collect();
    let grounding = Grounding {
        open_book: None,
        retrieved_from: ids,
        shown,
    };
    answer_with(ctx, question, instruction, grounding, policy)
}

/// What a question was shown besides itself.
#[derive(Default)]
struct Grounding {
    open_book: Option<SourceId>,
    retrieved_from: Vec<SourceId>,
    shown: Vec<ShownPassage>,
}

/// Puts `instruction` to `policy` and records the answer to `question`.
fn answer_with(
    ctx: &Context,
    question: &str,
    instruction: String,
    grounding: Grounding,
    policy: &ModelRef,
) -> Result<Answer, OrchestratorError> {
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
    let options = model.solving(SolveOptions::new(DEFAULT_ASK_DEADLINE));
    let solution = ctx.block_on(solve(
        &task,
        &ResolvedEnvironment::ClosedBook,
        model.provider.clone(),
        options,
    ))?;
    let answer = solution
        .final_output
        .ok_or_else(|| OrchestratorError::Model {
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
        open_book: grounding.open_book,
        retrieved_from: grounding.retrieved_from,
        shown: grounding.shown,
        asked_at: ctx.clock().utc_now(),
    };
    let id = ctx.answers().put(&record)?;
    Ok(Answer {
        id,
        question: record.question,
        answer: record.answer,
        model: record.model,
        open_book: record.open_book,
        retrieved_from: record.retrieved_from,
        shown: record.shown,
        release: record.release,
    })
}

/// The source's text parts, each under its name; refused past
/// [`MAX_OPEN_BOOK_BYTES`] rather than cut, since a cut would silently
/// change what the model was shown.
fn material(ctx: &Context, id: &SourceId) -> Result<Vec<String>, OrchestratorError> {
    let store = ctx.sources();
    let source = store.get_source(id)?;
    let mut parts = Vec::new();
    let mut bytes_shown = 0;
    for part in source
        .parts
        .iter()
        .filter(|p| p.media_type.starts_with("text/"))
    {
        let bytes = store.read_blob(&part.content)?;
        let text = String::from_utf8_lossy(&bytes);
        let identity = SourceIdentity::of(&source.origin, &part.name, &text, &part.media_type);
        let piece = format!("--- {} ---\n{text}", identity.label());
        bytes_shown += piece.len();
        if bytes_shown > MAX_OPEN_BOOK_BYTES {
            return Err(OrchestratorError::Refused(format!(
                "source {id} holds more than {MAX_OPEN_BOOK_BYTES} bytes of text, more than one \
                 question can carry"
            )));
        }
        parts.push(piece);
    }
    if parts.is_empty() {
        return Err(OrchestratorError::Refused(format!(
            "source {id} has no text part to show"
        )));
    }
    Ok(parts)
}
