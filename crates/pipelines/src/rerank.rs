// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A model as retrieval's [`Reranker`]: it reads the question and one
//! passage and says whether the passage bears on the question.
//!
//! The model is asked through the same closed-book solve every task runs
//! through, and its reply's first word is the verdict (`YES`; anything else,
//! an empty reply included, leaves the passage where search put it). The
//! reply is bounded to a few tokens: the verdict is all that is read.

use std::time::Duration;

use splinter_agent::solve::{solve, Model, SolveOptions};
use splinter_core::experience::{Environment, Task};
use splinter_core::model_ref::ModelRef;
use splinter_knowledge::retrieve::{Passage, RerankError, Reranker};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_sandbox::ResolvedEnvironment;

/// The task kind of the closed-book question a reranker is asked.
const RERANK_TASK_KIND: &str = "rerank";

/// How long one judgment may take.
const DEADLINE: Duration = Duration::from_secs(120);

/// The most tokens a judgment may spend: the verdict is one word, and a
/// reader that goes on has no verdict to give, which leaves the passage where
/// search put it. Hundreds of passages are read for one question, so what a
/// reason would cost is paid hundreds of times for nothing that is used.
const MAX_JUDGMENT_TOKENS: u64 = 16;

const INSTRUCTIONS: &str = "You are helping someone answer a question from a body of writing. \
You see the question and one passage. Say whether the passage bears on the question: whether \
someone answering it would draw on what the passage says, not merely whether it mentions the \
same words. Reply with one word: YES or NO.";

/// A model judging passages, on `ctx`.
pub struct ModelReranker<'a> {
    ctx: &'a Context,
    model: Model,
}

impl<'a> ModelReranker<'a> {
    /// The model `reference` names, loaded.
    pub fn new(ctx: &'a Context, reference: &ModelRef) -> Result<Self, OrchestratorError> {
        Ok(Self {
            ctx,
            model: ctx.model(reference)?,
        })
    }
}

fn prompt(question: &str, passage: &Passage) -> String {
    format!(
        "{INSTRUCTIONS}\n\nQUESTION:\n{question}\n\nPASSAGE:\n{}\n",
        passage.text
    )
}

impl Reranker for ModelReranker<'_> {
    fn relevant(&self, question: &str, passage: &Passage) -> Result<bool, RerankError> {
        let failed = |e: &dyn std::fmt::Display| RerankError(e.to_string());
        let asked = Task::new(
            RERANK_TASK_KIND,
            vec![],
            Environment::closed_book(),
            prompt(question, passage),
            vec![],
        )
        .map_err(|e| failed(&e))?;
        let mut options = SolveOptions::new(DEADLINE);
        options.max_output_tokens = Some(MAX_JUDGMENT_TOKENS);
        options.stream_idle = self.model.stream_idle;
        let solution = self
            .ctx
            .block_on(solve(
                &asked,
                &ResolvedEnvironment::ClosedBook,
                self.model.provider.clone(),
                options,
            ))
            .map_err(|e| failed(&e))?;
        Ok(solution.final_output.is_some_and(|reply| {
            reply.split_whitespace().next().is_some_and(|word| {
                word.trim_matches(|c: char| !c.is_ascii_alphabetic())
                    .eq_ignore_ascii_case("yes")
            })
        }))
    }
}
