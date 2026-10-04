// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that hold a conversation as a
// person from what that person wrote, for its clients. If your team needs
// expertise in grounding a model's conversation in a body of writing, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A dialogue solved as one agent conversation.
//!
//! [`converse_prompted`] runs the teacher through one sven agent: the first
//! message is the opening prompt (the instruction with the grounding
//! material beside it, [`crate::solve::open_book_prompt`]), and each later
//! user turn is what the [`Interlocutor`] says next, so the trajectory sven
//! exports is the dialogue itself.
//!
//! The interlocutor is shown the opening instruction and the replies, never
//! the material: it plays someone who does not have the writing. The final
//! output is every reply in order, separated by blank lines, which is what a
//! verifier reads. A reply that does not conclude successfully ends the
//! dialogue with no output: half a dialogue teaches nothing.

use std::sync::Arc;

use sven_sdk::model::ModelProvider;
use sven_sdk::{RunConclusion, Usage};

use crate::solve::{engine, is_empty_reply, Solution, SolveError, SolveOptions, SOLVER_MODE};
use splinter_core::experience::Task;
use splinter_sandbox::ResolvedEnvironment;

/// One exchange: what the other speaker said and the reply to it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exchange {
    /// What the other speaker said.
    pub said: String,
    /// The reply it was given.
    pub reply: String,
}

/// The other speaker of a dialogue.
#[async_trait::async_trait]
pub trait Interlocutor: Send + Sync {
    /// What to say after `so_far`, whose first `said` is the opening
    /// instruction and whose last reply is the one to answer; `None` when
    /// there is nothing more to say.
    async fn next(&self, so_far: &[Exchange]) -> Option<String>;
}

/// Separates the replies in a dialogue's final output.
pub const REPLY_SEPARATOR: &str = "\n\n";

/// Solves `task` as a dialogue of at most `turns` exchanges: `opening` is
/// the teacher's first prompt, and `interlocutor` supplies the rest; see the
/// module documentation.
pub async fn converse_prompted(
    task: &Task,
    opening: &str,
    environment: &ResolvedEnvironment,
    model: Arc<dyn ModelProvider>,
    interlocutor: &dyn Interlocutor,
    turns: usize,
    options: SolveOptions,
) -> Result<Solution, SolveError> {
    let offered = environment.record()?;
    if offered != task.environment {
        return Err(SolveError::EnvironmentMismatch {
            task_kind: task.environment.kind.clone(),
            task_snapshot: task.environment.snapshot.clone(),
            offered_kind: offered.kind,
            offered_snapshot: offered.snapshot,
        });
    }
    let engine = engine(
        environment,
        model,
        options.engine_config(),
        options.system_prompt(),
    )?;
    let mut agent = engine.agent(SOLVER_MODE);
    let mut exchanges: Vec<Exchange> = Vec::new();
    let mut said = opening.to_string();
    let mut shown = task.instruction.clone();
    let mut usage = Usage::default();
    let mut conclusion = RunConclusion::Success;
    for _ in 0..turns.max(1) {
        let outcome = match agent.send_with(&said, options.run_options()).await {
            Ok(outcome) => outcome,
            // A reply of nothing ends the dialogue unanswered, as a stopped
            // run does: half a dialogue teaches nothing.
            Err(e) if is_empty_reply(&e) => {
                conclusion = RunConclusion::AgentError;
                break;
            }
            Err(e) => return Err(e.into()),
        };
        usage = add(usage, &outcome.usage);
        conclusion = outcome.conclusion;
        if conclusion != RunConclusion::Success {
            break;
        }
        exchanges.push(Exchange {
            said: std::mem::take(&mut shown),
            reply: outcome.reply,
        });
        if exchanges.len() == turns {
            break;
        }
        match interlocutor.next(&exchanges).await {
            Some(next) => {
                shown.clone_from(&next);
                said = next;
            }
            None => break,
        }
    }
    let final_output = (conclusion == RunConclusion::Success).then(|| {
        exchanges
            .iter()
            .map(|e| e.reply.as_str())
            .collect::<Vec<_>>()
            .join(REPLY_SEPARATOR)
    });
    Ok(Solution {
        trajectory: agent.trajectory(),
        final_output,
        conclusion,
        usage,
    })
}

/// `total` and `more` summed; a count neither reports stays unreported.
fn add(total: Usage, more: &Usage) -> Usage {
    let sum = |a: Option<u64>, b: Option<u64>| match (a, b) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0) + b.unwrap_or(0)),
    };
    Usage {
        input_tokens: sum(total.input_tokens, more.input_tokens),
        output_tokens: sum(total.output_tokens, more.output_tokens),
    }
}
