// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that survey a body of writing and
// plan what to learn from it, for its clients. If your team needs expertise
// in turning an unfamiliar corpus into a learning plan, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The plan: how to learn from a set of sources, chosen by a planner model
//! and held to what the sources hold by code.
//!
//! The planner is shown the learner's goal and a [`Survey`] of the sources and
//! chooses which kinds of task to generate, whether to distil from a teacher
//! instead of measuring the student's own attempts, and who the learner is
//! becoming. It chooses from a menu, so it cannot invent a method; code then
//! checks the choice against the survey, so it cannot choose to teach advice
//! the sources do not contain. A plan that breaks a rule is sent back once
//! with the rule it broke (the call's own correction), and one still broken
//! is refused. No default replaces a refused plan: a run that cannot be
//! planned says so.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_agent::schemars::JsonSchema;
use splinter_agent::typed::TypedCall;
use splinter_agent::CancelToken;
use splinter_knowledge::survey::Survey;

use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// How long the planner may take.
pub const PLAN_DEADLINE: Duration = Duration::from_secs(300);

/// The most tokens the plan may run to.
pub const PLAN_MAX_OUTPUT_TOKENS: u64 = 1024;

/// The fewest sections that read as advice a survey must find for the plan
/// to teach advice.
pub const MIN_ADVICE_SECTIONS: usize = 8;

/// The fewest sections in which the writer judges - advises or states a
/// position - a survey must find for the plan to teach a conversation.
pub const MIN_JUDGMENT_SECTIONS: usize = 8;

/// The most kinds one plan may choose.
pub const MAX_KINDS: usize = 4;

/// The longest persona a plan may name, in characters.
const MAX_PERSONA_CHARS: usize = 80;

/// The task kinds a plan may choose, and what each teaches: the planner's
/// menu.
pub const MENU: [(&str, &str); 5] = [
    (
        "recall",
        "facts the writing states, asked as questions with short exact answers",
    ),
    (
        "advise",
        "the writer's own advice to a correspondent, asked as a predicament put to the writer \
         and answered by quoting the writer's own words; only where the writing holds advice",
    ),
    (
        "converse",
        "a conversation with the writer: someone opens with a matter the writing bears on and \
         the writer answers, from the writing and without inventing a number, name or quotation \
         it does not give; teaches how the writer talks and reasons",
    ),
    (
        "explain",
        "why or how something the writing describes works",
    ),
    (
        "denoise",
        "restoring a corrupted passage of the writing, word for word",
    ),
];

/// How to learn from the sources.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(crate = "splinter_agent::schemars")]
pub struct Plan {
    /// Who the learner is becoming, when the goal is to think like a person.
    pub persona: Option<String>,
    /// The kinds of task to generate, from the menu.
    pub kinds: Vec<String>,
    /// Whether to skip the student's own attempts and train on a teacher's
    /// verified answers: right when the tasks ask for passages the student
    /// could not know.
    pub distill: bool,
    /// Why, in a sentence.
    pub rationale: String,
}

/// What the planner is given.
#[derive(Serialize)]
struct Brief<'a> {
    goal: Option<&'a str>,
    sources: &'a Survey,
}

const ROLE: &str = "You plan how a learning system should learn from a body of writing: you \
choose which kinds of task to generate from it.";

/// Why `plan` breaks a rule, or `Ok` when it does not.
fn check(plan: &Plan, survey: &Survey) -> Result<(), String> {
    if plan.kinds.is_empty() || plan.kinds.len() > MAX_KINDS {
        return Err(format!("choose between one and {MAX_KINDS} kinds"));
    }
    for (at, kind) in plan.kinds.iter().enumerate() {
        if !MENU.iter().any(|(name, _)| name == kind) {
            let names: Vec<&str> = MENU.iter().map(|(name, _)| *name).collect();
            return Err(format!(
                "{kind:?} is not on the menu; the kinds are {names:?}"
            ));
        }
        if plan.kinds[..at].contains(kind) {
            return Err(format!("{kind:?} is chosen twice"));
        }
    }
    if plan.kinds.iter().any(|k| k == "advise") && survey.advice_sections < MIN_ADVICE_SECTIONS {
        return Err(format!(
            "the survey found {} section(s) that read as advice and at least {MIN_ADVICE_SECTIONS} \
             are needed to teach advice; choose other kinds",
            survey.advice_sections
        ));
    }
    if plan.kinds.iter().any(|k| k == "converse")
        && survey.judgment_sections < MIN_JUDGMENT_SECTIONS
    {
        return Err(format!(
            "the survey found {} section(s) in which the writer judges and at least \
             {MIN_JUDGMENT_SECTIONS} are needed to teach a conversation; choose other kinds",
            survey.judgment_sections
        ));
    }
    if plan.rationale.trim().is_empty() {
        return Err("give a rationale".into());
    }
    if plan
        .persona
        .as_ref()
        .is_some_and(|p| p.trim().is_empty() || p.chars().count() > MAX_PERSONA_CHARS)
    {
        return Err(format!(
            "the persona is a name of at most {MAX_PERSONA_CHARS} characters, or null"
        ));
    }
    Ok(())
}

/// The prompt's account of the menu.
fn task() -> String {
    let mut task = String::from(
        "Choose how to learn from these sources, given the goal. Choose from these kinds of task \
         only, and only those that serve the goal and that the sources can support:",
    );
    for (name, meaning) in MENU {
        task.push_str(&format!("\n- {name}: {meaning}"));
    }
    task.push_str(
        "\nSet distill to true when the tasks ask for passages the learner could not know without \
         the sources, so it should be taught from a teacher's verified answers and not from its \
         own attempts. Name the persona when the goal is to think like a person, else null.",
    );
    task
}

/// The planner `planner`'s plan for learning from the sources `survey`
/// describes, toward `goal`; see the module documentation.
pub fn plan(
    ctx: &Context,
    survey: &Survey,
    goal: Option<&str>,
    planner: &ModelRef,
    cancel: &CancelToken,
) -> Result<Plan, OrchestratorError> {
    let model = ctx.model(planner)?;
    let facts = Arc::new(survey.clone());
    let call = TypedCall::<Plan>::new("plan_learning", task(), ROLE, PLAN_DEADLINE)
        .max_output_tokens(PLAN_MAX_OUTPUT_TOKENS)
        .cancel(cancel.clone())
        .postcondition(move |chosen: &Plan| check(chosen, &facts));
    ctx.block_on(call.run(
        &model,
        &Brief {
            goal,
            sources: survey,
        },
    ))
    .map_err(|source| OrchestratorError::Call {
        method: "plan_learning",
        source,
    })
}
