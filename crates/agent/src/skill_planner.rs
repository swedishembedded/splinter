// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! A planner model asked to propose the skills a documented domain asks of a
//! learner.
//!
//! The call is typed: the reply is read as [`SkillsReply`] and a reply that
//! is not is sent back for correction. The model is shown the learner's
//! objective and numbered sections of the sources, and cites sections by
//! number and quotes them. Nothing it says is trusted: admission by code
//! ([`splinter_knowledge::skills::admit`]) keeps only what the sections bear
//! out.

use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use splinter_knowledge::skills::{Proposed, ProposedSkill, Statement};
use sven_sdk::{CallError, CancelToken};

use crate::solve::Model;
use crate::typed::TypedCall;

/// The characters of one section the planner is shown.
pub const SECTION_EXCERPT_CHARS: usize = 1200;

/// The planner's reply: the skills it proposes.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SkillsReply {
    skills: Vec<SkillReply>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SkillReply {
    /// Lowercase letters, digits and `-`.
    id: String,
    title: String,
    /// Positions of the sections it is documented in.
    sections: Vec<usize>,
    /// Names the sections use that the skill is about.
    #[serde(default)]
    symbols: Vec<String>,
    /// What code must respect, each with the words of a section it comes from.
    #[serde(default)]
    constraints: Vec<StatementReply>,
    /// What code must be shown to do, each with the words of a section it
    /// comes from.
    #[serde(default)]
    requirements: Vec<StatementReply>,
    /// Ids of the skills this one uses.
    #[serde(default)]
    prerequisites: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct StatementReply {
    statement: String,
    /// Words of one of the skill's sections, copied exactly.
    quote: String,
}

impl From<StatementReply> for Statement {
    fn from(s: StatementReply) -> Self {
        Self {
            statement: s.statement,
            quote: s.quote,
        }
    }
}

impl From<SkillsReply> for Proposed {
    fn from(reply: SkillsReply) -> Self {
        Self {
            skills: reply
                .skills
                .into_iter()
                .map(|s| ProposedSkill {
                    id: s.id,
                    title: s.title,
                    sections: s.sections,
                    symbols: s.symbols,
                    constraints: s.constraints.into_iter().map(Into::into).collect(),
                    requirements: s.requirements.into_iter().map(Into::into).collect(),
                    prerequisites: s.prerequisites,
                })
                .collect(),
        }
    }
}

const BRIEF: &str = "You plan what to learn from documentation. You are given a learner's \
    objective and numbered sections of the documentation. Propose the skills the documentation \
    asks of a learner who is to reach the objective. Each skill cites the sections it is \
    documented in by number, names the symbols (functions, keywords) the sections use that it is \
    about, states at least one requirement - something code must be shown to do - and any \
    constraints, each with a quote copied exactly from one of its sections, and names the other \
    skills it uses as prerequisites. Do not invent anything the sections do not say.";

#[derive(Serialize)]
struct SectionShown<'a> {
    n: usize,
    text: &'a str,
}

#[derive(Serialize)]
struct PlanInput<'a> {
    objective: &'a str,
    sections: Vec<SectionShown<'a>>,
}

/// One request to the planner: the objective and the sections it may cite.
pub struct SkillPlanRequest<'a> {
    /// What the learner is after.
    pub objective: &'a str,
    /// The sections shown, each with the number it is cited by.
    pub sections: &'a [(usize, String)],
    /// How long the call may take, every correction included.
    pub deadline: Duration,
    /// Output tokens the call may generate.
    pub max_output_tokens: Option<u64>,
    /// Stops the call.
    pub cancel: Option<CancelToken>,
}

/// The first [`SECTION_EXCERPT_CHARS`] characters of `text`, cut at a char
/// boundary.
fn excerpt(text: &str) -> &str {
    match text.char_indices().nth(SECTION_EXCERPT_CHARS) {
        Some((end, _)) => &text[..end],
        None => text,
    }
}

/// The skills `model` proposes for `request`.
pub async fn propose_skills(
    model: &Model,
    request: &SkillPlanRequest<'_>,
) -> Result<Proposed, CallError> {
    let mut call =
        TypedCall::<SkillsReply>::new("plan_skills", BRIEF, "planner", request.deadline).repairs(1);
    call.max_output_tokens = request.max_output_tokens;
    call.cancel = request.cancel.clone();
    let input = PlanInput {
        objective: request.objective,
        sections: request
            .sections
            .iter()
            .map(|(n, text)| SectionShown {
                n: *n,
                text: excerpt(text),
            })
            .collect(),
    };
    call.run(model, &input).await.map(Proposed::from)
}
