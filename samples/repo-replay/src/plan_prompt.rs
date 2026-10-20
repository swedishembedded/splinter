// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! PROMPT 1 of 3: the plan prompt.
//!
//! Who sees it: the planning model, once per (commit, plan style), with no
//! tools. It is never part of a training record.
//!
//! What it must do: turn a commit (message and diff) into the request a user
//! would have written to get that commit made - the task the student sees as
//! its first user turn. It must describe the behaviour wanted and where to
//! look, and must not hand over the change: the plan checker
//! ([`crate::plan::validate`]) rejects verbatim diff lines, diff markers,
//! the commit id and code-dense plans.
//!
//! Two granularities are asked for ([`PlanStyle`]) so a student sees both a
//! terse request and a detailed one for the same change.

use serde::{Deserialize, Serialize};

/// How detailed a plan is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStyle {
    /// A request in one to three sentences, the way a person types it.
    Terse,
    /// A request with goal, motivation, behaviour, areas and constraints.
    Detailed,
}

impl PlanStyle {
    /// The name used on the command line and in records.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Terse => "terse",
            Self::Detailed => "detailed",
        }
    }

    /// The style named `name`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "terse" => Some(Self::Terse),
            "detailed" => Some(Self::Detailed),
            _ => None,
        }
    }
}

/// The system prompt shared by every plan style: what a plan is and is not.
const PLAN_COMMON: &str = "You turn a finished software change into the request a user would \
have written to ask for it. You are shown the change as the author's commit message and a \
unified diff. Write only the request, as the user speaking to a software engineer who has the \
repository but has not seen the change. Rules: describe the behaviour that should be different \
and why it matters; you may name files, modules or functions that exist before the change so the \
engineer knows where to look; do not paste code, diff lines or whole lines from the diff, do not \
quote the commit message, and do not mention a diff, a patch, a commit, a revision or these \
instructions; do not write a code block longer than two lines; do not give step-by-step edit \
instructions or the exact new text of any line. But when the change creates, moves, renames or \
deletes a file, say which one and, for a new or moved file, the exact new path; every name, \
number and piece of text that the change introduces and that the engineer cannot find in the \
repository or work out from your description must be given in the request, as a person who asks \
for something gives them: a new function or option and what it is called, a new limit or default \
and its value, an error message and its words. Leave nothing for the engineer \
to invent. Write plain prose.";

/// Examples of requests as a person writes them, for other changes. They show
/// the manner (intent, behaviour, where in words, constraints, no code); the
/// planner is told not to copy them.
const PLAN_EXAMPLES: &str = "Examples of good requests, for other projects (show the manner; do \
not copy them):\n\
- \"Uploads to the object store give up after three tries, which is too few on a flaky network. \
Make the number of tries configurable in the client's config file and keep three as the default.\"\n\
- \"Pressing escape in the search dialog should close it; today it first clears the text. The \
dialog lives in the ui crate.\"\n\
- \"The export command prints a stack trace when the output directory does not exist. It should \
create the directory, parents included, and carry on. This is in the export module of the cli \
crate. A path that exists but is a file must still fail with the current error message, and no \
new dependency is wanted.\"";

/// The instruction that sets the plan's length and shape.
const PLAN_TERSE: &str = "Style: terse. One to three sentences, as a person types a request in \
a hurry: the goal and the behaviour wanted. Name at most one file or area, and only when the \
request would otherwise be unclear.";

/// The instruction that sets the plan's length and shape.
const PLAN_DETAILED: &str = "Style: detailed. Three short paragraphs at most: the goal and its \
motivation; the behaviour to change and the files or areas involved; the constraints the \
engineer must keep (compatibility, tests, naming conventions the surrounding code follows).";

/// The system prompt of the planning model for `style`.
#[must_use]
pub fn plan_system_prompt(style: PlanStyle) -> String {
    let style_text = match style {
        PlanStyle::Terse => PLAN_TERSE,
        PlanStyle::Detailed => PLAN_DETAILED,
    };
    format!("{PLAN_COMMON}\n\n{style_text}\n\n{PLAN_EXAMPLES}")
}

/// The user turn of the planning model: the change to describe.
#[must_use]
pub fn plan_request(message: &str, diff: &str) -> String {
    format!("Commit message:\n{message}\n\nChange:\n{diff}\n\nWrite the request.")
}

/// The user turn that follows a refused plan: what was wrong with it, so the
/// next try differs from the last (a seeded model asked the same thing says
/// the same thing).
#[must_use]
pub fn plan_refusal_notice(reason: &str) -> String {
    format!(
        "That request cannot be used: {reason}. Write it again as plain prose in the same style, \
         without the thing that was wrong, and without mentioning this correction."
    )
}
