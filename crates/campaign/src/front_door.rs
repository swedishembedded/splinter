// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements plain-language front ends for agent
// systems where the model interprets and code decides, for its clients. If
// your team needs expertise in typed model interfaces, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The front door: a sentence becomes one of the commands.
//!
//! The policy model classifies the sentence through sven's typed method
//! call ([`sven_sdk::Method`] returning [`Classification`], bounded by
//! [`CLASSIFY_DEADLINE`] and [`CLASSIFY_MAX_OUTPUT_TOKENS`]): candidate
//! [`Intent`]s, each with its arguments and a confidence. What happens next
//! is decided here, by [`route`], never by the model:
//!
//! 1. the candidates are ranked by confidence; none at all is a question;
//! 2. the top one is **ambiguous** - a question back, listing the
//!    candidates as commands - when its confidence is below
//!    [`MIN_CONFIDENCE`], or a different intent is within [`MIN_MARGIN`]
//!    of it;
//! 3. a model reference in it that does not parse, or that needs the
//!    network without the opt-in, is **refused**;
//! 4. an intent that is destructive (cancelling a run), runs a command
//!    (a `cmd:` source) or opts into the network (a remote model) is
//!    **confirmed**, never run on a guess; so is one missing what it needs;
//! 5. anything else **runs**, as exactly the command
//!    [`Intent::command_line`] names.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_agent::solve::SolveOptions;
use sven_sdk::schemars::JsonSchema;
use sven_sdk::{Engine, Method, Toolset};

use crate::context::Context;
use crate::error::CampaignError;
use crate::model_ref::ModelRef;
use crate::sources::COMMAND_PREFIX;

/// The confidence below which the top reading is a question.
pub const MIN_CONFIDENCE: f64 = 0.7;

/// How far ahead of every other reading the top one must be.
pub const MIN_MARGIN: f64 = 0.2;

/// How long classifying one sentence may take.
pub const CLASSIFY_DEADLINE: Duration = Duration::from_secs(120);

/// Output tokens classifying one sentence may spend, repairs included.
pub const CLASSIFY_MAX_OUTPUT_TOKENS: u64 = 1024;

/// What a sentence asks Splinter to do.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(crate = "sven_sdk::schemars")]
#[serde(tag = "verb", rename_all = "snake_case")]
pub enum Intent {
    /// Learn from sources: files, directories, or `cmd:<command>` whose
    /// output is captured; runs the whole pipeline.
    Learn {
        /// What to learn from, exactly as the sentence names it.
        sources: Vec<String>,
        /// What the learner is after, if the sentence says.
        #[serde(default)]
        goal: Option<String>,
    },
    /// Answer a question with the current policy, from what it knows.
    Ask {
        /// The question, as the sentence asks it.
        question: String,
        /// A model reference, only when the sentence names a model.
        #[serde(default)]
        policy: Option<String>,
    },
    /// Show the policy in use, recent runs and what the stores hold.
    Status,
    /// Store a source without learning from it yet.
    AddSource {
        /// A file, a directory, or `cmd:<command>`.
        target: String,
    },
    /// List the stored sources.
    ListSources,
    /// List the recorded runs.
    ListRuns,
    /// Stop a run in progress.
    CancelRun {
        /// The run's id.
        run: String,
    },
}

/// One reading of a sentence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(crate = "sven_sdk::schemars")]
pub struct Candidate {
    /// What the sentence asks for, read this way.
    pub intent: Intent,
    /// How likely this reading is, from 0 to 1.
    pub confidence: f64,
}

/// Every plausible reading of a sentence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(crate = "sven_sdk::schemars")]
pub struct Classification {
    /// The readings, most likely first; empty when none fits.
    pub candidates: Vec<Candidate>,
}

/// What the model is given.
#[derive(Serialize)]
struct Sentence<'a> {
    sentence: &'a str,
}

/// What the model is told to do.
const TASK: &str = "Classify the sentence as one of Splinter's commands. Give every reading \
that fits, most likely first, each with its arguments taken from the sentence and a confidence \
from 0 to 1. Keep paths, commands and questions exactly as written. Give no candidates when \
nothing fits, and lower confidences when the sentence could mean several things.";

/// Who the model is while it classifies.
const ROLE: &str = "You route a person's sentence to the command of a learning system that \
acquires knowledge from documents, repositories and command output.";

/// What the front door does with a sentence.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "routed", rename_all = "snake_case")]
pub enum Routed {
    /// Run this intent, as its command line.
    Execute(Intent),
    /// Ask the person first.
    Clarify(Clarification),
    /// Refuse, saying why.
    Refuse(String),
}

/// A question back to the person.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Clarification {
    /// Why the front door is asking.
    pub reason: String,
    /// The readings to choose from, most likely first.
    pub candidates: Vec<Proposal>,
}

/// One reading, as the command it would run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Proposal {
    /// The reading.
    pub intent: Intent,
    /// Its confidence.
    pub confidence: f64,
    /// The command it runs, as arguments after `splinter`.
    pub command: Vec<String>,
}

impl From<&Candidate> for Proposal {
    fn from(candidate: &Candidate) -> Self {
        Self {
            intent: candidate.intent.clone(),
            confidence: candidate.confidence,
            command: candidate.intent.command_line(),
        }
    }
}

impl Intent {
    /// The command this intent runs, as arguments after `splinter`.
    #[must_use]
    pub fn command_line(&self) -> Vec<String> {
        let words = |words: &[&str]| words.iter().map(|w| w.to_string()).collect::<Vec<_>>();
        match self {
            Self::Learn { sources, goal } => {
                let mut line = words(&["learn"]);
                line.extend(sources.iter().cloned());
                if let Some(goal) = goal {
                    line.extend([String::from("--goal"), goal.clone()]);
                }
                line
            }
            Self::Ask { question, policy } => {
                let mut line = words(&["ask", question]);
                if let Some(policy) = policy {
                    line.extend([String::from("--policy"), policy.clone()]);
                }
                line
            }
            Self::Status => words(&["status"]),
            Self::AddSource { target } => {
                let mut line = words(&["source", "add"]);
                if target.starts_with(COMMAND_PREFIX) {
                    line.extend(target.split_whitespace().map(str::to_string));
                } else {
                    line.push(target.clone());
                }
                line
            }
            Self::ListSources => words(&["source", "list"]),
            Self::ListRuns => words(&["runs", "list"]),
            Self::CancelRun { run } => words(&["runs", "cancel", run]),
        }
    }

    /// Why this intent must be confirmed before it runs, if it must.
    fn needs_confirmation(&self, remote: bool) -> Option<String> {
        let runs_command = |target: &str| target.starts_with(COMMAND_PREFIX);
        match self {
            Self::CancelRun { .. } => Some("cancelling a run cannot be undone".into()),
            Self::AddSource { target } if runs_command(target) => {
                Some("capturing a command runs it".into())
            }
            Self::Learn { sources, .. } if sources.iter().any(|s| runs_command(s)) => {
                Some("learning from a command runs it".into())
            }
            Self::Learn { sources, .. } if sources.is_empty() => {
                Some("the sentence names nothing to learn from".into())
            }
            Self::Ask { question, .. } if question.trim().is_empty() => {
                Some("the sentence asks no question".into())
            }
            Self::AddSource { target } if target.trim().is_empty() => {
                Some("the sentence names no source".into())
            }
            Self::Ask { .. } if remote => Some("the question would go over the network".into()),
            _ => None,
        }
    }
}

/// Classifies `sentence` with the policy model and routes the result.
pub fn interpret(ctx: &Context, sentence: &str) -> Result<Routed, CampaignError> {
    Ok(route(&classify(ctx, sentence)?, ctx.allow_remote()))
}

/// The policy model's readings of `sentence`.
pub fn classify(ctx: &Context, sentence: &str) -> Result<Classification, CampaignError> {
    let model = ctx.model(&ModelRef::policy_default())?;
    let method = Method::<Classification>::new("classify_sentence")
        .role(ROLE)
        .task(TASK)
        .postcondition(|c: &Classification| {
            if c.candidates
                .iter()
                .all(|c| (0.0..=1.0).contains(&c.confidence))
            {
                Ok(())
            } else {
                Err("every confidence must be between 0 and 1".into())
            }
        });
    let call = |source| CampaignError::Call {
        method: "classify_sentence",
        source,
    };
    let mut bounds = SolveOptions::new(CLASSIFY_DEADLINE);
    bounds.max_output_tokens = Some(CLASSIFY_MAX_OUTPUT_TOKENS);
    bounds.stream_idle = model.stream_idle;
    let engine = Engine::builder()
        .config(bounds.engine_config())
        .model_provider(model.provider)
        .toolset(Toolset::none())
        .build()
        .map_err(call)?;
    ctx.block_on(engine.call_with(&method, &Sentence { sentence }, bounds.run_options()))
        .map_err(call)
}

/// What to do with `classification`; see the module documentation.
#[must_use]
pub fn route(classification: &Classification, allow_remote: bool) -> Routed {
    let mut ranked: Vec<&Candidate> = classification.candidates.iter().collect();
    ranked.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let clarify = |reason: String, candidates: &[&Candidate]| {
        Routed::Clarify(Clarification {
            reason,
            candidates: candidates.iter().map(|c| Proposal::from(*c)).collect(),
        })
    };
    let Some(top) = ranked.first() else {
        return clarify(
            "the sentence matches no command; name one (splinter --help)".into(),
            &[],
        );
    };
    if top.confidence < MIN_CONFIDENCE {
        return clarify(
            format!(
                "no reading is confident enough (the best is {:.2}, {MIN_CONFIDENCE} is needed)",
                top.confidence
            ),
            &ranked,
        );
    }
    if ranked[1..]
        .iter()
        .any(|other| other.intent != top.intent && top.confidence - other.confidence < MIN_MARGIN)
    {
        return clarify(
            "the sentence could mean more than one thing".into(),
            &ranked,
        );
    }
    let mut remote = false;
    if let Intent::Ask {
        policy: Some(policy),
        ..
    } = &top.intent
    {
        match policy.parse::<ModelRef>() {
            Err(e) => return Routed::Refuse(e.to_string()),
            Ok(reference) if reference.is_remote() && !allow_remote => {
                return Routed::Refuse(
                    crate::model_ref::RefError::RemoteNotAllowed {
                        reference: reference.to_string(),
                    }
                    .to_string(),
                )
            }
            Ok(reference) => remote = reference.is_remote(),
        }
    }
    match top.intent.needs_confirmation(remote) {
        Some(reason) => clarify(format!("{reason}; confirm by running it"), &ranked[..1]),
        None => Routed::Execute(top.intent.clone()),
    }
}
