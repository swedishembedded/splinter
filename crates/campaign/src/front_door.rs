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
//! call (a typed call returning [`Classification`], bounded by
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
use splinter_agent::schemars::JsonSchema;
use splinter_agent::typed::TypedCall;
use splinter_agent::CallError;

use crate::context::Context;
use crate::error::CampaignError;
use crate::sources::{SourceTarget, COMMAND_PREFIX};
use splinter_core::model_ref::ModelRef;

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
#[schemars(crate = "splinter_agent::schemars")]
#[serde(tag = "verb", rename_all = "snake_case")]
pub enum Intent {
    /// Learn from sources: files, directories, or `cmd:<command>` whose
    /// output is captured; runs the whole pipeline.
    Learn {
        /// What to learn from, exactly as the sentence names it; a
        /// command-line tool is learned from its help, `cmd:<tool> --help`.
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
#[schemars(crate = "splinter_agent::schemars")]
pub struct Candidate {
    /// What the sentence asks for, read this way.
    pub intent: Intent,
    /// How likely this reading is, from 0 to 1.
    pub confidence: f64,
}

/// Every plausible reading of a sentence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(crate = "splinter_agent::schemars")]
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

/// Sentences shown to the model with their classifications: a small
/// model copies an example far more reliably than it reads a schema's
/// references.
const EXAMPLES: [(&str, &str); 4] = [
    (
        "Learn everything in the manual at ./manuals/pump.md",
        r#"{"candidates": [{"intent": {"verb": "learn", "sources": ["./manuals/pump.md"], "goal": "everything in the manual"}, "confidence": 0.9}]}"#,
    ),
    (
        "Learn what the tar command line can do",
        r#"{"candidates": [{"intent": {"verb": "learn", "sources": ["cmd:tar --help"], "goal": "what the tar command line can do"}, "confidence": 0.9}]}"#,
    ),
    (
        "Learn to think like Ada Lovelace based on the letters she wrote in directory ./ada",
        r#"{"candidates": [{"intent": {"verb": "learn", "sources": ["./ada"], "goal": "think like Ada Lovelace"}, "confidence": 0.9}]}"#,
    ),
    (
        "Explain the tar flags from what you know",
        r#"{"candidates": [{"intent": {"verb": "ask", "question": "Explain the tar flags"}, "confidence": 0.9}]}"#,
    ),
];

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
///
/// A reply the policy could not shape into a [`Classification`], even
/// after sven's corrections, is no reading: it is asked back, as a
/// sentence that matches no command is, with what was wrong with it.
pub fn interpret(ctx: &Context, sentence: &str) -> Result<Routed, CampaignError> {
    match classify(ctx, sentence) {
        Ok(mut classification) => {
            repair_paths(&mut classification, sentence);
            Ok(route(&classification, ctx.allow_remote()))
        }
        Err(CampaignError::Call {
            source: CallError::Invalid { detail, .. } | CallError::Postcondition { detail, .. },
            ..
        }) => Ok(Routed::Clarify(Clarification {
            reason: format!(
                "the sentence could not be read as a command ({detail}); name one \
                 (splinter --help)"
            ),
            candidates: Vec::new(),
        })),
        Err(e) => Err(e),
    }
}

/// Corrects, by code, a path the model copied wrongly: a source that is
/// no file or directory, no `cmd:` and no stored id is replaced by the one
/// path the sentence names that exists. Nothing is invented: with no such
/// path, or more than one, the reading keeps what the model wrote and is
/// refused later where the path is read.
fn repair_paths(classification: &mut Classification, sentence: &str) {
    let named: Vec<String> = sentence
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| {
                matches!(c, '"' | '\'' | ',' | ';' | ':' | '!' | '?' | '(' | ')')
            })
            .trim_end_matches('.')
            .to_string()
        })
        .filter(|word| word.contains('/') && std::path::Path::new(word).exists())
        .collect();
    let [only] = named.as_slice() else { return };
    // Wrong is what the source reader would refuse, except a URL: that is
    // refused as a URL, not mended into some other path.
    let wrong = |path: &str| !path.contains("://") && SourceTarget::from_learn_arg(path).is_err();
    for candidate in &mut classification.candidates {
        match &mut candidate.intent {
            Intent::Learn { sources, .. } => {
                for source in sources.iter_mut().filter(|s| wrong(s)) {
                    source.clone_from(only);
                }
            }
            Intent::AddSource { target } if wrong(target) => target.clone_from(only),
            _ => {}
        }
    }
}

/// The readings of `sentence` by the model that reads sentences: the policy,
/// or the model the configuration names for the front door.
pub fn classify(ctx: &Context, sentence: &str) -> Result<Classification, CampaignError> {
    let reader = match &ctx.config().front_door_model {
        Some(text) => text.parse::<ModelRef>().map_err(|e| {
            CampaignError::Refused(format!(
                "the front door model {text:?} is not a model reference: {e}"
            ))
        })?,
        None => ModelRef::policy_default(),
    };
    let model = ctx.model(&reader)?;
    let mut task = format!("{TASK}\n\nFor example:");
    for (example, classified) in EXAMPLES {
        task.push_str(&format!("\n{example:?} is classified as {classified}"));
    }
    let call = TypedCall::<Classification>::new("classify_sentence", task, ROLE, CLASSIFY_DEADLINE)
        .max_output_tokens(CLASSIFY_MAX_OUTPUT_TOKENS)
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
    ctx.block_on(call.run(&model, &Sentence { sentence }))
        .map_err(|source| CampaignError::Call {
            method: "classify_sentence",
            source,
        })
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
                    splinter_core::model_ref::RefError::RemoteNotAllowed {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_examples_shown_to_the_model_are_classifications() {
        for (_, classified) in EXAMPLES {
            let example: Result<Classification, _> = serde_json::from_str(classified);
            assert!(
                example.is_ok_and(|c| c.candidates.len() == 1),
                "{classified}"
            );
        }
    }
}
