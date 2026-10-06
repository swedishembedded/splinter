// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training requests that say what a writer's
// passage is about without handing the model its words, for its clients. If
// your team needs expertise in building fine-tuning data from a person's
// own writing, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The content descriptions the voice view asks passages by.
//!
//! A passage of the writer's text is the answer to a request that says what
//! it is about. The generator model writes that account, 20 to 60 words, in
//! its own words, shown the passage and nothing else; code admits it only if
//! it does not carry the passage's phrasing
//! ([`splinter_data::description::admit`]). The model never writes persona
//! prose: it describes, the writer's text answers. A passage it cannot
//! describe within the bounds keeps the request code writes from the
//! passage's heading and opening.
//!
//! Descriptions are kept in a file per generator and prompt version, one
//! line each, appended as they are admitted, so a rerun - or the other arm
//! of a comparison - reuses what was written and a crash loses at most one.

use std::collections::{BTreeMap, HashMap};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use splinter_agent::schemars::JsonSchema;
use splinter_agent::typed::TypedCall;
use splinter_agent::{CallError, CancelToken};
use splinter_core::digest::Digest;
use splinter_core::model_ref::ModelRef;
use splinter_data::description::{admit, MAX_WORDS, MIN_WORDS};
use splinter_knowledge::tasks::DEFAULT_REQUEST_DEADLINE;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};

use crate::tasks::remaining;

/// Bumped whenever the brief changes: descriptions written under another
/// brief are not reused.
pub const VERSION: &str = "1";

const ROLE: &str = "You summarise what passages are about.";

const TASK: &str = "Say what the passage is about, for someone who is to write it and has not \
read it: its subject, the occasion or person it concerns, and the position or conclusion it \
reaches. Use your own words. Never quote the passage or reuse its phrases; do not name the \
author or open with 'this passage'. Write one or two plain sentences of 20 to 60 words.";

/// How many times a description that does not stand is sent back.
const REPAIRS: u32 = 2;

/// The most tokens one description may take to write.
const MAX_OUTPUT_TOKENS: u64 = 256;

/// What the model is shown.
#[derive(Serialize)]
struct Shown<'a> {
    passage: &'a str,
}

/// What the model answers.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "splinter_agent::schemars")]
#[serde(deny_unknown_fields)]
struct Said {
    /// The description of the passage.
    description: String,
}

/// One round of describing.
#[derive(Clone, Debug, Serialize)]
pub struct DescribeRequest {
    /// The model that writes the descriptions.
    pub generator: ModelRef,
    /// No description is begun after this.
    #[serde(skip)]
    pub deadline: Option<Instant>,
    /// Stops the stage.
    #[serde(skip)]
    pub cancel: CancelToken,
}

/// What a round reports.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Described {
    /// Passages asked for descriptions of.
    pub passages: usize,
    /// Of those, already written by an earlier round.
    pub reused: usize,
    /// Written and admitted in this round.
    pub written: usize,
    /// Passages whose model could not give an admitted description, by
    /// whether it replied in the shape asked: those keep the request code
    /// writes from their heading and opening.
    pub rejected: BTreeMap<String, usize>,
    /// Requests stopped by their own deadline before any reply.
    pub failed: usize,
    /// Why some passages were not covered, if so.
    pub stopped: Option<String>,
}

/// One line of the file of descriptions.
#[derive(Serialize, Deserialize)]
struct Line {
    passage: Digest,
    description: String,
}

/// The descriptions of `passages`, by the digest of the passage's text: the
/// ones already written, and the ones `request.generator` writes now.
pub fn describe(
    ctx: &Context,
    request: &DescribeRequest,
    passages: &[&str],
) -> Result<(HashMap<Digest, String>, Described), OrchestratorError> {
    let model = ctx.model(&request.generator)?;
    let path = {
        let dir = ctx.root().work().join("describe");
        std::fs::create_dir_all(&dir).map_err(io(&dir))?;
        let who = Digest::of(model.identity.as_bytes());
        dir.join(format!("v{VERSION}-{}.jsonl", &who.hex()[..16]))
    };
    let mut known: HashMap<Digest, String> = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(&path) {
        // A line cut short by a crash is not one.
        known.extend(
            text.lines()
                .filter_map(|l| serde_json::from_str::<Line>(l).ok())
                .map(|l| (l.passage, l.description)),
        );
    }
    let mut report = Described {
        passages: passages.len(),
        ..Described::default()
    };
    let mut out: HashMap<Digest, String> = HashMap::new();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(io(&path))?;
    for passage in passages {
        let id = Digest::of(passage.as_bytes());
        if out.contains_key(&id) {
            continue;
        }
        // A description kept under an older admission is judged again: the
        // file is a cache of what a model wrote, not of what stood.
        if let Some(description) = known.get(&id).filter(|d| admit(d, passage).is_ok()) {
            out.insert(id, description.clone());
            report.reused += 1;
            continue;
        }
        if request.cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
        }
        if request.deadline.is_some_and(|d| Instant::now() >= d) {
            report.stopped = Some("the budget was spent before every passage was described".into());
            break;
        }
        let owned = (*passage).to_string();
        let call = TypedCall::<Said>::new(
            "describe_passage",
            TASK,
            ROLE,
            remaining(request.deadline, DEFAULT_REQUEST_DEADLINE),
        )
        .max_output_tokens(MAX_OUTPUT_TOKENS)
        .repairs(REPAIRS)
        .cancel(request.cancel.clone())
        .postcondition(move |said: &Said| {
            admit(&said.description, &owned).map_err(|why| match why {
                splinter_data::description::Rejection::TooShort => {
                    format!("write at least {MIN_WORDS} words")
                }
                splinter_data::description::Rejection::TooLong => {
                    format!("write at most {MAX_WORDS} words")
                }
                _ => "put it in other words: no phrase of the passage may be repeated".into(),
            })
        });
        match ctx.block_on(call.run(&model, &Shown { passage })) {
            Ok(said) => {
                let description = said.description.trim().to_string();
                let line = serde_json::to_string(&Line {
                    passage: id.clone(),
                    description: description.clone(),
                })
                .map_err(|e| OrchestratorError::Refused(format!("a description: {e}")))?;
                writeln!(file, "{line}").map_err(io(&path))?;
                file.flush().map_err(io(&path))?;
                out.insert(id, description);
                report.written += 1;
            }
            Err(CallError::Postcondition { .. }) => {
                *report.rejected.entry("not_admitted".into()).or_default() += 1;
            }
            Err(CallError::Invalid { .. }) => {
                *report.rejected.entry("malformed".into()).or_default() += 1;
            }
            Err(CallError::Stopped { .. }) => {
                if request.cancel.is_cancelled() {
                    return Err(OrchestratorError::Cancelled);
                }
                report.failed += 1;
            }
            Err(e) => {
                return Err(OrchestratorError::Refused(format!(
                    "asking {} to describe a passage: {e}",
                    model.identity
                )))
            }
        }
    }
    Ok((out, report))
}
