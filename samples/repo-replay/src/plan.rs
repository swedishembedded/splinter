// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Writing a commit's task, and checking that the task is not the answer.
//!
//! The plan is the student's first user turn. If it contains the change, a
//! student trained on it learns to copy a request into the tree, not to work
//! from one. [`validate`] is the deterministic gate on a plan: length bounds
//! for its style, no diff marker, no commit id, no phrase that points at the
//! change itself, no line of the diff of [`PlanCheck::min_line_tokens`] words
//! or more repeated verbatim, and a bound on code density (the share of the
//! plan that is a code block or inline code).

use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::chat::{reply, ChatOptions, Turn};
use splinter_sdk::agent::solve::Model;

use crate::plan_prompt::{plan_refusal_notice, plan_request, plan_system_prompt, PlanStyle};

/// What a plan is checked against.
#[derive(Clone, Debug)]
pub struct PlanCheck<'a> {
    /// The change's added and removed lines (without the `+`/`-`).
    pub diff_lines: &'a [String],
    /// The commit's id: a plan must not carry it.
    pub commit: &'a str,
    /// A diff line of at least this many words may not appear in the plan.
    pub min_line_tokens: usize,
}

/// Why a plan was refused.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "class", rename_all = "snake_case")]
pub enum PlanRejection {
    /// Fewer or more words than the style allows.
    #[error("the plan has {words} words; {style} plans have {min} to {max}")]
    Length {
        /// Words in the plan.
        words: usize,
        /// The style's name.
        style: String,
        /// Lower bound.
        min: usize,
        /// Upper bound.
        max: usize,
    },
    /// A line of the diff appears in the plan.
    #[error("the plan repeats a line of the change: {line:?}")]
    VerbatimDiffLine {
        /// The line.
        line: String,
    },
    /// A diff header, hunk marker or the commit id appears in the plan.
    #[error("the plan carries {what}")]
    DiffMarker {
        /// What was found.
        what: String,
    },
    /// The plan speaks of the change as an artifact ("the diff").
    #[error("the plan refers to the change itself: {phrase:?}")]
    RefersToChange {
        /// The phrase.
        phrase: String,
    },
    /// Too much of the plan is code.
    #[error("code is {percent}% of the plan; {style} plans may hold {limit}%")]
    CodeDensity {
        /// Percent of the plan's characters that are code.
        percent: usize,
        /// The style's name.
        style: String,
        /// The limit in percent.
        limit: usize,
    },
}

impl PlanRejection {
    /// The reason class a report counts it under.
    #[must_use]
    pub fn class(&self) -> &'static str {
        match self {
            Self::Length { .. } => "plan_length",
            Self::VerbatimDiffLine { .. } => "plan_verbatim_diff",
            Self::DiffMarker { .. } => "plan_diff_marker",
            Self::RefersToChange { .. } => "plan_refers_to_change",
            Self::CodeDensity { .. } => "plan_code_density",
        }
    }
}

/// Word bounds `(min, max)` of a style.
#[must_use]
pub fn word_bounds(style: PlanStyle) -> (usize, usize) {
    match style {
        PlanStyle::Terse => (4, 120),
        PlanStyle::Detailed => (25, 450),
    }
}

/// The share of a plan's characters that may be code, in percent.
#[must_use]
pub fn density_limit(style: PlanStyle) -> usize {
    match style {
        PlanStyle::Terse => 20,
        PlanStyle::Detailed => 35,
    }
}

fn squeeze(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn words(text: &str) -> usize {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .count()
}

/// Characters inside fenced blocks and inline code spans.
fn code_chars(plan: &str) -> usize {
    let fence =
        Regex::new(r"(?s)```.*?(```|$)").unwrap_or_else(|_| unreachable!("a fixed pattern"));
    fence.find_iter(plan).map(|m| m.as_str().len()).sum()
}

/// The share of `plan`'s characters, in percent, that are in fenced code
/// blocks. Inline names in backticks do not count: a good request names the
/// functions, types and files it is about. Code copied from the change is
/// caught by the verbatim checks, not by this share.
#[must_use]
pub fn code_percent(plan: &str) -> usize {
    code_chars(plan) * 100 / plan.len().max(1)
}

/// Accepts `plan` for a change of `style`, or says why not.
pub fn validate(plan: &str, style: PlanStyle, check: &PlanCheck<'_>) -> Result<(), PlanRejection> {
    let (min, max) = word_bounds(style);
    let count = words(plan);
    if count < min || count > max {
        return Err(PlanRejection::Length {
            words: count,
            style: style.name().into(),
            min,
            max,
        });
    }
    let marker = Regex::new(r"(?m)^(diff --git|@@ -|index [0-9a-f]{7,}\.\.|--- a/|\+\+\+ b/)")
        .unwrap_or_else(|_| unreachable!("a fixed pattern"));
    if let Some(found) = marker.find(plan) {
        return Err(PlanRejection::DiffMarker {
            what: format!("the diff marker {:?}", found.as_str()),
        });
    }
    if check.commit.len() >= 7 && plan.contains(&check.commit[..7]) {
        return Err(PlanRejection::DiffMarker {
            what: "the commit id".into(),
        });
    }
    let refers =
        Regex::new(r"(?i)\b(?:the|this|that|above|given|attached) (?:diff|patch|commit)\b")
            .unwrap_or_else(|_| unreachable!("a fixed pattern"));
    if let Some(found) = refers.find(plan) {
        return Err(PlanRejection::RefersToChange {
            phrase: found.as_str().to_string(),
        });
    }
    let squeezed = squeeze(plan);
    for line in check.diff_lines {
        let line = squeeze(line);
        if words(&line) >= check.min_line_tokens && squeezed.contains(&line) {
            return Err(PlanRejection::VerbatimDiffLine { line });
        }
    }
    // A long span in backticks that is a piece of a changed line is the
    // answer quoted, whatever the share of code.
    let inline = Regex::new(r"`([^`\n]{30,})`").unwrap_or_else(|_| unreachable!("a fixed pattern"));
    for span in inline.captures_iter(plan) {
        let span = squeeze(&span[1]);
        if check.diff_lines.iter().any(|l| squeeze(l).contains(&span)) {
            return Err(PlanRejection::VerbatimDiffLine { line: span });
        }
    }
    let percent = code_percent(plan);
    let limit = density_limit(style);
    if percent > limit {
        return Err(PlanRejection::CodeDensity {
            percent,
            style: style.name().into(),
            limit,
        });
    }
    Ok(())
}

/// The added and removed lines of a unified diff, without the `+`/`-` and
/// without the file headers.
#[must_use]
pub fn changed_lines(diff: &str) -> Vec<String> {
    diff.lines()
        .filter(|l| !l.starts_with("+++") && !l.starts_with("---"))
        .filter_map(|l| l.strip_prefix('+').or_else(|| l.strip_prefix('-')))
        .map(str::to_string)
        .collect()
}

/// Asks `model` for the plan of a change, up to `tries` times, until one
/// passes [`validate`]. Returns the accepted plan, or the last rejection (or
/// the model's failure) as the reason there is none.
pub async fn write_plan(
    model: &Model,
    style: PlanStyle,
    message: &str,
    diff: &str,
    check: &PlanCheck<'_>,
    tries: usize,
    deadline: Duration,
) -> Result<String, PlanFailure> {
    let planner = model.clone().with_system(plan_system_prompt(style));
    let mut turns = vec![Turn::user(plan_request(message, diff))];
    let mut last = None;
    for _ in 0..tries.max(1) {
        let plan = reply(&planner, &turns, &ChatOptions::new(deadline))
            .await
            .map_err(|e| PlanFailure::Model(e.to_string()))?;
        match validate(&plan, style, check) {
            Ok(()) => return Ok(plan),
            Err(rejection) => {
                turns.push(Turn::assistant(plan));
                turns.push(Turn::user(plan_refusal_notice(&rejection.to_string())));
                last = Some(rejection);
            }
        }
    }
    Err(PlanFailure::Rejected(
        last.unwrap_or_else(|| unreachable!("at least one try ran")),
    ))
}

/// Why no plan was had.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlanFailure {
    /// The planning model could not be reached or said nothing.
    #[error("the planning model failed: {0}")]
    Model(String),
    /// Every plan it wrote was refused; the last refusal.
    #[error("every plan was refused: {0}")]
    Rejected(PlanRejection),
}

impl PlanFailure {
    /// The reason class a report counts it under.
    #[must_use]
    pub fn class(&self) -> &'static str {
        match self {
            Self::Model(_) => "plan_model_error",
            Self::Rejected(r) => r.class(),
        }
    }
}
