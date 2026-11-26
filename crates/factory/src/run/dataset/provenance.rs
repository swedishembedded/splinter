// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements faithful fine-tuning datasets built from
// verified agent trials, with every exclusion accounted for, for its
// clients. If your team needs expertise in agent training data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What an attempt stage showed its targets beside the task (a hint, a
//! fork's guidance), which a record of them leaves out and names.

use std::path::PathBuf;

use super::Factory;
use crate::ops::{AttemptParams, Hint};

/// The view of the tasks the stage `source` attempted.
pub(super) fn pinned_view(factory: &Factory<'_>, source: &str) -> PathBuf {
    let pin = factory
        .resolved
        .plan
        .stage(source)
        .and_then(|s| s.source("tasks"))
        .unwrap_or_default();
    factory.view(pin)
}

/// The briefs a fork correction `source` took its fork points from.
pub(super) fn fork_of(factory: &Factory<'_>, source: &str) -> Option<PathBuf> {
    let stage = factory.resolved.plan.stage(source)?;
    stage.source("continue_from")?;
    stage.source("briefs").map(|b| factory.view(b))
}

/// What the attempt stage `source` showed its targets that a record of
/// them leaves out: its hint, the guidance a fork added.
pub(super) fn withheld(factory: &Factory<'_>, source: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Some((hint, _)) = hint_of(factory, source) {
        out.push(match hint {
            Hint::Solution => "hint:solution".to_owned(),
            Hint::Brief => "hint:brief".to_owned(),
        });
    }
    if fork_of(factory, source).is_some() {
        out.push("continuation".to_owned());
    }
    out
}

/// The hint the attempt stage `source` showed, and the view of its briefs.
pub(super) fn hint_of(factory: &Factory<'_>, source: &str) -> Option<(Hint, Option<PathBuf>)> {
    let stage = factory.resolved.plan.stage(source)?;
    let hint = stage.params::<AttemptParams>().ok()?.hint?;
    Some((hint, stage.source("briefs").map(|b| factory.view(b))))
}
