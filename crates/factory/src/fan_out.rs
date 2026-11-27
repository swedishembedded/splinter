// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data pipelines that fan work out over large
// corpora and survive crashes without redoing finished items, for its
// clients. If your team needs expertise in resumable data factories, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The bounds a recipe puts on a fan-out (`foreach`): how many items run at
//! once (`concurrency`) and how many may fail and be recorded as absent
//! rather than fail the stage (`tolerate_failed_items`), checked when the
//! recipe compiles.

use serde::{Deserialize, Serialize};

use crate::graph::StageSpec;
use crate::registry::OperationSpec;

/// How many items of a fan-out may fail and be recorded as absent: a number
/// of items, or a share of them (a number with a fraction, 0 to 1).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum FailedItems {
    /// At most this many items.
    Count(u64),
    /// At most this share of the items.
    Share(f64),
}

impl FailedItems {
    /// Whether `failed` of `items` items stay within the bound.
    #[must_use]
    pub fn allows(self, failed: usize, items: usize) -> bool {
        match self {
            Self::Count(n) => failed as u64 <= n,
            Self::Share(share) => items > 0 && failed as f64 <= share * items as f64,
        }
    }
}

/// The stage's bound on items at once, and every problem with its fan-out
/// bounds: only a stage that runs once per item has them, only an operation
/// the controller runs as a computation runs once per item, the bound on
/// items at once is a number of workers this machine can hold, and a share
/// of failed items is between 0 and 1.
pub(crate) fn bounds(
    id: &str,
    stage: &StageSpec,
    operation: &dyn OperationSpec,
    fans_out: bool,
) -> (u32, Vec<String>) {
    let mut problems = Vec::new();
    if fans_out && operation.executor().is_none() {
        problems.push(format!(
            "stage {id}: {} does not run once per item; foreach is for an operation that computes one product from its inputs",
            stage.uses
        ));
    }
    let concurrency = match stage.concurrency {
        None => 1,
        Some(_) if !fans_out => {
            problems.push(format!(
                "stage {id}: concurrency bounds a foreach, and the stage has none"
            ));
            1
        }
        Some(0) => {
            problems.push(format!("stage {id}: concurrency is 0; it is at least 1"));
            1
        }
        Some(n) => n,
    };
    match stage.tolerate_failed_items {
        Some(_) if !fans_out => problems.push(format!(
            "stage {id}: tolerate_failed_items bounds a foreach's failed items, and the stage has none"
        )),
        Some(FailedItems::Share(share)) if !(0.0..=1.0).contains(&share) => problems.push(format!(
            "stage {id}: tolerate_failed_items {share} is neither a count nor a share from 0 to 1"
        )),
        _ => {}
    }
    (concurrency, problems)
}
