// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that keep what a producer
// recorded intact and addressable, for its clients. If your team needs
// expertise in training-data provenance, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Namespaced records a trajectory's producer put in `extra`.
//!
//! ATIF leaves `extra` to its producer. A producer that records something
//! other consumers must be able to rely on (a gate decision, the memory a
//! step used) files it under `extra.<namespace>.<name>`. Splinter reads such
//! a record as a data contract: it names the path it keeps, copies the value
//! verbatim and addresses it by the digest of its canonical form, and knows
//! nothing of the producer's types.

use atif::Trajectory;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::digest::{canonical_json, Digest};

/// One record found in an `extra` object, kept as it was written.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtraRecord {
    /// The step whose `extra` held it; absent for the trajectory's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_id: Option<u64>,
    /// Where it sat under `extra`, as `<namespace>.<name>`.
    pub path: String,
    /// The content address of `record`'s canonical form.
    pub digest: Digest,
    /// The value, verbatim.
    pub record: Value,
}

// A JSON value holds no NaN, so its equality is reflexive.
impl Eq for ExtraRecord {}

/// Every record at one of `paths` (each `<namespace>.<name>`) in the
/// trajectory's `extra` and in its steps', in document order.
///
/// A record the canonical form cannot hold (a float JSON cannot write) is
/// reported rather than skipped: dropping it would be the silent loss this
/// exists to prevent.
pub fn extra_records(
    trajectory: &Trajectory,
    paths: &[&str],
) -> Result<Vec<ExtraRecord>, serde_json::Error> {
    let mut found = Vec::new();
    collect(trajectory.extra.as_ref(), None, paths, &mut found)?;
    for step in &trajectory.steps {
        collect(step.extra.as_ref(), Some(step.step_id), paths, &mut found)?;
    }
    Ok(found)
}

fn collect(
    extra: Option<&Value>,
    step_id: Option<u64>,
    paths: &[&str],
    found: &mut Vec<ExtraRecord>,
) -> Result<(), serde_json::Error> {
    let Some(extra) = extra else {
        return Ok(());
    };
    for path in paths {
        let Some((namespace, name)) = path.split_once('.') else {
            continue;
        };
        if let Some(record) = extra.get(namespace).and_then(|n| n.get(name)) {
            found.push(ExtraRecord {
                step_id,
                path: (*path).to_string(),
                digest: Digest::of(&canonical_json(record)?),
                record: record.clone(),
            });
        }
    }
    Ok(())
}
