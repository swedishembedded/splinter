// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements versioned, machine-checked interfaces
// between the stages of a learning pipeline for its clients. If your team
// needs expertise in data contracts, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The JSON Schema of every factory contract, generated from the Rust types
//! so the two cannot drift. The repository keeps them under
//! `schemas/factory/` for tools that do not link Splinter; a test fails when
//! a committed schema differs from the generated one.

use schemars::{schema_for, Schema};

use super::attempt::{AttemptReport, ATTEMPT_SCHEMA};
use super::execution::{ExecutionSpec, EXECUTION_SCHEMA};
use super::identity::{StageIdentity, STAGE_SCHEMA};
use super::items::{ItemIndex, ITEMS_SCHEMA};
use super::operation::{OperationDescriptor, OPERATION_SCHEMA};
use super::product::{
    DatasetProduct, EvalResult, ModelCandidate, TrialResult, CANDIDATE_SCHEMA, DATASET_SCHEMA,
    EVAL_SCHEMA, TRIAL_SCHEMA,
};
use super::task::{TaskCollection, TASKS_SCHEMA};

/// Every contract's schema name and JSON Schema, in a fixed order.
#[must_use]
pub fn all() -> Vec<(&'static str, Schema)> {
    vec![
        (STAGE_SCHEMA, schema_for!(StageIdentity)),
        (TASKS_SCHEMA, schema_for!(TaskCollection)),
        (EXECUTION_SCHEMA, schema_for!(ExecutionSpec)),
        (ATTEMPT_SCHEMA, schema_for!(AttemptReport)),
        (TRIAL_SCHEMA, schema_for!(TrialResult)),
        (DATASET_SCHEMA, schema_for!(DatasetProduct)),
        (CANDIDATE_SCHEMA, schema_for!(ModelCandidate)),
        (EVAL_SCHEMA, schema_for!(EvalResult)),
        (OPERATION_SCHEMA, schema_for!(OperationDescriptor)),
        (ITEMS_SCHEMA, schema_for!(ItemIndex)),
    ]
}

/// The file a schema is kept in: `splinter.trial/1` -> `splinter.trial.v1.schema.json`.
#[must_use]
pub fn file_name(schema: &str) -> String {
    format!("{}.schema.json", schema.replace('/', ".v"))
}
