// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data and learning pipelines built from typed,
// reusable operations instead of one-off scripts, for its clients. If your
// team needs expertise in reproducible machine-learning data factories, you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a recipe composes: typed artifacts and the operations that turn some
//! into others.
//!
//! An [`ArtifactKind`] says what a product holds (documents, tasks, attempts,
//! a dataset, a model ...); a domain's own schema lives inside the product,
//! never as a new kind. An [`OperationDescriptor`] is an operation's contract:
//! its typed input and output [`Port`]s, the JSON Schema its parameters must
//! satisfy (an unknown parameter is an error), whether it reproduces its
//! output ([`Reproducibility`]), and the least sandbox isolation it needs. A
//! recipe is type-checked against descriptors before anything runs; the
//! controller, never the operation, owns leases, retries and publication.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::execution::Isolation;

/// The schema of an [`OperationDescriptor`].
pub const OPERATION_SCHEMA: &str = "splinter.operation/1";

/// What a product holds.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// Document sections with their source positions.
    Documents,
    /// Concepts, claims, relationships, rules and their evidence.
    Knowledge,
    /// A versioned retrieval index over a pinned collection.
    Index,
    /// Task definitions with their verifiers.
    Tasks,
    /// Reproducible execution environments.
    Environments,
    /// Graded agent attempts with their trajectories.
    Attempts,
    /// Training records with their curation record.
    Dataset,
    /// A checkpoint or adapter, with how it is served.
    Model,
    /// Metrics, verdicts and their evidence.
    Evaluation,
    /// A promotion decision and the model it leaves in use.
    Release,
    /// A Sven harness, compiled and pinned.
    Harness,
    /// A versioned collection of files.
    Files,
}

/// One named input or output of an operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Port {
    /// The port's name in a recipe (`tasks`, `attempts`).
    pub name: String,
    /// What it carries.
    pub kind: ArtifactKind,
    /// Several products of the kind (an input bound to a list).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub many: bool,
    /// An input a stage may leave unbound.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub optional: bool,
}

/// Whether an operation gives the same output for the same identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Reproducibility {
    /// Byte-identical output from the same inputs and parameters.
    Deterministic,
    /// Byte-identical output from the same inputs, parameters and seed.
    Seeded,
    /// Output that may differ when computed again (a model with no pinned
    /// revision, a remote service): its committed product is kept and reused
    /// within a run, never claimed to be reproducible on a fresh one.
    Nondeterministic,
}

/// An operation's contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OperationDescriptor {
    /// Always [`OPERATION_SCHEMA`].
    pub schema: String,
    /// The name a recipe's `uses:` names (`attempt.collect`).
    pub name: String,
    /// Bumped whenever the operation's output for the same inputs changes;
    /// part of every identity it computes.
    pub revision: u32,
    /// What it does, for a person.
    pub summary: String,
    /// Its inputs.
    pub inputs: Vec<Port>,
    /// Its outputs.
    pub outputs: Vec<Port>,
    /// The JSON Schema its parameters (`with:`) satisfy; it refuses unknown
    /// properties.
    pub parameters: Value,
    /// Whether it reproduces its output.
    pub reproducibility: Reproducibility,
    /// The least sandbox isolation its execution needs.
    #[serde(default)]
    pub isolation: Isolation,
}

/// Whether `name` is a plain port or stage name: lower-case letters, digits,
/// `-` and `_`, starting with a letter.
#[must_use]
pub fn plain_name(name: &str) -> bool {
    name.chars().next().is_some_and(|c| c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

impl OperationDescriptor {
    /// The input port named `name`.
    #[must_use]
    pub fn input(&self, name: &str) -> Option<&Port> {
        self.inputs.iter().find(|p| p.name == name)
    }

    /// The output port named `name`.
    #[must_use]
    pub fn output(&self, name: &str) -> Option<&Port> {
        self.outputs.iter().find(|p| p.name == name)
    }

    /// Every problem of the descriptor itself: its schema, a dotted name of
    /// plain parts, plain and unique port names, an output, and a parameter
    /// schema that is an object refusing unknown properties.
    #[must_use]
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if self.schema != OPERATION_SCHEMA {
            problems.push(format!(
                "schema is {:?}; expected {OPERATION_SCHEMA}",
                self.schema
            ));
        }
        if self.name.split('.').any(|part| !plain_name(part)) {
            problems.push(format!(
                "operation name {:?} is not dotted plain names",
                self.name
            ));
        }
        for (side, ports) in [("input", &self.inputs), ("output", &self.outputs)] {
            let mut seen = std::collections::BTreeSet::new();
            for port in ports {
                if !plain_name(&port.name) {
                    problems.push(format!("{side} {:?} is not a plain name", port.name));
                }
                if !seen.insert(port.name.as_str()) {
                    problems.push(format!("{side} {} is declared twice", port.name));
                }
            }
        }
        if self.outputs.is_empty() {
            problems.push(format!("{} produces nothing", self.name));
        }
        if self.outputs.iter().any(|p| p.optional) {
            problems.push("an output is always produced; none is optional".into());
        }
        if self.parameters.get("type") != Some(&Value::from("object"))
            || self.parameters.get("additionalProperties") != Some(&Value::Bool(false))
        {
            problems.push(format!(
                "{}'s parameter schema must be an object with additionalProperties: false",
                self.name
            ));
        }
        problems
    }
}
