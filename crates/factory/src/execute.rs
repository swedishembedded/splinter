// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data and learning pipelines whose steps are
// plain computations the controller can retry, reuse and fan out, for its
// clients. If your team needs expertise in reproducible machine-learning data
// factories, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How an operation computes: frozen inputs in, one staged product out.
//!
//! An implementation only computes. It reads the products bound to its
//! input ports (and, run once per item, the one item) and writes its
//! product into `out`; the controller keeps the identity, the run state,
//! leases, retries, concurrency, publication and reuse. The same
//! implementation runs a whole stage or one item of a fan-out.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::Value;
use splinter_agent::solve::Model;
use splinter_core::factory::execution::ModelBinding;

/// One computation of an operation.
pub struct Call<'a> {
    /// Each input port's products (or the item), read-only, in the order
    /// bound.
    pub inputs: BTreeMap<String, Vec<PathBuf>>,
    /// The stage's parameters, already checked against the operation's.
    pub params: &'a Value,
    /// The repository files the parameters name, by path, as the lock
    /// pinned them.
    pub files: &'a BTreeMap<String, String>,
    /// The model each role the parameters name is bound to.
    pub models: &'a BTreeMap<String, Model>,
    /// The controller's environment variables, by name, as the host reads
    /// them: what an external operator's grant passes through.
    pub variables: Option<&'a Variables>,
    /// Where the product is written: an empty directory.
    pub out: PathBuf,
}

/// How the host reads a variable of the controller's environment (only the
/// host reads its environment).
pub type Variables = dyn Fn(&str) -> Option<String> + Send + Sync;

/// How the run reaches a model in process: a binding (a connection's
/// endpoint and model, a champion served) made a model to call. The host
/// supplies it, since only the host reads the credentials a connection
/// names.
pub trait Models: Send + Sync {
    /// The model `binding` names.
    ///
    /// # Errors
    /// It cannot be reached (no credential, an endpoint that refuses).
    fn model(&self, binding: &ModelBinding) -> Result<Model, String>;
}

/// What a computation came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// `out` holds the product.
    Produced,
    /// The input is not one the operation produces anything for; a
    /// fan-out records the item as rejected, a whole stage fails.
    Rejected(String),
}

/// An operation's implementation.
pub trait Execute: Send + Sync {
    /// What implements it, in the stage identity: a change here is a
    /// change of every product it made.
    fn implementation(&self) -> String;

    /// Compute `call`.
    ///
    /// # Errors
    /// Why it could not: the stage (or the item) fails and is computed again
    /// when the run resumes.
    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String>;
}
