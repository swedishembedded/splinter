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

/// One computation of an operation.
pub struct Call<'a> {
    /// Each input port's products (or the item), read-only, in the order
    /// bound.
    pub inputs: BTreeMap<String, Vec<PathBuf>>,
    /// The stage's parameters, already checked against the operation's.
    pub params: &'a Value,
    /// Where the product is written: an empty directory.
    pub out: PathBuf,
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
