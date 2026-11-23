// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data and learning pipelines built from typed,
// reusable operations instead of one-off scripts, for its clients. If your
// team needs expertise in reproducible machine-learning data factories, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The operations a recipe may use: each one's contract, how its parameters
//! are checked, the model roles they name, and, for an operation the
//! controller runs as a computation, its implementation. The compiler reads
//! a [`Registry`]; the run computes with the same one.

use std::collections::BTreeMap;

use serde_json::Value;
use splinter_core::factory::operation::OperationDescriptor;

use crate::execute::Execute;

/// An operation as the compiler sees it: its contract, and how to check a
/// stage's parameters.
pub trait OperationSpec: Send + Sync {
    /// Its contract.
    fn descriptor(&self) -> OperationDescriptor;
    /// `params` checked against the operation's own typed parameters.
    ///
    /// # Errors
    /// Why they are refused (an unknown parameter, a missing or ill-typed
    /// one).
    fn check(&self, params: &Value) -> Result<(), String>;
    /// The model roles `params` name.
    fn model_roles(&self, _params: &Value) -> Vec<String> {
        Vec::new()
    }
    /// The repository files `params` name, which the lock pins and the
    /// implementation is given.
    fn files(&self, _params: &Value) -> Vec<String> {
        Vec::new()
    }
    /// Problems with the files `params` name, read (by path).
    fn check_files(&self, _params: &Value, _files: &BTreeMap<String, String>) -> Vec<String> {
        Vec::new()
    }
    /// Its implementation, when the controller runs it as a plain
    /// computation ([`Execute`]); only such an operation runs once per item.
    fn executor(&self) -> Option<&dyn Execute> {
        None
    }
}

/// The operations a recipe may use, by name.
#[derive(Default)]
pub struct Registry {
    operations: BTreeMap<String, Box<dyn OperationSpec>>,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.operations.keys()).finish()
    }
}

impl Registry {
    /// Add `operation`; a second one of the same name replaces the first.
    #[must_use]
    pub fn with(mut self, operation: impl OperationSpec + 'static) -> Self {
        self.operations
            .insert(operation.descriptor().name, Box::new(operation));
        self
    }

    /// The operation named `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn OperationSpec> {
        self.operations.get(name).map(AsRef::as_ref)
    }

    /// Every operation's name.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.operations.keys().map(String::as_str).collect()
    }
}
