// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment planners that show the work an
// experiment will do before it spends anything, for its clients. If your
// team needs expertise in machine-learning experiment orchestration, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A compiled recipe as the controller runs it: each stage with the stages
//! it consumes, its bound inputs and parameters, and the work it will do.
//!
//! The graph is the compiled recipe's ([`crate::graph`]): stages are listed
//! in an order in which each comes after all it consumes, and a stage's key
//! in the run (its product's view, its slots' namespace) is its id. Their
//! identities are computed when they run, from the products of what they
//! consume.

use std::collections::{BTreeMap, BTreeSet};

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use crate::graph::Source;

/// One planned stage.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlannedStage {
    /// Its id in the recipe, and its key in the run.
    pub key: String,
    /// The operation it runs.
    pub operation: String,
    /// The operation's revision.
    pub revision: u32,
    /// The keys of the stages it consumes.
    pub depends_on: Vec<String>,
    /// What it will do, for a person.
    pub work: String,
    /// Each input port's sources.
    #[serde(skip)]
    pub inputs: BTreeMap<String, Vec<Source>>,
    /// Its parameters, checked against the operation's when compiled.
    #[serde(skip)]
    pub params: Value,
    /// The collection it runs once per item of.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub foreach: Option<Source>,
    /// How many of its items run at once.
    #[serde(skip)]
    pub concurrency: u32,
}

impl PlannedStage {
    /// The stages feeding `port`, in the order bound.
    #[must_use]
    pub fn sources(&self, port: &str) -> Vec<&str> {
        self.inputs
            .get(port)
            .into_iter()
            .flatten()
            .filter_map(|s| match s {
                Source::Stage { stage, .. } => Some(stage.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The one stage feeding `port`.
    #[must_use]
    pub fn source(&self, port: &str) -> Option<&str> {
        self.sources(port).into_iter().next()
    }

    /// The recipe input bound to `port`, when it is one.
    #[must_use]
    pub fn input(&self, port: &str) -> Option<&str> {
        self.inputs_of(port).into_iter().next()
    }

    /// Every recipe input bound to `port`, in the order bound.
    #[must_use]
    pub fn inputs_of(&self, port: &str) -> Vec<&str> {
        self.inputs
            .get(port)
            .into_iter()
            .flatten()
            .filter_map(|s| match s {
                Source::Input(name) => Some(name.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Its parameters as the operation's type.
    ///
    /// # Errors
    /// They do not fit it (refused when compiled; kept as an error here).
    pub fn params<P: DeserializeOwned>(&self) -> Result<P, String> {
        crate::ops::params(&self.params).map_err(|e| format!("stage {}: {e}", self.key))
    }

    /// The operation with its revision, as an identity names it.
    #[must_use]
    pub fn versioned_operation(&self) -> String {
        format!("{}@{}", self.operation, self.revision)
    }
}

/// The planned graph, in dependency order.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Plan {
    /// The stages.
    pub stages: Vec<PlannedStage>,
}

impl Plan {
    /// The stage with `key`.
    #[must_use]
    pub fn stage(&self, key: &str) -> Option<&PlannedStage> {
        self.stages.iter().find(|s| s.key == key)
    }

    /// The stages running `operation`, in order.
    pub fn of_operation<'p>(
        &'p self,
        operation: &'p str,
    ) -> impl Iterator<Item = &'p PlannedStage> {
        self.stages.iter().filter(move |s| s.operation == operation)
    }

    /// Whether every stage comes after all it depends on.
    #[must_use]
    pub fn is_ordered(&self) -> bool {
        let mut seen = BTreeSet::new();
        self.stages.iter().all(|stage| {
            let ok = stage.depends_on.iter().all(|d| seen.contains(d.as_str()));
            seen.insert(stage.key.as_str());
            ok
        })
    }

    /// Every stage that depends on `key`, directly or not.
    #[must_use]
    pub fn downstream(&self, key: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut frontier = vec![key.to_owned()];
        while let Some(at) = frontier.pop() {
            for stage in &self.stages {
                if stage.depends_on.contains(&at) && out.insert(stage.key.clone()) {
                    frontier.push(stage.key.clone());
                }
            }
        }
        out
    }
}
