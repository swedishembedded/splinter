// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running matched, single-variable
// training experiments whose result can be attributed, for its clients. If
// your team needs expertise in experiment design for model training you can
// procure our services by sending an email to info@swedishembedded.com.

//! Whether an arm is a single-variable change from the champion, and whether
//! an engineering change stays inside the paths it may touch.
//!
//! Both configurations are JSON values flattened to dotted leaf paths
//! ([`leaves`]): an object is walked, an array is one leaf, and a missing leaf
//! equals `null`, so a setting left out and a setting written as its absent
//! value are the same. A changed leaf is one whose value differs
//! ([`changed`]). A path lies under a variable when it is the variable or
//! starts with it and a dot ([`under`]).

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

/// Why an arm is not the experiment it claims to be.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum VariableError {
    /// The arm is the champion's configuration: there is no hypothesis.
    #[error("the arm's configuration is the champion's: it changes nothing, so it tests nothing")]
    NoChange,
    /// The arm changes more than the declared variable.
    #[error(
        "the arm changes {} besides its declared variable {variable:?}: one experiment changes one variable",
        paths.join(", ")
    )]
    Undeclared {
        /// The declared variable.
        variable: String,
        /// Every changed path outside it.
        paths: Vec<String>,
    },
    /// An engineering change reaches outside the engineering paths.
    #[error(
        "an engineering change may touch only {}; it changes {}",
        allowed.join(", "),
        paths.join(", ")
    )]
    Breach {
        /// The paths it may touch.
        allowed: Vec<String>,
        /// Every changed path outside them.
        paths: Vec<String>,
    },
}

/// Every leaf of `value` by its dotted path; `null` leaves are left out.
#[must_use]
pub fn leaves(value: &Value) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    flatten("", value, &mut out);
    out
}

fn flatten(prefix: &str, value: &Value, out: &mut BTreeMap<String, Value>) {
    match value {
        Value::Null => {}
        Value::Object(map) => {
            for (key, inner) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten(&path, inner, out);
            }
        }
        leaf => {
            out.insert(prefix.to_string(), leaf.clone());
        }
    }
}

/// The paths whose leaf differs between `champion` and `arm`, sorted.
#[must_use]
pub fn changed(champion: &Value, arm: &Value) -> Vec<String> {
    let (ours, theirs) = (leaves(champion), leaves(arm));
    let paths: BTreeSet<&String> = ours.keys().chain(theirs.keys()).collect();
    paths
        .into_iter()
        .filter(|path| ours.get(*path) != theirs.get(*path))
        .cloned()
        .collect()
}

/// Whether `path` is `variable` or lies under it.
#[must_use]
pub fn under(path: &str, variable: &str) -> bool {
    path == variable
        || path
            .strip_prefix(variable)
            .is_some_and(|rest| rest.starts_with('.'))
}

/// The paths `arm` changes from `champion`, all under `variable`.
///
/// # Errors
/// No change, or a changed path outside `variable`, each named.
pub fn single_variable(
    champion: &Value,
    arm: &Value,
    variable: &str,
) -> Result<Vec<String>, VariableError> {
    let paths = changed(champion, arm);
    if paths.is_empty() {
        return Err(VariableError::NoChange);
    }
    let outside: Vec<String> = paths
        .iter()
        .filter(|path| !under(path, variable))
        .cloned()
        .collect();
    if outside.is_empty() {
        Ok(paths)
    } else {
        Err(VariableError::Undeclared {
            variable: variable.to_string(),
            paths: outside,
        })
    }
}

/// The paths `arm` changes from `champion`, all under one of `engineering`.
///
/// # Errors
/// No change, or a changed path outside every engineering path, each named.
pub fn firewall(
    champion: &Value,
    arm: &Value,
    engineering: &[String],
) -> Result<Vec<String>, VariableError> {
    let paths = changed(champion, arm);
    if paths.is_empty() {
        return Err(VariableError::NoChange);
    }
    let outside: Vec<String> = paths
        .iter()
        .filter(|path| !engineering.iter().any(|allowed| under(path, allowed)))
        .cloned()
        .collect();
    if outside.is_empty() {
        Ok(paths)
    } else {
        Err(VariableError::Breach {
            allowed: engineering.to_vec(),
            paths: outside,
        })
    }
}
