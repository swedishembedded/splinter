// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The terms data came under, carried from a source to every dataset and
//! release made from it.
//!
//! Access to research data is not permission to use it for anything: one
//! cohort forbids commercial use, another forbids redistribution, a third
//! allows training only under an approved project. A [`Terms`] states, per
//! kind of use, whether it is allowed, forbidden or unknown; combining the
//! terms of several sources keeps the most restrictive of each, and
//! "unknown" never permits - an unread licence is not a granted one.

use serde::{Deserialize, Serialize};

/// Whether one kind of use is permitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    /// Explicitly allowed.
    Allowed,
    /// Not stated, or not yet read: treated as not allowed.
    Unknown,
    /// Explicitly forbidden.
    Forbidden,
}

/// A kind of use a campaign makes of data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Use {
    /// Training a model on it.
    Training,
    /// Releasing a model trained on it for commercial use.
    CommercialUse,
    /// Passing the data, or a dataset built from it, to others.
    Redistribution,
}

/// The terms of one source, or of several combined.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Terms {
    /// What the terms are called (a licence, a data use agreement).
    pub name: String,
    /// Training a model on the data.
    pub training: Permission,
    /// Commercial use of what is built from it.
    pub commercial_use: Permission,
    /// Redistributing the data or derived datasets.
    pub redistribution: Permission,
    /// Conditions that travel with every use (attribution, citation,
    /// disclosure rules).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<String>,
}

impl Terms {
    /// Terms of public-domain data: every use allowed.
    #[must_use]
    pub fn public_domain(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            training: Permission::Allowed,
            commercial_use: Permission::Allowed,
            redistribution: Permission::Allowed,
            conditions: Vec::new(),
        }
    }

    /// The permission for `use_`.
    #[must_use]
    pub fn permission(&self, use_: Use) -> Permission {
        match use_ {
            Use::Training => self.training,
            Use::CommercialUse => self.commercial_use,
            Use::Redistribution => self.redistribution,
        }
    }

    /// `Ok` when `use_` is allowed, else why not.
    pub fn permits(&self, use_: Use) -> Result<(), String> {
        match self.permission(use_) {
            Permission::Allowed => Ok(()),
            p => Err(format!(
                "{}: {use_:?} is {p:?} under these terms",
                self.name
            )),
        }
    }

    /// The terms of data drawn from all of `parts`: the most restrictive
    /// permission of each kind, every condition kept. `None` for no parts.
    #[must_use]
    pub fn combine(parts: &[Terms]) -> Option<Terms> {
        let first = parts.first()?;
        let mut out = first.clone();
        for t in &parts[1..] {
            out.name = format!("{} + {}", out.name, t.name);
            out.training = out.training.max(t.training);
            out.commercial_use = out.commercial_use.max(t.commercial_use);
            out.redistribution = out.redistribution.max(t.redistribution);
            for c in &t.conditions {
                if !out.conditions.contains(c) {
                    out.conditions.push(c.clone());
                }
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combining_keeps_the_most_restrictive_and_unknown_never_permits() {
        let survey = Terms::public_domain("survey");
        let cohort = Terms {
            name: "cohort DUA".into(),
            training: Permission::Allowed,
            commercial_use: Permission::Forbidden,
            redistribution: Permission::Forbidden,
            conditions: vec!["cite the cohort".into()],
        };
        let unread = Terms {
            name: "trial".into(),
            training: Permission::Unknown,
            commercial_use: Permission::Allowed,
            redistribution: Permission::Allowed,
            conditions: vec![],
        };
        let both = Terms::combine(&[survey.clone(), cohort.clone()]).unwrap();
        assert!(both.permits(Use::Training).is_ok());
        assert!(both.permits(Use::CommercialUse).is_err());
        assert_eq!(both.conditions, vec!["cite the cohort".to_string()]);
        let three = Terms::combine(&[survey.clone(), unread]).unwrap();
        assert_eq!(three.training, Permission::Unknown);
        assert!(three
            .permits(Use::Training)
            .unwrap_err()
            .contains("Unknown"));
        assert!(Terms::combine(&[]).is_none());
        assert!(survey.permits(Use::Redistribution).is_ok());
    }
}
