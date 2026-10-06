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
//!
//! A source states its terms as one of five machine-readable
//! [`UsagePolicy`] labels, each a fixed setting of the three axes
//! ([`UsagePolicy::terms`]); a dataset, a training run, a candidate and a
//! release carry the combination of everything they were made from
//! ([`combine_stated`]), so the strongest restriction of any ancestor is the
//! one that binds. Whether a release may be handed to others is decided from
//! those terms by [`Terms::permits_unrestricted_release`].

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

/// The usage-policy classification a source's data is under: the label a
/// person states, mapped by [`UsagePolicy::terms`] onto the three axes of
/// [`Terms`]. The mapping is a decision recorded here, not inferred:
///
/// | label | training | commercial use | redistribution |
/// |---|---|---|---|
/// | `redistributable` | allowed | allowed | allowed |
/// | `research_only` | allowed | forbidden | forbidden |
/// | `noncommercial` | allowed | forbidden | allowed |
/// | `restricted_DUA` | allowed | forbidden | forbidden, plus the condition that the data use agreement governs every use |
/// | `unknown` | unknown | unknown | unknown |
///
/// `unknown` fails closed: no axis is allowed, so no release made from it is
/// distributable. A label allows training only where the data holder permits
/// it; a source whose licence forbids training has no label here and is
/// stated as explicit [`Terms`] instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsagePolicy {
    /// Free to use, release and pass on.
    #[serde(rename = "redistributable")]
    Redistributable,
    /// Research use only: no commercial use, no redistribution.
    #[serde(rename = "research_only")]
    ResearchOnly,
    /// Any non-commercial use; may be passed on.
    #[serde(rename = "noncommercial")]
    Noncommercial,
    /// Held under a data use agreement that restricts every use.
    #[serde(rename = "restricted_DUA")]
    RestrictedDua,
    /// Not stated or not read.
    #[serde(rename = "unknown")]
    Unknown,
}

impl UsagePolicy {
    /// Every label, in the order the table above lists them.
    pub const ALL: [UsagePolicy; 5] = [
        UsagePolicy::Redistributable,
        UsagePolicy::ResearchOnly,
        UsagePolicy::Noncommercial,
        UsagePolicy::RestrictedDua,
        UsagePolicy::Unknown,
    ];

    /// The label as it is written in data and on a command line.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Redistributable => "redistributable",
            Self::ResearchOnly => "research_only",
            Self::Noncommercial => "noncommercial",
            Self::RestrictedDua => "restricted_DUA",
            Self::Unknown => "unknown",
        }
    }

    /// The label written as `text`, or why it is not one.
    pub fn parse(text: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|p| p.as_str() == text)
            .ok_or_else(|| {
                let known: Vec<&str> = Self::ALL.iter().map(|p| p.as_str()).collect();
                format!(
                    "{text:?} is not a usage policy; the labels are {}",
                    known.join(", ")
                )
            })
    }

    /// The terms this label states for the data called `name`.
    #[must_use]
    pub fn terms(self, name: impl Into<String>) -> Terms {
        use Permission::{Allowed, Forbidden, Unknown};
        let (training, commercial_use, redistribution) = match self {
            Self::Redistributable => (Allowed, Allowed, Allowed),
            Self::ResearchOnly | Self::RestrictedDua => (Allowed, Forbidden, Forbidden),
            Self::Noncommercial => (Allowed, Forbidden, Allowed),
            Self::Unknown => (Unknown, Unknown, Unknown),
        };
        let conditions = if self == Self::RestrictedDua {
            vec!["the data use agreement governs every use".to_string()]
        } else {
            Vec::new()
        };
        Terms {
            name: name.into(),
            training,
            commercial_use,
            redistribution,
            conditions,
        }
    }
}

/// How widely a release may be handed on, as its maker asks and the manifest
/// records.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Distribution {
    /// Kept where it was made, recording the restrictions on it. The default:
    /// nothing is distributed unless it is asked to be and allowed to be.
    #[default]
    Restricted,
    /// Handed to others and used commercially: every axis of the terms must
    /// be allowed ([`Terms::permits_unrestricted_release`]).
    Unrestricted,
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

    /// Terms that state nothing, named `name`: every axis unknown, so nothing
    /// is permitted. What data with no stated terms is treated as.
    #[must_use]
    pub fn unknown(name: impl Into<String>) -> Self {
        UsagePolicy::Unknown.terms(name)
    }

    /// `Ok` when a release made under these terms may be handed to others
    /// and used commercially: training, commercial use and redistribution
    /// are all allowed. Else every axis that is not, so the refusal names
    /// what to resolve.
    pub fn permits_unrestricted_release(&self) -> Result<(), String> {
        let blocked: Vec<String> = [Use::Training, Use::CommercialUse, Use::Redistribution]
            .into_iter()
            .filter_map(|u| match self.permission(u) {
                Permission::Allowed => None,
                p => Some(format!("{u:?} is {p:?}")),
            })
            .collect();
        if blocked.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "terms {:?} do not allow an unrestricted release: {}",
                self.name,
                blocked.join(", ")
            ))
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

/// The terms of something made from parts whose terms may not be stated:
/// `None` when no part states any (nothing is known, and the consumer treats
/// it as [`Terms::unknown`]); otherwise the combination, where a part that
/// states none counts as unknown, so one unread licence among stated ones
/// still fails closed.
#[must_use]
pub fn combine_stated<'a>(parts: impl IntoIterator<Item = Option<&'a Terms>>) -> Option<Terms> {
    let parts: Vec<Option<&Terms>> = parts.into_iter().collect();
    if parts.iter().all(Option::is_none) {
        return None;
    }
    let stated: Vec<Terms> = parts
        .into_iter()
        .map(|t| t.cloned().unwrap_or_else(|| Terms::unknown("unstated")))
        .collect();
    Terms::combine(&stated)
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

    #[test]
    fn every_label_maps_to_fixed_axes_and_unknown_fails_closed() {
        let redistributable = UsagePolicy::Redistributable.terms("open");
        assert!(redistributable.permits_unrestricted_release().is_ok());
        for label in [
            UsagePolicy::ResearchOnly,
            UsagePolicy::Noncommercial,
            UsagePolicy::RestrictedDua,
            UsagePolicy::Unknown,
        ] {
            let terms = label.terms("data");
            assert!(
                terms.permits_unrestricted_release().is_err(),
                "{label:?} must not allow an unrestricted release"
            );
        }
        let unknown = UsagePolicy::Unknown.terms("unread");
        assert!(unknown.permits(Use::Training).is_err());
        assert!(unknown.permits(Use::Redistribution).is_err());
        assert_eq!(
            UsagePolicy::Noncommercial.terms("nc").redistribution,
            Permission::Allowed
        );
        assert!(!UsagePolicy::RestrictedDua
            .terms("dua")
            .conditions
            .is_empty());
        for label in UsagePolicy::ALL {
            assert_eq!(UsagePolicy::parse(label.as_str()), Ok(label));
            let json = serde_json::to_string(&label).unwrap();
            assert_eq!(json, format!("\"{}\"", label.as_str()));
        }
        assert!(UsagePolicy::parse("open")
            .unwrap_err()
            .contains("research_only"));
    }

    #[test]
    fn terms_unstated_anywhere_fail_closed_and_none_stated_stays_none() {
        let open = UsagePolicy::Redistributable.terms("open");
        assert!(combine_stated([None, None]).is_none());
        assert!(combine_stated([]).is_none());
        let mixed = combine_stated([Some(&open), None]).unwrap();
        assert_eq!(mixed.training, Permission::Unknown);
        assert!(mixed.permits_unrestricted_release().is_err());
        let both = combine_stated([Some(&open), Some(&open)]).unwrap();
        assert!(both.permits_unrestricted_release().is_ok());
    }
}
