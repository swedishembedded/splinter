// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements outcome definitions as data for clinical
// risk models, so that what a model predicts is the same thing in every
// cohort it is trained or checked on. If your team needs expertise in
// harmonising outcomes across cohorts, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The outcome ontology, as data.
//!
//! `ontology/outcomes-v1.json` is a read-only copy of the definitions the
//! project's health work shares: each outcome has an event code, a
//! definition, the definition each source dataset uses and the datasets
//! present on this machine that can supply it (`supported_locally`; an empty
//! list means the outcome is defined but cannot be trained or evaluated
//! here). `ontology/supports-v1.json` adds the datasets this sample supplies.
//! An outcome is requested for a dataset by [`Ontology::request`], which
//! refuses one the dataset is not listed for: incompatible definitions across
//! cohorts are never pooled, so a dataset supplies an outcome only where its
//! own definition is written down.
//!
//! The one source mapping that code carries is the NHANES linked-mortality
//! cause recode ([`nhanes_cause`]), pinned by golden tests against the
//! definitions in the data.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};
use serde::Deserialize;

/// The ontology shipped with the sample.
pub const OUTCOMES: &str = include_str!("../ontology/outcomes-v1.json");
/// The datasets the sample supplies, beside the ontology's own lists.
pub const SUPPORTS: &str = include_str!("../ontology/supports-v1.json");
/// The dataset id the NHANES linked-mortality timelines are imported as.
pub const NHANES_DATASET: &str = "nhanes_mortality";
/// The dataset id the synthetic cohort is imported as.
pub const SYNTHETIC_DATASET: &str = "synthetic_health";
/// The all-cause outcome's event code: the union of the cause groups.
pub const ALL_CAUSE: &str = "death:any";

/// One outcome's definition.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Outcome {
    /// Its canonical name.
    pub canonical_name: String,
    /// The event code it travels under in a timeline.
    pub event_code: String,
    /// `terminal`, `incident` or `recurrent`.
    pub kind: String,
    /// What it means.
    pub definition: String,
    /// The definition each source dataset uses.
    #[serde(default)]
    pub source_definitions: BTreeMap<String, String>,
    /// The event codes that compete with it.
    #[serde(default)]
    pub competing_events: Vec<String>,
    /// The datasets on this machine that can supply it.
    #[serde(default)]
    pub supported_locally: Vec<String>,
    /// What it must not be mistaken for.
    #[serde(default)]
    pub caveats: Vec<String>,
}

#[derive(Deserialize)]
struct Document {
    schema: String,
    outcomes: Vec<Outcome>,
}

#[derive(Deserialize)]
struct Supports {
    schema: String,
    datasets: BTreeMap<String, Vec<String>>,
}

/// Why an outcome cannot be requested.
#[derive(Debug)]
pub enum OntologyError {
    /// The ontology defines no outcome with this event code.
    Unknown {
        /// The code asked for.
        code: String,
    },
    /// The outcome is defined, but the dataset is not listed for it.
    Unsupported {
        /// The code asked for.
        code: String,
        /// The dataset it was asked for.
        dataset: String,
        /// The datasets that are listed for it; empty when none is, which the
        /// ontology calls unsupported locally.
        supported_by: Vec<String>,
    },
}

/// The ontology and the datasets listed for each outcome.
#[derive(Clone, Debug)]
pub struct Ontology {
    outcomes: BTreeMap<String, Outcome>,
    supports: BTreeMap<String, BTreeSet<String>>,
}

impl Ontology {
    /// The ontology the sample ships.
    pub fn bundled() -> Result<Self> {
        Self::parse(OUTCOMES, SUPPORTS)
    }

    /// An ontology from the text of its two files.
    pub fn parse(outcomes: &str, supports: &str) -> Result<Self> {
        let document: Document = serde_json::from_str(outcomes).context("outcomes-v1.json")?;
        anyhow::ensure!(
            document.schema == "health-outcome-ontology-v1",
            "outcomes-v1.json: schema {:?} is not health-outcome-ontology-v1",
            document.schema
        );
        let extra: Supports = serde_json::from_str(supports).context("supports-v1.json")?;
        anyhow::ensure!(
            extra.schema == "health-outcome-supports-v1",
            "supports-v1.json: schema {:?} is not health-outcome-supports-v1",
            extra.schema
        );
        let mut by_code = BTreeMap::new();
        for outcome in document.outcomes {
            let code = outcome.event_code.clone();
            anyhow::ensure!(
                by_code.insert(code.clone(), outcome).is_none(),
                "outcomes-v1.json defines {code} twice"
            );
        }
        let mut supports: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (code, outcome) in &by_code {
            for dataset in &outcome.supported_locally {
                supports
                    .entry(dataset.clone())
                    .or_default()
                    .insert(code.clone());
            }
        }
        for (dataset, codes) in extra.datasets {
            for code in &codes {
                anyhow::ensure!(
                    by_code.contains_key(code),
                    "supports-v1.json lists {code} for {dataset}, which outcomes-v1.json does not define"
                );
            }
            supports.entry(dataset).or_default().extend(codes);
        }
        Ok(Self {
            outcomes: by_code,
            supports,
        })
    }

    /// The outcome with event code `code`.
    pub fn outcome(&self, code: &str) -> Option<&Outcome> {
        self.outcomes.get(code)
    }

    /// Every event code defined, in order.
    pub fn codes(&self) -> impl Iterator<Item = &str> {
        self.outcomes.keys().map(String::as_str)
    }

    /// The datasets listed for `code`.
    pub fn supported_by(&self, code: &str) -> Vec<String> {
        self.supports
            .iter()
            .filter(|(_, codes)| codes.contains(code))
            .map(|(dataset, _)| dataset.clone())
            .collect()
    }

    /// The outcome `code` as `dataset` supplies it, or why it cannot be
    /// requested: it is not defined, or the dataset is not listed for it.
    pub fn request(&self, code: &str, dataset: &str) -> Result<&Outcome, OntologyError> {
        let outcome = self
            .outcomes
            .get(code)
            .ok_or_else(|| OntologyError::Unknown {
                code: code.to_owned(),
            })?;
        if self.supports.get(dataset).is_some_and(|c| c.contains(code)) {
            Ok(outcome)
        } else {
            Err(OntologyError::Unsupported {
                code: code.to_owned(),
                dataset: dataset.to_owned(),
                supported_by: self.supported_by(code),
            })
        }
    }

    /// Whether `code` is named as a competing event by an outcome `dataset`
    /// is listed for: a cause of death that competes with the outcomes asked
    /// for is part of how they are defined, though it is not itself one.
    fn competes_on(&self, code: &str, dataset: &str) -> bool {
        self.outcomes.values().any(|o| {
            o.competing_events.iter().any(|c| c == code)
                && self
                    .supports
                    .get(dataset)
                    .is_some_and(|s| s.contains(&o.event_code))
        })
    }

    /// Every code of `codes` as `dataset` supplies it, or the first refusal:
    /// each is a requestable outcome, or a competing event of one.
    pub fn request_all(&self, codes: &[String], dataset: &str) -> Result<(), OntologyError> {
        codes
            .iter()
            .try_for_each(|c| match self.request(c, dataset) {
                Ok(_) => Ok(()),
                Err(OntologyError::Unsupported { .. } | OntologyError::Unknown { .. })
                    if self.competes_on(c, dataset) =>
                {
                    Ok(())
                }
                Err(e) => Err(e),
            })
    }
}

/// The cause group of an NHANES linked-mortality record by its
/// `UCOD_LEADING` recode: heart disease (1) and cerebrovascular disease (5)
/// are cardiovascular, malignant neoplasms (2) are cancer, every other
/// recode is another cause.
#[must_use]
pub fn nhanes_cause(ucod_leading: u16) -> &'static str {
    match ucod_leading {
        1 | 5 => "death:cvd",
        2 => "death:cancer",
        _ => "death:other",
    }
}

/// The recode numbers a definition names (`recode 001 (...) or 005 (...)`):
/// the three-digit numbers that follow the word `recode`, up to the end of
/// the sentence.
#[must_use]
pub fn recodes_named(definition: &str) -> BTreeSet<u16> {
    let mut found = BTreeSet::new();
    if let Some(at) = definition.find("recode") {
        let mut digits = String::new();
        for c in definition[at..].chars().chain(std::iter::once(' ')) {
            if c.is_ascii_digit() {
                digits.push(c);
                continue;
            }
            if digits.len() == 3 {
                if let Ok(n) = digits.parse() {
                    found.insert(n);
                }
            }
            digits.clear();
        }
    }
    found
}

/// The union of cause groups as one outcome: a record has the all-cause
/// outcome when it has any of `causes`.
#[must_use]
pub fn all_cause<'a>(causes: impl IntoIterator<Item = &'a str>) -> bool {
    causes
        .into_iter()
        .any(|c| c.starts_with("death:") && c != ALL_CAUSE)
}

impl std::fmt::Display for OntologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { code } => write!(f, "outcome {code:?} is not defined by the ontology"),
            Self::Unsupported {
                code,
                dataset,
                supported_by,
            } if supported_by.is_empty() => write!(
                f,
                "outcome {code} is defined but unsupported locally: no dataset is listed for it, so it cannot be requested for dataset {dataset:?}"
            ),
            Self::Unsupported {
                code,
                dataset,
                supported_by,
            } => write!(
                f,
                "outcome {code} cannot be requested for dataset {dataset:?}, which is not listed for it; listed: {}",
                supported_by.join(", ")
            ),
        }
    }
}

impl std::error::Error for OntologyError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundled() -> Ontology {
        Ontology::bundled().unwrap()
    }

    #[test]
    fn the_nhanes_cause_recode_is_pinned_to_the_definitions_in_the_data() {
        // Golden: the mapping the lifecourse timelines use.
        for (recode, group) in [
            (1, "death:cvd"),
            (2, "death:cancer"),
            (3, "death:other"),
            (4, "death:other"),
            (5, "death:cvd"),
            (6, "death:other"),
            (7, "death:other"),
            (8, "death:other"),
            (9, "death:other"),
            (10, "death:other"),
        ] {
            assert_eq!(nhanes_cause(recode), group, "UCOD_LEADING {recode}");
        }
        // The definitions in outcomes-v1.json name the same recodes.
        let o = bundled();
        let named = |code: &str| {
            recodes_named(&o.outcome(code).unwrap().source_definitions[NHANES_DATASET])
        };
        assert_eq!(named("death:cvd"), BTreeSet::from([1, 5]));
        assert_eq!(named("death:cancer"), BTreeSet::from([2]));
        let every: BTreeSet<&str> = (1..=10).map(nhanes_cause).collect();
        assert_eq!(
            every,
            BTreeSet::from(["death:cvd", "death:cancer", "death:other"])
        );
        // Every recode named maps to the outcome that names it.
        for recode in named("death:cvd") {
            assert_eq!(nhanes_cause(recode), "death:cvd");
        }
    }

    #[test]
    fn all_cause_is_the_union_of_the_cause_groups() {
        let o = bundled();
        let all = o.outcome(ALL_CAUSE).unwrap();
        for group in ["death:cvd", "death:cancer", "death:other"] {
            assert!(
                all.definition.contains(group),
                "{group} is a member of the union"
            );
            assert!(all_cause([group]));
        }
        assert!(
            !all_cause(["dx:t2d"]),
            "an incident diagnosis is not a death"
        );
        assert!(!all_cause([]));
        // Every recode of the linked file is some cause, so all-cause is the union.
        assert!((1..=10).all(|r| all_cause([nhanes_cause(r)])));
    }

    #[test]
    fn an_outcome_marked_unsupported_locally_cannot_be_requested_for_a_dataset_that_does_not_list_it(
    ) {
        let o = bundled();
        // Defined, with no dataset listed.
        let t2d = o.outcome("dx:t2d").unwrap();
        assert!(t2d.supported_locally.is_empty());
        for dataset in [NHANES_DATASET, SYNTHETIC_DATASET, "any-other"] {
            let why = o.request("dx:t2d", dataset).unwrap_err();
            assert!(
                matches!(why, OntologyError::Unsupported { ref supported_by, .. } if supported_by.is_empty()),
                "{why}"
            );
            assert!(why.to_string().contains("unsupported locally"), "{why}");
        }
        // Defined and supported, but not for this dataset.
        let why = o.request("death:cvd", "another_cohort").unwrap_err();
        assert!(why.to_string().contains("not listed"), "{why}");
        assert!(
            why.to_string().contains(NHANES_DATASET) && why.to_string().contains(SYNTHETIC_DATASET)
        );
        // Not defined at all.
        assert!(matches!(
            o.request("death:zzz", NHANES_DATASET),
            Err(OntologyError::Unknown { .. })
        ));
        // Listed: requestable.
        assert!(o.request("death:cvd", NHANES_DATASET).is_ok());
        assert!(o.request("death:cvd", SYNTHETIC_DATASET).is_ok());
        assert!(o
            .request_all(&["death:cvd".into(), "dx:t2d".into()], SYNTHETIC_DATASET)
            .is_err());
        // A competing cause of death is requestable where an outcome that names it is.
        let competing = [
            "death:cvd".to_owned(),
            "death:cancer".to_owned(),
            "death:other".to_owned(),
        ];
        assert!(
            o.request("death:other", NHANES_DATASET).is_err(),
            "it is not an outcome of its own"
        );
        assert!(o.request_all(&competing, NHANES_DATASET).is_ok());
        assert!(o.request_all(&competing, "another_cohort").is_err());
    }

    #[test]
    fn a_supports_file_lists_a_dataset_for_an_outcome_and_only_for_a_defined_one() {
        let extra =
            r#"{"schema":"health-outcome-supports-v1","datasets":{"future_cohort":["dx:t2d"]}}"#;
        let o = Ontology::parse(OUTCOMES, extra).unwrap();
        assert!(
            o.request("dx:t2d", "future_cohort").is_ok(),
            "a dataset that lists it may request it"
        );
        assert!(o.request("dx:mi", "future_cohort").is_err());
        let bad = r#"{"schema":"health-outcome-supports-v1","datasets":{"x":["dx:nonsense"]}}"#;
        assert!(Ontology::parse(OUTCOMES, bad)
            .unwrap_err()
            .to_string()
            .contains("dx:nonsense"));
        let wrong = r#"{"schema":"other","datasets":{}}"#;
        assert!(Ontology::parse(OUTCOMES, wrong).is_err());
    }
}
