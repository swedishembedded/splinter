// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Harmonisation of a record file's variable onto a shared concept,
//! proposed by a model and admitted by code.
//!
//! Pooling many sources means mapping hundreds of variables - renamed across
//! releases, measured in other units, with refusal codes mixed into the
//! numbers - onto one set of concepts. A model reads the variable's codebook
//! entry and proposes the mapping ([`MappingProposal`]), quoting the words it
//! relied on; [`admit`] accepts it only when the quotation is really in the
//! codebook, every code it calls missing is one the codebook says is not a
//! measurement and none of those is left inside the valid range, and the
//! values, converted, fall where the concept's values from other sources
//! fall. A mapping that passes is evidence, not a guarantee: the rules check
//! what can be checked mechanically.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::codebook::VariableDoc;

/// What a model proposes for one variable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MappingProposal {
    /// The concept the variable measures, by name, or `none` if it measures
    /// none of them.
    pub concept: String,
    /// The factor that turns a recorded value into the concept's unit.
    pub factor: f64,
    /// The smallest real measurement, in the concept's unit.
    pub valid_min: f64,
    /// The largest real measurement, in the concept's unit.
    pub valid_max: f64,
    /// Recorded codes that are not measurements (refused, don't know,
    /// missing), as recorded (before the factor).
    pub missing_codes: Vec<f64>,
    /// The codebook words the proposal rests on, copied exactly.
    pub quote: String,
}

/// Where a concept's values from other sources lie.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reference {
    /// 5th percentile.
    pub q05: f64,
    /// Median.
    pub median: f64,
    /// 95th percentile.
    pub q95: f64,
}

/// A concept a variable may map onto.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConceptSpec {
    /// Its name.
    pub name: String,
    /// Its canonical unit.
    pub unit: String,
    /// What it is.
    pub description: String,
    /// Its values in the sources already mapped, when there are any.
    pub reference: Option<Reference>,
}

/// A model that proposes mappings.
#[async_trait]
pub trait MappingProposer: Send + Sync {
    /// The string provenance records for the model.
    fn identity(&self) -> &str;
    /// A mapping of `variable` onto one of `concepts`.
    async fn propose(
        &self,
        variable: &VariableDoc,
        concepts: &[ConceptSpec],
    ) -> Result<MappingProposal, String>;
}

/// The verdict on one proposal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Admission {
    /// Whether every rule held.
    pub admitted: bool,
    /// Every rule that failed, in words.
    pub reasons: Vec<String>,
    /// The converted values' `(q05, median, q95)`, when there were values.
    pub converted: Option<(f64, f64, f64)>,
}

fn normalised(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Codebook meanings that say a code is not a measurement.
fn not_a_measurement(meaning: &str) -> bool {
    let m = meaning.to_lowercase();
    [
        "refused",
        "don't know",
        "dont know",
        "do not know",
        "missing",
        "not applicable",
        "unknown",
        "cannot be determined",
    ]
    .iter()
    .any(|w| m.contains(w))
}

/// The numeric codes `variable`'s codebook marks as refused, unknown,
/// missing or not applicable: what [`admit`] accepts as missing codes.
#[must_use]
pub fn non_measurement_codes(variable: &VariableDoc) -> Vec<f64> {
    variable
        .codes
        .iter()
        .filter(|c| not_a_measurement(&c.meaning))
        .filter_map(|c| c.value.trim().parse().ok())
        .collect()
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// Values a reference needs before a distribution check is made.
pub const MIN_VALUES: usize = 10;
/// How far the converted spread may differ from the reference's, as a ratio.
pub const SPREAD_RATIO: f64 = 5.0;
/// The largest share of recorded values, missing codes aside, a mapping may
/// put outside its own valid range. A valid range describes the real
/// measurements; when many fall outside it, the range or the factor is
/// wrong - and the values that remain inside can look entirely plausible.
pub const MAX_OUT_OF_RANGE: f64 = 0.1;

/// Check `p` for `variable` against its codebook entry, its recorded
/// `values` and the `concepts`.
#[must_use]
pub fn admit(
    p: &MappingProposal,
    variable: &VariableDoc,
    values: &[Option<f64>],
    concepts: &[ConceptSpec],
) -> Admission {
    let mut reasons = Vec::new();
    let concept = concepts.iter().find(|c| c.name == p.concept);
    if concept.is_none() {
        reasons.push(format!(
            "concept {} is not one of the concepts offered",
            p.concept
        ));
    }
    if p.quote.trim().is_empty()
        || !normalised(&variable.full_text()).contains(&normalised(&p.quote))
    {
        reasons.push(format!(
            "the quotation {:?} is not in the codebook entry",
            p.quote
        ));
    }
    if !(p.factor.is_finite() && p.factor > 0.0) {
        reasons.push(format!("factor {} is not a positive number", p.factor));
    }
    if !(p.valid_min.is_finite() && p.valid_max.is_finite() && p.valid_min < p.valid_max) {
        reasons.push(format!(
            "valid range [{}, {}] is not an interval",
            p.valid_min, p.valid_max
        ));
    }
    let in_range = |x: f64| (p.valid_min..=p.valid_max).contains(&(x * p.factor));
    for code in &p.missing_codes {
        let documented = variable.codes.iter().any(|c| {
            c.value.trim().parse::<f64>().ok() == Some(*code) && not_a_measurement(&c.meaning)
        });
        if !documented {
            reasons.push(format!(
                "code {code} is not documented as a non-measurement"
            ));
        }
        if in_range(*code) {
            reasons.push(format!("missing code {code} lies inside the valid range"));
        }
    }
    for code in non_measurement_codes(variable) {
        if !p.missing_codes.contains(&code) && in_range(code) {
            reasons.push(format!(
                "documented non-measurement code {code} would be read as a measurement"
            ));
        }
    }
    let recorded: Vec<f64> = values
        .iter()
        .flatten()
        .filter(|x| !p.missing_codes.contains(x))
        .map(|x| x * p.factor)
        .collect();
    let mut converted: Vec<f64> = recorded
        .iter()
        .copied()
        .filter(|x| (p.valid_min..=p.valid_max).contains(x))
        .collect();
    converted.sort_by(f64::total_cmp);
    let outside = recorded.len() - converted.len();
    if !recorded.is_empty() && outside as f64 > MAX_OUT_OF_RANGE * recorded.len() as f64 {
        reasons.push(format!(
            "{outside} of {} recorded values fall outside the valid range",
            recorded.len()
        ));
    }
    let summary = (converted.len() >= MIN_VALUES).then(|| {
        (
            quantile(&converted, 0.05),
            quantile(&converted, 0.5),
            quantile(&converted, 0.95),
        )
    });
    match (summary, concept.and_then(|c| c.reference.as_ref())) {
        (Some((lo, med, hi)), Some(r)) => {
            if !(r.q05..=r.q95).contains(&med) {
                reasons.push(format!(
                    "the converted median {med} lies outside the concept's 5-95 range [{}, {}]",
                    r.q05, r.q95
                ));
            }
            let (spread, reference) = (hi - lo, r.q95 - r.q05);
            if reference > 0.0
                && !(reference / SPREAD_RATIO..=reference * SPREAD_RATIO).contains(&spread)
            {
                reasons.push(format!("the converted 5-95 spread {spread} is not within a factor {SPREAD_RATIO} of the concept's {reference}"));
            }
        }
        (None, Some(_)) => reasons.push(format!(
            "fewer than {MIN_VALUES} values fall in the proposed range"
        )),
        _ => {}
    }
    Admission {
        admitted: reasons.is_empty(),
        reasons,
        converted: summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::Code;

    fn crp_doc() -> VariableDoc {
        VariableDoc {
            name: "LBXHSCRP".into(),
            label: "HS C-Reactive Protein (mg/L)".into(),
            text: "High-Sensitivity C-Reactive Protein (hs-CRP) (mg/L)".into(),
            target: "Both males and females 1 YEARS - 150 YEARS".into(),
            codes: vec![Code {
                value: "0.08 to 188.5".into(),
                meaning: "Range of Values".into(),
                count: Some(7867),
            }],
        }
    }

    fn crp_concept() -> Vec<ConceptSpec> {
        vec![ConceptSpec {
            name: "crp_mgdl".into(),
            unit: "mg/dL".into(),
            description: "C-reactive protein".into(),
            reference: Some(Reference {
                q05: 0.02,
                median: 0.2,
                q95: 1.5,
            }),
        }]
    }

    fn values() -> Vec<Option<f64>> {
        // Skewed like the real thing: 0.2 to 20 mg/L, evenly on a log scale.
        (0..200)
            .map(|i| Some(0.2 * 100f64.powf(f64::from(i) / 199.0)))
            .chain([None])
            .collect()
    }

    #[test]
    fn a_correct_unit_conversion_is_admitted_and_a_wrong_one_is_not() {
        let good = MappingProposal {
            concept: "crp_mgdl".into(),
            factor: 0.1,
            valid_min: 0.0,
            valid_max: 50.0,
            missing_codes: vec![],
            quote: "hs-CRP) (mg/L)".into(),
        };
        let a = admit(&good, &crp_doc(), &values(), &crp_concept());
        assert!(a.admitted, "{:?}", a.reasons);
        // mg/L read as if it were mg/dL: the values land ten times too high.
        let wrong = MappingProposal {
            factor: 1.0,
            ..good.clone()
        };
        let a = admit(&wrong, &crp_doc(), &values(), &crp_concept());
        assert!(
            !a.admitted && a.reasons.iter().any(|r| r.contains("median")),
            "{:?}",
            a.reasons
        );
        let invented = MappingProposal {
            quote: "measured in mg/dL".into(),
            ..good
        };
        assert!(admit(&invented, &crp_doc(), &values(), &crp_concept())
            .reasons
            .iter()
            .any(|r| r.contains("quotation")));
    }

    #[test]
    fn a_unit_error_hidden_by_the_valid_range_is_rejected() {
        // A ratio capped at 5: read ten times too large, nine in ten values
        // fall past the cap, and the tenth that remains looks plausible.
        let doc = VariableDoc {
            name: "INDFMPIR".into(),
            label: "Ratio of family income to poverty".into(),
            text: "A ratio of family income to poverty guidelines.".into(),
            target: "all".into(),
            codes: vec![Code {
                value: "0 to 5".into(),
                meaning: "Range of Values".into(),
                count: None,
            }],
        };
        let concept = vec![ConceptSpec {
            name: "income_poverty_ratio".into(),
            unit: "ratio".into(),
            description: "income to poverty".into(),
            reference: Some(Reference {
                q05: 0.5,
                median: 2.5,
                q95: 5.0,
            }),
        }];
        let values: Vec<Option<f64>> = (0..500).map(|i| Some(f64::from(i) / 100.0)).collect();
        let tenfold = MappingProposal {
            concept: "income_poverty_ratio".into(),
            factor: 10.0,
            valid_min: 0.0,
            valid_max: 5.0,
            missing_codes: vec![],
            quote: "Ratio of family income to poverty".into(),
        };
        let a = admit(&tenfold, &doc, &values, &concept);
        assert!(
            a.reasons
                .iter()
                .any(|r| r.contains("outside the valid range")),
            "{:?}",
            a.reasons
        );
    }

    #[test]
    fn refusal_codes_must_be_documented_and_kept_out_of_range() {
        let doc = VariableDoc {
            name: "SMD030".into(),
            label: "Age started smoking cigarettes regularly".into(),
            text: "How old were you when you first started to smoke cigarettes fairly regularly?"
                .into(),
            target: "18+".into(),
            codes: vec![
                Code {
                    value: "7 to 76".into(),
                    meaning: "Range of Values".into(),
                    count: None,
                },
                Code {
                    value: "777".into(),
                    meaning: "Refused".into(),
                    count: None,
                },
                Code {
                    value: "999".into(),
                    meaning: "Don't know".into(),
                    count: None,
                },
            ],
        };
        let concept = vec![ConceptSpec {
            name: "smoking_start_age".into(),
            unit: "years".into(),
            description: "age started".into(),
            reference: None,
        }];
        let ok = MappingProposal {
            concept: "smoking_start_age".into(),
            factor: 1.0,
            valid_min: 5.0,
            valid_max: 85.0,
            missing_codes: vec![777.0, 999.0],
            quote: "first started to smoke".into(),
        };
        assert!(admit(&ok, &doc, &[Some(17.0), Some(777.0)], &concept).admitted);
        let leaky = MappingProposal {
            valid_max: 1000.0,
            missing_codes: vec![],
            ..ok.clone()
        };
        let a = admit(&leaky, &doc, &[], &concept);
        assert!(
            a.reasons
                .iter()
                .filter(|r| r.contains("would be read as a measurement"))
                .count()
                == 2,
            "{:?}",
            a.reasons
        );
        let invented_code = MappingProposal {
            missing_codes: vec![777.0, 999.0, 0.0],
            ..ok
        };
        assert!(admit(&invented_code, &doc, &[], &concept)
            .reasons
            .iter()
            .any(|r| r.contains("code 0 is not documented")));
    }
}
