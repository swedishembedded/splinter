// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-horizon risk prediction from cohort and
// survey data for its clients. If your team needs expertise in building,
// validating and proving time-to-event models on real health records, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Two models compared on the cross-validation folds both ran on: the
//! corrected resampled t-test per metric. A side is a trained arm
//! (`horizon`) or a baseline's predictions scored by [`crate::external`]
//! (`external:<name>`).

use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;

use anyhow::{bail, Result};
use clap::ValueEnum;
use splinter_sdk::model::timeline::survival::compare::corrected_resampled_t;

use crate::commands::{cv_runs, frozen, PREREGISTERED_SEED};
use crate::experiment::Arm;
use crate::metrics::Metrics;

/// One side of a comparison.
#[derive(Clone, Debug, PartialEq)]
pub enum Comparand {
    /// A trained arm, from its cross-validation runs.
    Arm(Arm),
    /// A baseline whose predictions were scored by `external`.
    External(String),
    /// A recipe (or an average of recipes) kept by `recipe`.
    Recipe(String),
}

/// Prefix of an external baseline on the command line.
const EXTERNAL_PREFIX: &str = "external:";
/// Prefix of a recipe on the command line.
const RECIPE_PREFIX: &str = "recipe:";

impl FromStr for Comparand {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        if let Some(name) = s.strip_prefix(RECIPE_PREFIX) {
            return match name {
                "" => Err(format!("{RECIPE_PREFIX}<recipe name>")),
                _ => Ok(Comparand::Recipe(name.to_string())),
            };
        }
        match s.strip_prefix(EXTERNAL_PREFIX) {
            Some(name) if !name.is_empty() => Ok(Comparand::External(name.to_string())),
            Some(_) => Err(format!("{EXTERNAL_PREFIX}<baseline name>")),
            None => Arm::from_str(s, true).map(Comparand::Arm).map_err(|e| {
                format!("{e} (or {EXTERNAL_PREFIX}<baseline name>, {RECIPE_PREFIX}<recipe name>)")
            }),
        }
    }
}

impl Comparand {
    /// Its name in output.
    pub fn name(&self) -> String {
        match self {
            Comparand::Arm(a) => a.name().to_string(),
            Comparand::External(n) => format!("{EXTERNAL_PREFIX}{n}"),
            Comparand::Recipe(n) => format!("{RECIPE_PREFIX}{n}"),
        }
    }

    /// The metrics of each cross-validation fold it was scored on.
    pub fn folds(&self, data: &Path) -> Result<BTreeMap<(usize, usize), Metrics>> {
        match self {
            Comparand::Arm(a) => Ok(cv_runs(data, *a, PREREGISTERED_SEED)?
                .into_iter()
                .map(|(fold, run)| (fold, run.metrics))
                .collect()),
            Comparand::External(n) => crate::external::scores(data, n),
            Comparand::Recipe(n) => crate::recipe_run::scores(data, n),
        }
    }
}

/// A metric two arms are compared on; its name says which way is better.
pub struct Compared {
    /// Name, with the better direction.
    pub name: &'static str,
    /// Reads it from a run's metrics.
    pub get: fn(&Metrics) -> Option<f64>,
}

/// The metrics compared.
pub fn compared_metrics() -> Vec<Compared> {
    vec![
        Compared {
            name: "ibs_0_15 (lower better)",
            get: |m| m.ibs_0_15.as_ref().map(|x| x.value),
        },
        Compared {
            name: "brier_10 (lower better)",
            get: |m| m.brier.get(&10).map(|x| x.value),
        },
        Compared {
            name: "uno_c_5 (higher better)",
            get: |m| m.uno_c.get(&5).map(|x| x.value),
        },
        Compared {
            name: "uno_c_10 (higher better)",
            get: |m| m.uno_c.get(&10).map(|x| x.value),
        },
        Compared {
            name: "calibration_slope_10",
            get: |m| m.calibration_10.as_ref().map(|c| c.slope),
        },
    ]
}

/// Two models on the folds both ran on (only those of `repeats`, when given).
pub fn compare(data: &Path, a: &Comparand, b: &Comparand, repeats: &[usize]) -> Result<()> {
    let (ra, rb) = (a.folds(data)?, b.folds(data)?);
    let folds: Vec<&(usize, usize)> = ra
        .keys()
        .filter(|k| rb.contains_key(k) && (repeats.is_empty() || repeats.contains(&k.0)))
        .collect();
    if folds.len() < 2 {
        bail!(
            "{} folds ran for both {} and {}; at least two are needed",
            folds.len(),
            a.name(),
            b.name()
        );
    }
    // Each fold tests on about 1/(K-1) of what it trains on: K from the
    // partition, never from which folds happen to have run.
    let k = frozen(data)?.partition.spec.folds as usize;
    let test_over_train = 1.0 / (k as f64 - 1.0);
    let (na, nb) = (a.name(), b.name());
    println!(
        "{na} vs {nb} on {} folds (test/train {test_over_train:.3})",
        folds.len()
    );
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>22} {:>8}",
        "metric", na, nb, "diff", "95% CI (corrected)", "p"
    );
    for Compared { name, get } in compared_metrics() {
        let pairs: Vec<(f64, f64)> = folds
            .iter()
            .filter_map(|f| Some((get(&ra[f])?, get(&rb[f])?)))
            .collect();
        let diffs: Vec<f64> = pairs.iter().map(|(x, y)| x - y).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (ma, mb) = (
            mean(&pairs.iter().map(|p| p.0).collect::<Vec<_>>()),
            mean(&pairs.iter().map(|p| p.1).collect::<Vec<_>>()),
        );
        match corrected_resampled_t(&diffs, test_over_train) {
            Some(t) => println!(
                "{name:<28} {ma:>10.5} {mb:>10.5} {:>+10.5} [{:>+9.5}, {:>+9.5}] {:>8.4}",
                t.mean, t.ci95.0, t.ci95.1, t.p_two_sided
            ),
            None => println!("{name:<28} {ma:>10.5} {mb:>10.5} (no variance across folds)"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_side_is_an_arm_or_an_external_baseline() {
        assert_eq!("horizon".parse(), Ok(Comparand::Arm(Arm::Horizon)));
        assert_eq!("age-sex".parse(), Ok(Comparand::Arm(Arm::AgeSex)));
        assert_eq!(
            "external:gbs-all".parse(),
            Ok(Comparand::External("gbs-all".into()))
        );
        assert_eq!(
            "recipe:d32-s1".parse(),
            Ok(Comparand::Recipe("d32-s1".into()))
        );
        assert!("recipe:".parse::<Comparand>().is_err());
        assert!("external:".parse::<Comparand>().is_err());
        assert!("nonsense".parse::<Comparand>().is_err());
    }
}
