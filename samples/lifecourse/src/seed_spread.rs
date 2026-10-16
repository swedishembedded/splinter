// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements risk models that report how much their own
// training varies for its clients. If your team needs expertise in model
// uncertainty for clinical prediction, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The report's seed-spread section: the same cross-validation folds
//! retrained with different seeds, and how far the score moves with the seed
//! alone. It reads the seeds the runs record, so a run trained with a drawn
//! seed is in it as much as one trained with the pre-registered seed.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;

use crate::commands::{cv_by_seed, FoldRuns, Run};
use crate::experiment::Arm;

/// The arms the section compares across seeds.
const ARMS: [Arm; 2] = [Arm::Horizon, Arm::Standard];

/// The section's markdown: one row per arm that has runs under at least two
/// seeds, each seed compared on the folds every one of them ran.
pub fn section(data: &Path) -> Result<String> {
    let mut out = String::new();
    writeln!(out, "\n## Secondary: how much a result moves with the training seed\n\nCross-validated mean IBS 1-15 per seed on the folds every listed seed ran, and per fold the spread (largest minus smallest) across seeds, averaged over folds.\n\n| arm | seeds | folds | mean IBS per seed | mean per-fold spread |\n|---|---|---|---|---|")?;
    for arm in ARMS {
        let by_seed: Vec<(u64, FoldRuns)> = cv_by_seed(data, arm)?.into_iter().collect();
        if by_seed.len() < 2 {
            continue;
        }
        let folds: Vec<(usize, usize)> = by_seed[0]
            .1
            .keys()
            .filter(|k| by_seed.iter().all(|(_, r)| r.contains_key(k)))
            .copied()
            .collect();
        let ibs = |r: &Run| r.metrics.ibs_0_15.as_ref().map(|m| m.value);
        let means: Vec<String> = by_seed
            .iter()
            .map(|(seed, r)| {
                let v: Vec<f64> = folds.iter().filter_map(|k| ibs(&r[k])).collect();
                format!(
                    "s{seed} {:.5}",
                    v.iter().sum::<f64>() / v.len().max(1) as f64
                )
            })
            .collect();
        let spreads: Vec<f64> = folds
            .iter()
            .filter_map(|k| {
                let v: Vec<f64> = by_seed.iter().filter_map(|(_, r)| ibs(&r[k])).collect();
                (v.len() == by_seed.len()).then(|| {
                    v.iter().copied().fold(f64::MIN, f64::max)
                        - v.iter().copied().fold(f64::MAX, f64::min)
                })
            })
            .collect();
        writeln!(
            out,
            "| {} | {} | {} | {} | {:.5} |",
            arm.name(),
            by_seed.len(),
            folds.len(),
            means.join(", "),
            spreads.iter().sum::<f64>() / spreads.len().max(1) as f64
        )?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::write_run;
    use crate::experiment::RunInfo;
    use crate::metrics::{Measured, Metrics};

    fn run(arm: Arm, seed: u64, fold: usize, ibs: f64) -> Run {
        Run {
            arm,
            seed,
            fold: Some((0, fold)),
            permuted: false,
            dataset: String::new(),
            partition: String::new(),
            info: RunInfo {
                steps: 1,
                early_stopping_nll: 0.0,
                parameters: 1,
                truncated_tokens: 0,
                subjects: (1, 1),
            },
            n_test: 1,
            seconds: 0.0,
            metrics: Metrics {
                ibs_0_15: Some(Measured {
                    value: ibs,
                    n: 1,
                    events: 0,
                }),
                ..Metrics::default()
            },
            intervals_10: None,
            reason: None,
        }
    }

    /// Writes one run per fold for each of `seeds`, the IBS rising with the seed's rank.
    fn write_runs(data: &Path, arm: Arm, seeds: &[u64]) {
        for (rank, seed) in seeds.iter().enumerate() {
            for fold in 0..2 {
                let name = format!("{}-s{seed}-r0-k{fold}.json", arm.name());
                let ibs = 0.1 + 0.01 * rank as f64;
                write_run(data, &name, &run(arm, *seed, fold, ibs)).unwrap();
            }
        }
    }

    #[test]
    fn a_run_trained_with_a_drawn_seed_is_in_the_seed_spread() {
        let dir = tempfile::tempdir().unwrap();
        // seed: a value outside any range a section could enumerate
        let drawn = 0xABCD;
        write_runs(dir.path(), Arm::Horizon, &[1, drawn]);
        let text = section(dir.path()).unwrap();
        assert!(
            text.contains(&format!(
                "| horizon | 2 | 2 | s1 0.10000, s{drawn} 0.11000 |"
            )),
            "{text}"
        );
    }

    #[test]
    fn an_arm_with_runs_under_one_seed_has_no_row() {
        let dir = tempfile::tempdir().unwrap();
        write_runs(dir.path(), Arm::Horizon, &[1]);
        write_runs(dir.path(), Arm::Standard, &[1, 2, 3]);
        let text = section(dir.path()).unwrap();
        assert!(!text.contains("| horizon |"), "{text}");
        assert!(text.contains("| standard | 3 | 2 |"), "{text}");
    }
}
