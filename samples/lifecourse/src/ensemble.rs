// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements risk models that report how much their own
// training varies for its clients. If your team needs expertise in model
// uncertainty for clinical prediction, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A secondary analysis, not one of the pre-registered criteria: the locked
//! test scored by an ensemble of one arm's models, trained with different
//! seeds on the same subjects as `final`.
//!
//! The ensemble's probabilities are the members' means. The first member is
//! the model `final` saved, trained with the pre-registered seed; the others
//! are trained with seeds derived from the ensemble's own seed, so a rerun
//! with the same `--seed` repeats the ensemble and another seed makes another.
//! How far the members disagree about a subject's ten-year risk is the part of
//! its uncertainty that comes from training alone - the spread a single run
//! hides.

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde::Serialize;
use splinter_sdk::model::timeline::{Prediction, Subject, TimelineModel};
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::commands::{frozen, locked_split, nhanes_terms, require_training, PREREGISTERED_SEED};
use crate::experiment::{fit, Arm, Training, STEPS};
use crate::metrics::{evaluate, Metrics};

/// The horizon the members' disagreement is reported at, in years.
const SPREAD_HORIZON: f64 = 10.0;

#[derive(Serialize)]
struct EnsembleRun {
    arm: Arm,
    members: u64,
    /// The seed the members after the first were derived from.
    seed: u64,
    /// Every member's training seed, in order.
    member_seeds: Vec<u64>,
    dataset: String,
    partition: String,
    n_test: usize,
    metrics: Metrics,
    /// Per subject, the largest minus the smallest member's all-cause risk by
    /// ten years: the mean and the 90th percentile over the locked test.
    spread_10: (f64, f64),
}

/// The training seeds of an ensemble of `members`: the pre-registered seed
/// first, so the model `final` saved is a member, then seeds derived from the
/// ensemble's `seed`, all distinct.
pub fn member_seeds(seed: u64, members: u64) -> Vec<u64> {
    let mut rng = SplitMix64::new(seed);
    let mut seeds = vec![PREREGISTERED_SEED];
    while (seeds.len() as u64) < members {
        let next = rng.next_u64();
        if !seeds.contains(&next) {
            seeds.push(next);
        }
    }
    seeds
}

/// Score the locked test with `members` models of `arm`: the one `final`
/// saved and the rest trained with seeds derived from `seed`. A member whose
/// model was already saved is loaded rather than trained again.
pub fn ensemble(data: &Path, arm: Arm, members: u64, seed: u64) -> Result<()> {
    if members < 2 {
        bail!("an ensemble needs at least two members, not {members}");
    }
    let f = frozen(data)?;
    require_training(&nhanes_terms())?;
    let name = format!("{}-ens{members}-locked.json", arm.name());
    let path = data.join("runs").join(&name);
    if path.exists() {
        bail!("{name}: the locked test was already scored by this ensemble");
    }
    let (train_ids, test_ids) = locked_split(&f);
    let train: Vec<&Subject> = train_ids.iter().map(|id| &f.subjects[*id]).collect();
    let test: Vec<Subject> = test_ids
        .iter()
        .map(|id| arm.view(&f.subjects[*id]))
        .collect();
    let mut by_member: Vec<Vec<Prediction>> = Vec::new();
    let member_seeds = member_seeds(seed, members);
    for member in &member_seeds {
        let saved = data
            .join("runs")
            .join(format!("{}-s{member}-locked-model", arm.name()));
        let model = if saved.join("manifest.json").is_file() {
            println!("member seed {member}: the model final saved");
            TimelineModel::load(&saved)?
        } else {
            println!("member seed {member}: training");
            fit(
                arm,
                &train,
                &Training {
                    seed: *member,
                    steps: STEPS,
                },
            )?
            .0
        };
        by_member.push(model.predict(&test)?);
    }
    let preds: Vec<Prediction> = (0..test.len())
        .map(|i| {
            let parts: Vec<Prediction> = by_member.iter().map(|m| m[i].clone()).collect();
            Prediction::ensemble(&parts)
                .ok_or_else(|| anyhow!("members disagree on codes or horizon"))
        })
        .collect::<Result<_>>()?;
    let mut spread: Vec<f64> = preds
        .iter()
        .map(|p| {
            let risks: Vec<f64> = p
                .members()
                .iter()
                .map(|m| 1.0 - m.survival(SPREAD_HORIZON))
                .collect();
            risks.iter().copied().fold(f64::MIN, f64::max)
                - risks.iter().copied().fold(f64::MAX, f64::min)
        })
        .collect();
    spread.sort_by(f64::total_cmp);
    let run = EnsembleRun {
        arm,
        members,
        seed,
        member_seeds,
        dataset: f.digests.0.clone(),
        partition: f.digests.1.clone(),
        n_test: test.len(),
        metrics: evaluate(&test, &preds, &f.horizons),
        spread_10: (
            spread.iter().sum::<f64>() / spread.len() as f64,
            spread[spread.len() * 9 / 10],
        ),
    };
    std::fs::write(&path, serde_json::to_vec_pretty(&run)?)?;
    println!("{name}: {}", serde_json::to_string_pretty(&run.metrics)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two different seeds, drawn.
    fn two_seeds() -> (u64, u64) {
        let a = splinter_sdk::seed::resolve("seed", None);
        (a, a.wrapping_add(1))
    }

    #[test]
    fn the_first_member_is_the_one_final_saved_and_the_rest_follow_the_ensemble_seed() {
        let (a, b) = two_seeds();
        let seeds = member_seeds(a, 5);
        assert_eq!(seeds.len(), 5);
        assert_eq!(seeds[0], PREREGISTERED_SEED);
        let mut distinct = seeds.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 5, "no member is trained twice: {seeds:?}");
        assert_eq!(member_seeds(a, 5), seeds, "the seed reproduces the members");
        assert_ne!(
            member_seeds(b, 5)[1..],
            seeds[1..],
            "another seed makes another ensemble"
        );
    }

    #[test]
    fn a_larger_ensemble_keeps_the_members_of_a_smaller_one() {
        let (a, _) = two_seeds();
        assert_eq!(member_seeds(a, 3), member_seeds(a, 6)[..3]);
    }
}
