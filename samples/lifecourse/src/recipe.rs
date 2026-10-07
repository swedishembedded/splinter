// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-horizon risk prediction from cohort and
// survey data for its clients. If your team needs expertise in building,
// validating and proving time-to-event models on real health records, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A training recipe: the horizon model's spec as `key=value` overrides of
//! the arm's defaults, so a variant is named, recorded and repeatable.
//!
//! Only what brain's `TimelineSpec` exposes can be set. Weight decay is not
//! exposed (brain fixes it), and asking for it is an error that says so
//! rather than a silent no-op.
//!
//! | Key | Meaning |
//! |---|---|
//! | `arm` | `horizon` (default) or `additive` |
//! | `d_model`, `layers`, `heads`, `d_ff`, `rank`, `value_bins`, `time_bins` | the encoder's shape |
//! | `value_weight` | weight of the masked-value objective |
//! | `max_tokens`, `steps`, `lr`, `batch`, `mask_rate`, `patience`, `eval_interval` | the training procedure |
//! | `knots` | hazard knots in years, colon separated (`0:1:2:5:10:21`) |
//! | `forecasts` | `<per subject>:<weight>` (future measurements; this data has none) |
//! | `next_events` | weight of the next-outcome objective |
//! | `train_share` | fraction of the fold's training subjects used (learning curve) |
//! | `train_seed` | seed of that subsample (default 1) |
//! | `bootstrap` | `true`: train on a bootstrap resample of the training subjects |

use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Context, Result};
use splinter_sdk::model::timeline::{Subject, TimelineConfig, TimelineModel, TimelineSpec};
use splinter_sdk::vocabulary::digest::Digest;

use crate::build::CODES;
use crate::experiment::{early_stopping_share, Arm, RunInfo, BATCH, KNOTS, MAX_TOKENS, STEPS};

/// Subsample seed when a recipe gives a `train_share` and no `train_seed`.
pub const DEFAULT_TRAIN_SEED: u64 = 1;

/// A named set of overrides, kept in the order and spelling given so the
/// record of a run says exactly what was asked.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recipe {
    overrides: BTreeMap<String, String>,
}

fn number<T: std::str::FromStr>(key: &str, value: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|e| anyhow!("{key}={value}: not a valid value ({e})"))
}

impl Recipe {
    /// From `key=value` strings. An unknown key, a repeated key, an
    /// unparsable value or a hyperparameter brain does not expose is an
    /// error naming it.
    pub fn parse(pairs: &[String]) -> Result<Recipe> {
        let mut overrides = BTreeMap::new();
        for pair in pairs {
            let (k, v) = pair
                .split_once('=')
                .ok_or_else(|| anyhow!("{pair:?}: expected key=value"))?;
            if k == "weight_decay" {
                bail!("weight_decay is not exposed by TimelineSpec (brain fixes it at its training default): a brain gap, not settable here");
            }
            if overrides.insert(k.to_string(), v.to_string()).is_some() {
                bail!("{k} is given twice");
            }
        }
        let recipe = Recipe { overrides };
        // Resolve once: every key and value is checked before any training.
        recipe.arm()?;
        recipe.spec(0)?;
        recipe.train_share()?;
        recipe.bootstrap()?;
        Ok(recipe)
    }

    /// The overrides as given, for the run's record.
    pub fn overrides(&self) -> &BTreeMap<String, String> {
        &self.overrides
    }

    fn get<T: std::str::FromStr>(&self, key: &str) -> Result<Option<T>>
    where
        T::Err: std::fmt::Display,
    {
        self.overrides.get(key).map(|v| number(key, v)).transpose()
    }

    /// The arm trained: the set encoder unless `arm=additive`.
    pub fn arm(&self) -> Result<Arm> {
        match self.overrides.get("arm").map(String::as_str) {
            None | Some("horizon") => Ok(Arm::Horizon),
            Some("additive") => Ok(Arm::Additive),
            Some(other) => bail!("arm={other}: only horizon and additive are recipes"),
        }
    }

    /// `(share, seed)` of the training subsample, when one is asked for.
    pub fn train_share(&self) -> Result<Option<(f64, u64)>> {
        let Some(share) = self.get::<f64>("train_share")? else {
            return Ok(None);
        };
        if !(share > 0.0 && share <= 1.0) {
            bail!("train_share={share}: must be in (0, 1]");
        }
        let seed = self.get("train_seed")?.unwrap_or(DEFAULT_TRAIN_SEED);
        Ok(Some((share, seed)))
    }

    /// Whether the training subjects are bootstrap resampled.
    pub fn bootstrap(&self) -> Result<bool> {
        Ok(self.get("bootstrap")?.unwrap_or(false))
    }

    fn shaped(&self) -> Result<Option<TimelineConfig>> {
        const SHAPE: [&str; 8] = [
            "d_model",
            "layers",
            "heads",
            "d_ff",
            "rank",
            "value_bins",
            "time_bins",
            "value_weight",
        ];
        if !SHAPE.iter().any(|k| self.overrides.contains_key(*k)) {
            return Ok(None);
        }
        // The vocabulary size and code count are set from the fitted
        // vocabulary when the spec builds its configuration.
        let mut cfg = TimelineConfig::default_for(0, 0);
        if let Some(v) = self.get("d_model")? {
            cfg.d_model = v;
        }
        if let Some(v) = self.get("layers")? {
            cfg.n_layers = v;
        }
        if let Some(v) = self.get("heads")? {
            cfg.n_heads = v;
        }
        if let Some(v) = self.get("d_ff")? {
            cfg.d_ff = v;
        }
        if let Some(v) = self.get("rank")? {
            cfg.rank = v;
        }
        if let Some(v) = self.get("value_bins")? {
            cfg.value_bins = v;
        }
        if let Some(v) = self.get("time_bins")? {
            cfg.time_bins = v;
        }
        if let Some(v) = self.get("value_weight")? {
            cfg.value_weight = v;
        }
        Ok(Some(cfg))
    }

    fn knots(&self) -> Result<Vec<f32>> {
        let Some(text) = self.overrides.get("knots") else {
            return Ok(KNOTS.to_vec());
        };
        let knots: Vec<f32> = text
            .split(':')
            .map(|k| number("knots", k))
            .collect::<Result<_>>()?;
        if knots.first() != Some(&0.0) || knots.windows(2).any(|w| w[0] >= w[1]) {
            bail!("knots={text}: must start at 0 and increase");
        }
        Ok(knots)
    }

    /// The spec this recipe trains under with `seed`; unset keys keep the
    /// arm's defaults ([`crate::experiment::fit`]).
    pub fn spec(&self, seed: u64) -> Result<TimelineSpec> {
        let known = [
            "arm",
            "d_model",
            "layers",
            "heads",
            "d_ff",
            "rank",
            "value_bins",
            "time_bins",
            "value_weight",
            "max_tokens",
            "steps",
            "lr",
            "batch",
            "mask_rate",
            "patience",
            "eval_interval",
            "knots",
            "forecasts",
            "next_events",
            "train_share",
            "train_seed",
            "bootstrap",
        ];
        if let Some(k) = self.overrides.keys().find(|k| !known.contains(&k.as_str())) {
            bail!("{k}: not a recipe key (known: {})", known.join(", "));
        }
        let mut spec = TimelineSpec::new(CODES, CODES)
            .additive(self.arm()? == Arm::Additive)
            .knots(self.knots()?)
            .max_tokens(self.get("max_tokens")?.unwrap_or(MAX_TOKENS))
            .batch(self.get("batch")?.unwrap_or(BATCH))
            .steps(self.get("steps")?.unwrap_or(STEPS))
            .seed(seed);
        if let Some(cfg) = self.shaped()? {
            spec = spec.shape(cfg);
        }
        if let Some(v) = self.get("lr")? {
            spec = spec.lr(v);
        }
        if let Some(v) = self.get("mask_rate")? {
            spec = spec.mask_rate(v);
        }
        if let Some(v) = self.get("patience")? {
            spec = spec.patience(v);
        }
        if let Some(v) = self.get("eval_interval")? {
            spec = spec.eval_interval(v);
        }
        if let Some(text) = self.overrides.get("forecasts") {
            let (n, w) = text
                .split_once(':')
                .ok_or_else(|| anyhow!("forecasts={text}: expected <per subject>:<weight>"))?;
            spec = spec.forecasts(number("forecasts", n)?, number("forecasts", w)?);
        }
        if let Some(w) = self.get("next_events")? {
            spec = spec.next_events(CODES, w);
        }
        Ok(spec)
    }
}

/// The subjects of `ids` kept at `share`: each subject is in when a hash of
/// the seed and its id is below the share, so a smaller share is always a
/// subset of a larger one (nested learning-curve points), and every fold of
/// every repeat draws independently. There is no group id in this data:
/// the subject is the group the partition keeps together.
pub fn subsample<'a>(ids: &[&'a str], share: f64, seed: u64) -> Vec<&'a str> {
    ids.iter()
        .copied()
        .filter(|id| {
            let h = Digest::of(format!("subsample:{seed}:{id}").as_bytes());
            let top = u64::from_str_radix(&h.hex()[..13], 16).unwrap_or(u64::MAX);
            (top as f64) < share * (1u64 << 52) as f64
        })
        .collect()
}

/// `subjects` resampled with replacement, `subjects.len()` draws, from `seed`.
fn bootstrap(subjects: Vec<Subject>, seed: u64) -> Vec<Subject> {
    let n = subjects.len();
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            subjects[(state >> 33) as usize % n].clone()
        })
        .collect()
}

/// Train `recipe` on `subjects` with `seed`: a tenth of them (by the same
/// rule as every arm) are held out for early stopping and never trained
/// on; a bootstrap resamples only what remains for training.
pub fn fit(recipe: &Recipe, subjects: &[&Subject], seed: u64) -> Result<(TimelineModel, RunInfo)> {
    let arm = recipe.arm()?;
    let (held, train): (Vec<Subject>, Vec<Subject>) = subjects
        .iter()
        .map(|s| arm.view(s))
        .partition(|s| early_stopping_share(&s.subject_id, seed));
    if train.is_empty() || held.is_empty() {
        bail!(
            "too few subjects to train ({} + {})",
            train.len(),
            held.len()
        );
    }
    let train = if recipe.bootstrap()? {
        bootstrap(train, seed)
    } else {
        train
    };
    let (model, report) =
        TimelineModel::train(&train, &held, &recipe.spec(seed)?).context("training the recipe")?;
    let info = RunInfo {
        steps: report.steps,
        early_stopping_nll: report.held_out_event_nll,
        parameters: report.parameters,
        truncated_tokens: report.truncated_tokens,
        subjects: (train.len(), held.len()),
    };
    Ok((model, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(p: &[&str]) -> Vec<String> {
        p.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_recipe_refuses_what_it_cannot_set() {
        assert!(Recipe::parse(&pairs(&["d_model=32", "layers=1", "knots=0:1:5:21"])).is_ok());
        let wd = Recipe::parse(&pairs(&["weight_decay=0.3"])).unwrap_err();
        assert!(wd.to_string().contains("not exposed"), "{wd}");
        assert!(
            Recipe::parse(&pairs(&["dmodel=32"])).is_err(),
            "unknown key"
        );
        assert!(Recipe::parse(&pairs(&["lr=fast"])).is_err(), "bad value");
        assert!(
            Recipe::parse(&pairs(&["lr=1e-3", "lr=1e-2"])).is_err(),
            "repeat"
        );
        assert!(
            Recipe::parse(&pairs(&["knots=1:2"])).is_err(),
            "no zero knot"
        );
        assert!(Recipe::parse(&pairs(&["train_share=1.5"])).is_err());
        assert!(Recipe::parse(&pairs(&["arm=standard"])).is_err());
        assert_eq!(Recipe::parse(&[]).unwrap().arm().unwrap(), Arm::Horizon);
    }

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("nhanes-2003-{i}")).collect()
    }

    #[test]
    fn subsamples_are_deterministic_nested_and_near_their_share() {
        let owned = ids(4000);
        let all: Vec<&str> = owned.iter().map(String::as_str).collect();
        let (a, b) = (subsample(&all, 0.25, 7), subsample(&all, 0.5, 7));
        assert_eq!(a, subsample(&all, 0.25, 7));
        assert!(a.iter().all(|id| b.contains(id)), "a smaller share nests");
        assert!((a.len() as f64 - 1000.0).abs() < 100.0, "{}", a.len());
        assert_ne!(a, subsample(&all, 0.25, 8), "the seed changes the draw");
        assert_eq!(subsample(&all, 1.0, 7).len(), all.len());
    }
}
