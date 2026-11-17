// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from an agent's own verified
// experience without forgetting what it was taught, for its clients. If
// your team needs expertise in expert iteration and rehearsal for small
// agents, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The data the model improves on: its own verified episodes, mixed with
//! demonstrations it was fine-tuned on (rehearsal), so a few passes over a
//! small set of its own successes teach without overwriting the rest.
//!
//! The same mix is drawn for the spurious-reward control arm, with one
//! difference: its own episodes are chosen by a seeded draw that ignores the
//! verifier, as many as the verifier accepted. A control that the
//! examination would promote means the exam rewards what it should not.

use anyhow::Context as _;
use serde_json::json;
use splinter_sdk::store::stage_ledger::{Commit, StageWork};
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::recipe;
use crate::stages::{write, Run};

/// A world constant: separates the control arm's draw from every other
/// stream under one run seed.
const CONTROL_SALT: u64 = 0x636f_6e74_726f_6c00;

/// A world constant: the stream rehearsal records are drawn from.
const REHEARSAL_SALT: u64 = 0x7265_6865_6172_7300;

impl Run<'_> {
    /// The improvement data of `label`, as unit `<label>/<name>/data`
    /// writing `data/<label>/<name>.jsonl`: the records of the own episodes
    /// (the verified ones, or for a `control` arm as many drawn without the
    /// verifier) and `experience.rehearsal` demonstration records per own
    /// record.
    ///
    /// # Errors
    /// A record file cannot be read or the mix written.
    pub fn experience(
        &mut self,
        label: &str,
        name: &str,
        episodes: &[Commit],
        demonstrations: &Commit,
        control: bool,
    ) -> anyhow::Result<Commit> {
        let demonstrations_commit = demonstrations.clone();
        let demos = demonstrations
            .output("records")
            .context("the demonstrations have no records")?
            .clone();
        let digests = crate::collect::episode_inputs(episodes);
        let mut inputs: Vec<(&str, &Digest)> =
            digests.iter().map(|(n, d)| (n.as_str(), d)).collect();
        inputs.push(("demonstrations", &demos));
        let unit = format!("{label}/{name}/data");
        let slice = json!({ "experience": self.config.experience, "control": control });
        let key = self.key(&slice, &unit, &inputs, recipe::TRAIN, &[]);
        let rel = format!("data/{label}/{name}.jsonl");
        let episodes = episodes.to_vec();
        self.unit(&unit, &key, move |run, _| {
            let with_records: Vec<&Commit> = episodes
                .iter()
                .filter(|c| c.output("records").is_some())
                .collect();
            let accepted = episodes
                .iter()
                .filter(|c| crate::collect::admitted(c))
                .count();
            let chosen: Vec<&Commit> = if control {
                let mut pool = with_records.clone();
                SplitMix64::new(run.seed ^ CONTROL_SALT).shuffle(&mut pool);
                pool.into_iter().take(accepted).collect()
            } else {
                with_records
                    .into_iter()
                    .filter(|c| crate::collect::admitted(c))
                    .collect()
            };
            let reader = run.episode_reader(&format!("{label}/explore"))?;
            let mut own = Vec::new();
            for commit in &chosen {
                own.extend(reader.read(commit, "records")?.lines().map(String::from));
            }
            let demos_path = run.path(&crate::teacher::output_path(
                &demonstrations_commit,
                "records",
            )?);
            let demo_text = std::fs::read_to_string(&demos_path)
                .with_context(|| format!("reading {}", demos_path.display()))?;
            let mut pool: Vec<&str> = demo_text.lines().collect();
            SplitMix64::new(run.seed ^ REHEARSAL_SALT).shuffle(&mut pool);
            let wanted = rehearsal_count(own.len(), run.config.experience.rehearsal);
            let rehearsed: Vec<&str> = pool.into_iter().take(wanted).collect();
            let passed_chosen = chosen
                .iter()
                .filter(|c| crate::collect::admitted(c))
                .count();
            let mut text = String::new();
            for line in own
                .iter()
                .map(String::as_str)
                .chain(rehearsed.iter().copied())
            {
                text.push_str(line);
                text.push('\n');
            }
            write(&run.path(&rel), &text)?;
            Ok(StageWork::new(json!({
                "episodes": chosen.len(), "verified": passed_chosen, "own_records": own.len(),
                "rehearsal_records": rehearsed.len(), "control": control,
            }))
            .output("records", &rel))
        })
    }
}

/// How many rehearsal records go with `own` records at `ratio` per record.
fn rehearsal_count(own: usize, ratio: f64) -> usize {
    if own == 0 || !(ratio.is_finite() && ratio > 0.0) {
        return 0;
    }
    // A ratio is a handful at most, so the product is a small count.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let n = (own as f64 * ratio).round() as usize;
    n
}

#[cfg(test)]
mod tests {
    use super::rehearsal_count;

    #[test]
    fn rehearsal_scales_with_the_own_records_and_is_off_at_zero() {
        assert_eq!(rehearsal_count(10, 1.0), 10);
        assert_eq!(rehearsal_count(10, 2.5), 25);
        assert_eq!(rehearsal_count(10, 0.0), 0);
        assert_eq!(rehearsal_count(0, 3.0), 0);
        assert_eq!(rehearsal_count(10, f64::NAN), 0);
    }
}
