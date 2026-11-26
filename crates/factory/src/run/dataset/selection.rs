// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training sets curated to a fixed budget per
// task, every choice reproducible from a seed, for its clients. If your team
// needs expertise in curating agent fine-tuning data, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which admitted trials of a task an SFT dataset keeps under a per-task
//! limit (`per_task`, `prefer`).

use std::collections::BTreeMap;

use serde_json::json;
use splinter_core::digest::digest_of;

use super::{Admitted, Excluded};
use crate::ops::Prefer;

/// `admitted` with at most `limit` trials of each task, chosen by `prefer`
/// and `seed` ([`kept_per_task`]); every other one is added to `excluded` as
/// `per_task_limit`.
pub(super) fn limit_per_task(
    admitted: Vec<Admitted>,
    limit: usize,
    prefer: Prefer,
    seed: u64,
    excluded: &mut Vec<Excluded>,
) -> Vec<Admitted> {
    let candidates: Vec<Candidate<'_>> = admitted
        .iter()
        .map(|a| Candidate {
            task: &a.trial.task,
            trial: format!("{}/{}", a.source, a.trial.slot),
            actions: a.taken,
        })
        .collect();
    let keep = kept_per_task(&candidates, limit, prefer, seed);
    let mut offered: BTreeMap<String, usize> = BTreeMap::new();
    for a in &admitted {
        *offered.entry(a.trial.task.clone()).or_default() += 1;
    }
    let mut kept = Vec::new();
    for (admitted, keep) in admitted.into_iter().zip(keep) {
        if keep {
            kept.push(admitted);
            continue;
        }
        excluded.push(Excluded {
            detail: format!(
                "{} actions; {limit} of the task's {} admitted trials kept ({})",
                admitted.taken,
                offered[&admitted.trial.task],
                json!(prefer).as_str().unwrap_or_default()
            ),
            pass: admitted.source,
            slot: admitted.trial.slot.to_string(),
            task: admitted.trial.task,
            reason: "per_task_limit".into(),
        });
    }
    kept
}

/// One admitted trial as a per-task limit sees it.
pub(super) struct Candidate<'a> {
    /// Its task.
    pub(super) task: &'a str,
    /// What tells it from every other trial (its pass and slot).
    pub(super) trial: String,
    /// The actions it took.
    pub(super) actions: usize,
}

/// For each of `candidates`, whether a limit of `limit` trials per task keeps
/// it: under `fewest-actions` the trials with the fewest actions, ties drawn
/// by `seed`; under `random` trials drawn by `seed`. A trial's draw depends
/// only on the seed and the trial, so the order the candidates come in
/// changes nothing.
pub(super) fn kept_per_task(
    candidates: &[Candidate<'_>],
    limit: usize,
    prefer: Prefer,
    seed: u64,
) -> Vec<bool> {
    let mut tasks: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        tasks.entry(candidate.task).or_default().push(index);
    }
    let mut kept = vec![false; candidates.len()];
    for indices in tasks.into_values() {
        let mut ranked: Vec<(usize, u64, &str, usize)> = indices
            .into_iter()
            .map(|i| {
                let c = &candidates[i];
                let draw = digest_of(&(seed, c.task, &c.trial)).leading_u64();
                let actions = match prefer {
                    Prefer::FewestActions => c.actions,
                    Prefer::Random => 0,
                };
                (actions, draw, c.trial.as_str(), i)
            })
            .collect();
        ranked.sort_unstable();
        for (.., index) in ranked.into_iter().take(limit) {
            kept[index] = true;
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates(actions: &[(&'static str, usize)]) -> Vec<Candidate<'static>> {
        actions
            .iter()
            .enumerate()
            .map(|(i, (task, actions))| Candidate {
                task,
                trial: format!("pass/{i}"),
                actions: *actions,
            })
            .collect()
    }

    #[test]
    fn at_most_the_limit_is_kept_per_task_and_the_fewest_actions_first() {
        let all = candidates(&[("a", 5), ("a", 2), ("a", 9), ("a", 3), ("b", 7)]);
        // seed: a few seeds, each must give the same choice where nothing ties.
        for seed in [1, 2, 3] {
            let kept = kept_per_task(&all, 2, Prefer::FewestActions, seed);
            assert_eq!(kept, [false, true, false, true, true]);
            let kept = kept_per_task(&all, 1, Prefer::Random, seed);
            assert_eq!(kept.iter().filter(|k| **k).count(), 2, "one of each task");
            assert!(kept[4]);
        }
    }

    #[test]
    fn ties_and_random_draws_are_the_seeds_whatever_the_order() {
        let all = candidates(&[("a", 4), ("a", 4), ("a", 4), ("a", 4), ("a", 9)]);
        let draws: Vec<Vec<bool>> = (0..16)
            .map(|seed| kept_per_task(&all, 1, Prefer::FewestActions, seed))
            .collect();
        assert!(draws.iter().all(|k| !k[4]), "never the longest");
        assert!(draws.iter().any(|k| k != &draws[0]), "a tie is drawn");
        let random: Vec<Vec<bool>> = (0..16)
            .map(|seed| kept_per_task(&all, 1, Prefer::Random, seed))
            .collect();
        assert!(random.iter().any(|k| k[4]), "random may keep the longest");
        // The same seed keeps the same trials when they come in another order.
        let mut reversed = candidates(&[("a", 4), ("a", 4), ("a", 4), ("a", 4), ("a", 9)]);
        reversed.reverse();
        for seed in 0..16 {
            let kept = kept_per_task(&reversed, 1, Prefer::Random, seed);
            let forward: Vec<bool> = random[seed as usize].iter().rev().copied().collect();
            assert_eq!(kept, forward);
        }
    }
}
