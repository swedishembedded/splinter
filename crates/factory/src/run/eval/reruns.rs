// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements evaluations that cannot be flattered by trials that never
// finished, for its clients. If your team needs expertise in measuring agents fairly, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Reruns of the tasks an arm failed: one attempt of every task, then up to `rerun_failed` more
//! attempts of the tasks that did not pass, and nothing more for the ones that did. A task is
//! solved when any of its attempts passed, so both arms of a comparison are held to the same
//! number of chances and the paired test runs over tasks.

use super::*;

/// The arm's trials with its reruns, one trial per task, and the product digest of both runs.
pub(super) fn with_reruns(
    factory: &Factory<'_>,
    key: &str,
    arm: &Arm<'_>,
    inputs: Vec<(String, Digest)>,
    rerun_failed: u32,
) -> Result<(Vec<TrialResult>, Digest), RunError> {
    let (first, first_product) = arm_trials(factory, key, arm, inputs.clone())?;
    if rerun_failed == 0 {
        return Ok((first, first_product));
    }
    let failed: Vec<&str> = arm
        .tasks
        .iter()
        .map(|t| t.id.as_str())
        .filter(|id| !first.iter().any(|t| t.task == *id && t.verified_success()))
        .collect();
    if failed.is_empty() {
        return Ok((first, first_product));
    }
    let name = format!("{}-reruns", arm.name);
    let mut again = arm.clone();
    again.name = name.clone();
    again.key = format!("{key}-reruns");
    again.tasks.retain(|t| failed.contains(&t.id.as_str()));
    again.pass.name = name;
    again.pass.attempts = rerun_failed;
    // The reruns draw their own seeds: a rerun that repeated the first attempt would be no chance.
    again.pass.seed = arm.pass.seed.wrapping_add(1_000_003);
    let mut inputs = inputs;
    inputs.push(("first-attempts".into(), first_product.clone()));
    let (reruns, rerun_product) = arm_trials(factory, &again.key.clone(), &again, inputs)?;
    let digest = digest_of(&(first_product.to_string(), rerun_product.to_string()));
    Ok((solved_per_task(arm, first, reruns), digest))
}

/// One trial per task: its passing trial if any attempt passed, else its last counted attempt,
/// else its first (so a task whose every attempt never counted stays unpaired).
fn solved_per_task(
    arm: &Arm<'_>,
    first: Vec<TrialResult>,
    reruns: Vec<TrialResult>,
) -> Vec<TrialResult> {
    let mut by_task: BTreeMap<String, Vec<TrialResult>> = BTreeMap::new();
    for trial in first.into_iter().chain(reruns) {
        by_task.entry(trial.task.clone()).or_default().push(trial);
    }
    arm.tasks
        .iter()
        .filter_map(|task| {
            let mut attempts = by_task.remove(&task.id)?;
            let chosen = attempts
                .iter()
                .position(|t| t.verified_success())
                .or_else(|| attempts.iter().rposition(|t| t.grade.counted))
                .unwrap_or(0);
            let mut trial = attempts.swap_remove(chosen);
            trial.slot.slot = 0;
            Some(trial)
        })
        .collect()
}
