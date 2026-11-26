// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements negative controls that tell what a model
// learned from its data apart from mere exposure to its format, for its
// clients. If your team needs expertise in controlled fine-tuning
// experiments, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The `shuffled-requests` control: every record of a task is given the
//! request (its first user message) of another admitted task, by a
//! derangement of the tasks drawn by the stage's seed, and keeps everything
//! else. The records keep the tool-use format and the actions but lose the
//! mapping from a request to what answers it.

use std::collections::BTreeMap;

use splinter_core::digest::digest_of;
use splinter_data::RecordBody;

use super::Admitted;

/// A permutation of `0..n` that moves every element (Sattolo's algorithm,
/// each choice a digest of `seed` and the step): one cycle through all `n`.
/// `n` is at least 2.
pub(super) fn derangement(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (digest_of(&(seed, i)).leading_u64() % i as u64) as usize;
        order.swap(i, j);
    }
    order
}

/// Gives every record of `admitted` the request of the task a derangement of
/// the tasks drawn by `seed` maps its task to; the request of a task is the
/// first user message of its first record. Returns, for each record, the task
/// its request is from.
///
/// # Errors
/// Fewer than two tasks, or a record with no user message.
pub(super) fn shuffle_requests(
    admitted: &mut [Admitted],
    seed: u64,
) -> Result<Vec<String>, String> {
    let mut requests: BTreeMap<String, splinter_core::chat::WireMessage> = BTreeMap::new();
    for a in admitted.iter() {
        if !requests.contains_key(&a.trial.task) {
            let request = first_user(&a.record.body)
                .ok_or_else(|| format!("{} has no user message to swap", a.trial.task))?;
            requests.insert(a.trial.task.clone(), request.clone());
        }
    }
    if requests.len() < 2 {
        return Err(format!(
            "a shuffled-requests control needs at least two tasks; the admitted trials are of {}",
            requests.len()
        ));
    }
    let tasks: Vec<String> = requests.keys().cloned().collect();
    let to = derangement(tasks.len(), seed);
    let donor: BTreeMap<&str, &str> = tasks
        .iter()
        .zip(&to)
        .map(|(task, &other)| (task.as_str(), tasks[other].as_str()))
        .collect();
    let mut given = Vec::with_capacity(admitted.len());
    for a in admitted.iter_mut() {
        let from = donor[a.trial.task.as_str()];
        let RecordBody::Chat { messages, .. } = &mut a.record.body else {
            return Err(format!("{} is not a chat record", a.trial.task));
        };
        let at = messages
            .iter()
            .position(|m| m.role == "user")
            .ok_or_else(|| format!("{} has no user message to swap", a.trial.task))?;
        messages[at] = requests[from].clone();
        given.push(from.to_owned());
    }
    Ok(given)
}

fn first_user(body: &RecordBody) -> Option<&splinter_core::chat::WireMessage> {
    match body {
        RecordBody::Chat { messages, .. } => messages.iter().find(|m| m.role == "user"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::derangement;

    #[test]
    fn a_derangement_moves_every_task_and_is_the_seeds() {
        for n in 2..9 {
            let drawn: Vec<Vec<usize>> = (0..32).map(|seed| derangement(n, seed)).collect();
            for order in &drawn {
                let mut sorted = order.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "a permutation");
                assert!(order.iter().enumerate().all(|(i, &j)| i != j), "{order:?}");
            }
            assert_eq!(derangement(n, 7), drawn[7]);
            if n > 2 {
                assert!(drawn.iter().any(|d| d != &drawn[0]), "the seed draws it");
            }
        }
    }
}
