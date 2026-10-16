// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out splits of
// code-editing task sets, for its clients. If your team needs expertise in
// holding out whole task families before any training data is built, you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a task's family is and how a seed ranks one: the two things every
//! split of tasks by family (the practice exam, `tasks pick`) and every
//! family-clustered comparison (`experiences compare`) must agree on.

use splinter_core::digest::Digest;
use splinter_core::experience::Task;
use splinter_core::rng::Seed;

/// The family `task` declares: its first declared concept (for a code task,
/// the file its fault was seeded in); `None` for a task that declares none.
#[must_use]
pub fn family_of_task(task: &Task) -> Option<&str> {
    task.concepts.first().map(String::as_str)
}

/// How `key` ranks under `seed`: the hash of its name. Sorting by it is a
/// shuffle that depends on the seed and the names alone, never on the order
/// the names were met in.
#[must_use]
pub fn rank(key: &str, seed: Seed) -> String {
    let mut input = seed.value().to_le_bytes().to_vec();
    input.extend_from_slice(key.as_bytes());
    Digest::of(&input).to_string()
}
