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

use splinter_core::experience::Task;
pub use splinter_core::family::rank;

/// The family `task` declares: its first declared concept (for a code task,
/// the file its fault was seeded in); `None` for a task that declares none.
#[must_use]
pub fn family_of_task(task: &Task) -> Option<&str> {
    task.concepts.first().map(String::as_str)
}
