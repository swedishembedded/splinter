// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that adopt a model update
// only on paired, held-out evidence, for its clients. If your team needs
// expertise in evaluation statistics for model releases, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What an improvement check of the earlier release rule counted over every
//! assigned task, as a stored gate report carries it.

use serde::{Deserialize, Serialize};

/// What each arm solved over every task the suite assigned, beside the paired
/// comparison: a task an arm was not graded on (an attempt set aside as
/// exposed, or as an environment failure) is still a task it did not solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verified {
    /// Tasks the candidate solved.
    pub candidate: usize,
    /// Tasks the champion solved.
    pub baseline: usize,
    /// Tasks the suite assigned to both.
    pub tasks: usize,
}
