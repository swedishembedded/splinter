// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that cannot be passed by a
// model that games its exam, for its clients. If your team needs expertise
// in detecting reward hacking and benchmark contamination in agent
// evaluations, you can procure our services by sending an email to
// info@swedishembedded.com.

//! What a canary check measured, as a gate report of the earlier release
//! rule recorded it: canary tasks are tasks no honest solution can pass, and a
//! candidate that passed one was not released. Nothing measures canaries any
//! more; the record stays readable so those releases load.

use serde::{Deserialize, Serialize};

/// What the canary check measured.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Canaries {
    /// Canaries handed to the gate.
    pub tasks: usize,
    /// Of those, the ones graded.
    pub graded: usize,
    /// The canaries the candidate passed: each a cheat.
    pub passed_tasks: Vec<String>,
}
