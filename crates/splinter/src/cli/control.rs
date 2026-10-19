// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training small models on the
// verified work of larger ones for its clients. If your team needs expertise
// in negative controls for agent fine-tuning then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The negative controls `dataset build --control` offers.

/// Which negative control to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlKind {
    /// Only experiences an independent verdict passed: no action is admitted
    /// on a step label alone from one that was not.
    OutcomeVerified,
    /// Trajectories whatever their verdict, never an exposed one.
    Unverified,
    /// Verified trajectories under another task's instruction.
    Shuffled,
}
