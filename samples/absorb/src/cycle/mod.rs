// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it without
// forgetting what it knew, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The multi-night cycle: the nights a user's corrections are absorbed, the
//! evaluation of the release in effect after each night, and the report whose
//! verdict is computed by code from a rule registered before the first
//! session.

pub mod analysis;
pub mod cli;
pub mod driver;
pub mod eval;
#[cfg(test)]
mod fake;
#[cfg(test)]
mod fixtures;
pub mod grade;
pub mod items;
pub mod plan;
pub mod prereg;
pub mod report;
pub mod stats;
pub mod verdict;
pub mod world;

#[cfg(test)]
mod analysis_tests;
#[cfg(test)]
mod driver_tests;
#[cfg(test)]
mod items_tests;
#[cfg(test)]
mod plan_tests;
#[cfg(test)]
mod prereg_tests;
