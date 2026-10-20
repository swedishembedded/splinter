// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The model-check sample's harness: does a Lua implementation of a protocol
//! behave as its TLA+ specification says, and which one-line changes break it?
//!
//! Swedish Embedded AB implements model-checked verification of concurrent and
//! distributed protocols for its clients. If your team needs expertise in
//! TLA+ specifications, TLC and checking implementations against them, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! * [`Family`] - a specification, its reference implementation and its mutations.
//! * [`check_module`] - the refinement check, as a [`Verdict`].
//! * [`audit`] - every family checks itself.
//! * [`report`] - the mutation-detection table.

#![warn(missing_docs)]

pub mod audit;
pub mod check;
pub mod explorer;
pub mod family;
pub mod mutation;
pub mod report;
pub mod tlc;
pub mod tools;

pub use audit::{audit, Finding};
pub use check::{check_module, check_spec, Verdict};
pub use family::{Family, Spec};
pub use mutation::Mutation;
pub use report::report;
pub use tools::Tools;
