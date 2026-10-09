// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The first part of the absorb validation protocol: the fact pool, the
//! screening of the day-0 policy, the sealed probes and their leakage guard,
//! and the recording of live sessions.

pub mod answers;
pub mod build;
pub mod facts;
pub mod grading;
pub mod keys;
pub mod policy;
pub mod probes;
pub mod roles;
pub mod runtime;
pub mod screen;
pub mod seal;
pub mod unknowns;
pub mod writer;
