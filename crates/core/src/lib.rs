// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose every record is
// content-addressed and traceable to its source, for its clients. If your
// team needs expertise in training-data provenance or reproducible
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Splinter's vocabulary: what the rest of the system talks about.
//!
//! * [`digest`] - content addresses, and the canonical form they are
//!   computed over.
//! * [`clock`] - the UTC clock records are stamped by, injected so a test
//!   fixes it.
//! * [`source`] - what Splinter learns from: documents, repositories and
//!   command runs, their parts addressed by content.
//! * [`experience`] - the task a solver is given, and the immutable,
//!   content-addressed record of an attempt at it.
//! * [`annotation`] - what is said about an experience after the fact:
//!   verdicts, step labels and relations. Never a rewrite of the experience.
//!
//! This crate performs no I/O. It knows no store, no model and no agent
//! runtime: it is the language they share, and the invariants every record
//! in it keeps no matter who builds one.

#![warn(missing_docs)]

pub mod annotation;
pub mod clock;
pub mod digest;
pub mod experience;
pub mod source;
