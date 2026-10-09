// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in continual
// learning from user feedback, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: the first three stages of learning from a person's own sessions,
//! each runnable and inspectable alone over the stores and the run record.
//!
//! * `intake` records an ATIF session once, refuses what the training
//!   projection refuses with the reason, and stores nothing secret.
//! * `extract` has a model propose claims per session, with the quotes that
//!   support them; a reply that stays unusable is reported, not guessed at.
//! * `gate` rules on the proposals by code, keeps every ruling, and says why
//!   each refusal was made.
//!
//! The extractor is a scripted model; nothing here trains or uses a device.

mod extract;
mod fixtures;
mod gate;
mod intake;
