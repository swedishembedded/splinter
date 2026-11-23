// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements crash-safe controllers and immutable result
// stores for long-running automation, for its clients. If your team needs
// expertise in durable workflow state and content-addressed artifacts, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Durable state for the learning factory: what a run has decided, and the
//! immutable products its stages made.
//!
//! Both parts live under one run directory and neither touches the
//! experience database:
//!
//! ```text
//! <run>/
//!   blobs/sha256/<hex>     product files, once per content, read-only
//!   products/<stage>.json  committed product manifests (the commit point)
//!   <view>/                readable trees of a product, hard-linked blobs
//!   staging/<id>/          where a stage writes before it publishes
//!   pending/<stage>.json   a publish in flight, for crash recovery
//!   state/controller.db    the authoritative run state (SQLite, WAL)
//!   state/events.jsonl     one line per state change, appended after commit
//! ```
//!
//! * [`products`] - the content-addressed, immutable product store. A product
//!   exists exactly when its manifest file does.
//! * [`runstate`] - the controller's state: a fenced lease, stages with
//!   optimistic versions, and the slots and attempts of rollouts.

pub mod products;
pub mod runstate;

mod fsutil;
