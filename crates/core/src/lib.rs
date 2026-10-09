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
//! * [`claim`] - what a person taught an agent in conversation, as a
//!   proposal, a gated claim tied to the words that taught it, and the
//!   ledger entry that rules on it.
//! * [`annotation`] - what is said about an experience after the fact:
//!   verdicts, step labels and relations. Never a rewrite of the experience.
//!
//! * [`chat`] and [`prompt`] - the shapes a conversation is written in, and the
//!   one system turn every model run on a task is sent.
//! * [`evidence`] - how an executable verdict's evidence reads back.
//! * [`model_ref`] and [`role`] - how a command names a model, and which
//!   model plays which role in a run.
//! * [`dataset`], [`release`] and [`training`] - what names a dataset and a
//!   release, and the record of how a candidate was trained.
//! * [`kinds`] and [`selfcontained`] - the names tasks travel under, and the
//!   rule that an instruction must stand on its own.
//! * [`longitudinal`] - a participant's irregular history as a file states it
//!   and as it is kept: opaque keys in place of identifiers, provenance on
//!   every item.
//! * [`speech`] - how a speaking persona is described, and the label a
//!   synthetic voice always carries.
//! * [`speech_lesson`] - what a user teaches a speaking persona about how to
//!   speak, read from what they said and kept as lessons with a stated
//!   objective.
//! * [`speech_bundle`] - the parts of a speaking persona and the rule that each
//!   was made for the others it is attached to.
//! * [`spoken`] - recordings of text, each attached to the record it renders, and
//!   a split that tests on voices and questions never trained on.
//! * [`terms`] - the terms data came under, combined most-restrictively
//!   over sources and carried to datasets and releases.
//!
//! This crate performs no I/O. It knows no store, no model and no agent
//! runtime: it is the language they share, and the invariants every record
//! in it keeps no matter who builds one.

#![warn(missing_docs)]

pub mod annotation;
pub mod chat;
pub mod claim;
pub mod clock;
pub mod dataset;
pub mod digest;
pub mod evidence;
pub mod experience;
pub mod kinds;
pub mod longitudinal;
pub mod model_ref;
pub mod prompt;
pub mod release;
pub mod role;
pub mod selfcontained;
pub mod source;
pub mod speech;
pub mod speech_bundle;
pub mod speech_lesson;
pub mod spoken;
pub mod terms;
pub mod training;
