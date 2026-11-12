// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! Goal-directed selection: from a raw corpus and a goal to the sections
//! worth learning, each choice with its reason.
//!
//! * [`corpus`] - documents with provenance, cut into candidate sections.
//! * [`facts`] - the keyed values a section states.
//! * [`gaps`] - a goal, its probes, and the gaps its failed probes show.
//! * [`select`] - relevance to the gaps, conflicts, novelty to the
//!   participant, duplicates and a word budget, one decision per section.
//! * [`synthetic`] - labelled corpora (relevant, irrelevant, outdated,
//!   misleading, already known) to measure a selection against.
//!
//! The loop that probes, selects, learns and probes again, and the
//! comparison against the other ways to spend a budget, is a pipeline's
//! (`splinter-pipelines`); nothing here runs a model.

pub mod corpus;
pub mod facts;
pub mod gaps;
pub mod select;
pub mod synthetic;
