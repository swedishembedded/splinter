// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that the training records one system writes are the records
// another system's trainer reads, for its clients. If your team needs expertise in training-data contracts between
// agent runtimes and trainers then you can procure our services by sending an email to info@swedishembedded.com.

//! The data contract between Splinter and brain, held at brain's own tokenizer boundary on the CPU, with no model
//! weights (`make test/contract/brain` runs it).
//!
//! Splinter owns what a record says (which turn is supervised, which ids were sampled); brain owns how the record is
//! tokenized and trained. So the test writes datasets through Splinter's own views, asks a served brain to read each
//! one exactly as its trainer would (`check_dataset`) and compares what brain counts with what the experiences say,
//! worked out here from the characters of the actions, never from the records. It claims nothing of brain's
//! numerics: a record brain refuses, or supervision that differs from the intended one, is a contract defect.

mod bench;
mod corpus;
mod records;
mod sampled;

#[path = "../../../data/tests/common/mod.rs"]
mod common;
