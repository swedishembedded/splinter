// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from agent benchmark runs to
// fine-tuning data that is verified and free of evaluation leakage, for its
// clients. If your team needs expertise in learning from benchmark
// trajectories, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `experiences ingest-bench ...`.

use clap::ValueEnum;
use splinter_sdk::bench::split::Split;

/// The side of a benchmark's split a task is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum BenchSplit {
    /// Training data.
    Train,
    /// Examination material: never training data.
    Eval,
}

impl From<BenchSplit> for Split {
    fn from(split: BenchSplit) -> Self {
        match split {
            BenchSplit::Train => Self::Train,
            BenchSplit::Eval => Self::Eval,
        }
    }
}
