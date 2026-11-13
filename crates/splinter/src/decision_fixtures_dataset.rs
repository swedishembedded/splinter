// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `dataset decision-fixtures`: labelled decision fixtures from an ablation
//! benchmark's run records, as a JSON-lines file.

use std::path::Path;

use serde::Serialize;
use splinter_sdk::decision_fixtures::{build, to_jsonl, Split, SplitPlan};
use splinter_sdk::Error;

use crate::output::{emit, Report};

/// What was written.
#[derive(Serialize)]
struct Written {
    out: String,
    examples: usize,
    train: usize,
    validation: usize,
    test: usize,
}

impl Report for Written {
    fn human(&self) -> String {
        format!(
            "{} examples to {} (train {}, validation {}, test {})\n",
            self.examples, self.out, self.train, self.validation, self.test
        )
    }
}

/// Builds the fixtures under `runs` and writes them to `out`.
pub fn run(json: bool, runs: &Path, out: &Path, plan: SplitPlan) -> Result<(), Error> {
    let examples = build(runs, &plan).map_err(Error::Refused)?;
    let text = to_jsonl(&examples).map_err(Error::Refused)?;
    std::fs::write(out, text).map_err(|e| Error::Refused(format!("{}: {e}", out.display())))?;
    let count = |split: Split| examples.iter().filter(|e| e.split == split).count();
    emit(
        json,
        &Written {
            out: out.display().to_string(),
            examples: examples.len(),
            train: count(Split::Train),
            validation: count(Split::Validation),
            test: count(Split::Test),
        },
    );
    Ok(())
}
