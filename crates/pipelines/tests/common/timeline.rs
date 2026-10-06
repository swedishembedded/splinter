// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures the timeline specs share: a deterministic synthetic cohort of
//! known hazards written as a record file (households of two, so groups
//! matter), a context over a scratch state root, and the import and split
//! stages run on it.

use std::path::{Path, PathBuf};

use splinter_core::terms::{Terms, UsagePolicy};
use splinter_data::timeline_dataset::ProjectionSpec;
use splinter_model::timeline::synthetic;
use splinter_orchestrator::Context;
use splinter_pipelines::timeline::data::{
    import_records, split_timeline, ImportRequest, SplitPlan, SplitReport, SplitRequest,
};

use super::{config, Scratch};

/// The outcome codes of the synthetic population.
pub const CODES: [&str; 3] = ["death:a", "death:b", "onset"];
/// The absorbing ones.
pub const ABSORBING: [&str; 2] = ["death:a", "death:b"];

/// A context over a scratch state root, with no model.
pub fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false).unwrap();
    (scratch, ctx)
}

/// `n` synthetic subjects as a record file named `name` in `dir`: the
/// generator's own history and outcomes, a household of two per group, and
/// no interventions. Deterministic in `seed`.
pub fn write_synthetic(dir: &Path, name: &str, n: usize, seed: u64) -> PathBuf {
    let (subjects, _) = synthetic::population(n, seed);
    let lines: Vec<String> = subjects
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let mut line = serde_json::to_value(s).unwrap();
            line["group_id"] = format!("household-{}", i / 2).into();
            line["interventions"] = serde_json::json!([]);
            line.to_string()
        })
        .collect();
    let path = dir.join(name);
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    path
}

/// Imports `file` as `dataset` under `policy` and splits it by participant
/// group with a 30 percent test.
pub fn prepare(
    ctx: &Context,
    file: &Path,
    dataset: &str,
    policy: UsagePolicy,
    seed: u64,
) -> SplitReport {
    import_records(
        ctx,
        &ImportRequest {
            file: file.to_path_buf(),
            dataset: dataset.into(),
            terms: policy.terms(dataset),
            secret: b"test campaign secret".to_vec(),
        },
    )
    .unwrap();
    split_timeline(
        ctx,
        &SplitRequest {
            plan: SplitPlan::Participants {
                locked_share: 0.3,
                folds: 5,
            },
            seed,
            projection: ProjectionSpec::at_entry(),
        },
    )
    .unwrap()
}

/// The terms a policy label states, named `name`.
pub fn terms(policy: UsagePolicy, name: &str) -> Terms {
    policy.terms(name)
}
