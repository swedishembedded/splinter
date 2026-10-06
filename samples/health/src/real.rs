// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the same loop on real cohort records, with
// the terms of the data declared by whoever holds them and enforced into the
// release. If your team needs expertise in training risk models on regulated
// health data, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `real`: the same pipeline on a directory of `timeline-v1` files.
//!
//! Nothing of the data is in this repository. The command is pointed at a
//! directory (for instance the NHANES linked-mortality timelines the
//! `lifecourse` sample builds), imports the files that match, checks the
//! outcomes asked for against the ontology for the dataset's id and splits.
//! The terms of the data come from a declared policy file; with none, they
//! are `unknown`, which permits nothing: `train` refuses until someone who
//! holds the terms declares them, and a release is restricted unless asked
//! otherwise and allowed.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};

use crate::ontology::{Ontology, NHANES_DATASET};
use crate::source::Policy;
use crate::state::{Run, DEFAULT_SECRET};
use crate::steps::{import, split, ImportRecord, SplitArgs};

/// The files of `dir` that are timeline records: `timelines*.jsonl`, in name
/// order.
pub fn timeline_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|x| x == "jsonl")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("timelines"))
        })
        .collect();
    files.sort();
    anyhow::ensure!(
        !files.is_empty(),
        "{} holds no timelines*.jsonl file: point --timelines at a directory of timeline-v1 files",
        dir.display()
    );
    Ok(files)
}

/// What `real` is asked to do.
#[derive(Clone, Debug)]
pub struct RealArgs {
    /// The directory of timeline files.
    pub timelines: PathBuf,
    /// The policy file declaring their terms; none declares nothing.
    pub policy: Option<PathBuf>,
    /// The dataset id the files are imported as.
    pub dataset: String,
    /// The outcome codes to train on.
    pub codes: Vec<String>,
    /// How to cut.
    pub split: SplitArgs,
}

/// The codes the NHANES timelines supply.
pub fn nhanes_codes() -> Vec<String> {
    ["death:cvd", "death:cancer", "death:other"]
        .map(String::from)
        .to_vec()
}

/// Imports and splits the directory; returns what was done and what the
/// terms allow.
pub fn real(run: &Run, args: &RealArgs) -> Result<serde_json::Value> {
    let ontology = Ontology::bundled()?;
    ontology.request_all(&args.codes, &args.dataset)?;
    let policy = Policy::load(args.policy.as_deref())?;
    let terms = policy.terms_of(&args.dataset)?;
    let files = timeline_files(&args.timelines)?;
    anyhow::ensure!(
        files.len() == 1,
        "{} timeline files match ({}); one dataset is one file: point --timelines at a directory with one",
        files.len(),
        files.iter().map(|f| f.display().to_string()).collect::<Vec<_>>().join(", ")
    );
    let label = policy
        .terms
        .get(&args.dataset)
        .or_else(|| policy.terms.get("*"))
        .cloned()
        .unwrap_or_else(|| "unknown".into());
    let record = ImportRecord {
        dataset: args.dataset.clone(),
        supplies: args.codes.clone(),
        subgroups: policy.subgroups.clone(),
        usage: label.clone(),
    };
    let imported = import(run, &files[0], &record, terms.clone(), DEFAULT_SECRET)?;
    let mut cut = args.split.clone();
    cut.dataset = Some(args.dataset.clone());
    let split = split(run, &cut)?;
    let trains = terms.permits(splinter_sdk::vocabulary::terms::Use::Training);
    Ok(serde_json::json!({
        "dataset": args.dataset,
        "file": files[0].display().to_string(),
        "imported": imported,
        "split": split.report,
        "terms": {
            "usage": label,
            "training": format!("{:?}", terms.training),
            "unrestricted_release": terms.permits_unrestricted_release().map_or_else(|why| why, |()| "allowed".into()),
        },
        "next": match trains {
            Ok(()) => "train, eval and release as for the synthetic cohort; release is restricted unless --unrestricted is given and the terms allow it".to_owned(),
            Err(why) => format!("train will refuse ({why}): declare the data's terms in a policy file and import again with --policy"),
        },
        "ontology_dataset_is_nhanes": args.dataset == NHANES_DATASET,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_timeline_files_are_found_and_a_directory_without_one_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        assert!(timeline_files(dir.path()).is_err());
        std::fs::write(dir.path().join("design.jsonl"), "{}").unwrap();
        std::fs::write(dir.path().join("timelines.json"), "{}").unwrap();
        assert!(
            timeline_files(dir.path()).is_err(),
            "neither is a timelines*.jsonl file"
        );
        std::fs::write(dir.path().join("timelines.jsonl"), "{}").unwrap();
        std::fs::write(dir.path().join("timelines-2.jsonl"), "{}").unwrap();
        let found = timeline_files(dir.path()).unwrap();
        assert_eq!(
            found
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            ["timelines-2.jsonl", "timelines.jsonl"]
        );
    }

    #[test]
    fn outcomes_the_dataset_is_not_listed_for_are_refused_before_anything_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let args = RealArgs {
            timelines: dir.path().to_path_buf(),
            policy: None,
            dataset: NHANES_DATASET.into(),
            codes: vec!["death:cvd".into(), "dx:t2d".into()],
            split: SplitArgs {
                dataset: None,
                plan: splinter_sdk::timeline::data::SplitPlan::Participants {
                    locked_share: 0.3,
                    folds: 5,
                },
                seed: 3,
            },
        };
        let why = real(&Run::new(dir.path().join("run")), &args)
            .unwrap_err()
            .to_string();
        assert!(why.contains("unsupported locally"), "{why}");
    }
}
