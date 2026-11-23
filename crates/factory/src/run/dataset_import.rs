// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the reuse of published, verified training
// records in reproducible learning runs, for its clients. If your team needs
// expertise in carrying curated datasets into training pipelines, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `dataset.import`: training records made elsewhere, as a dataset product.
//!
//! The stage's input is a recipe input naming a directory that holds
//! `train.jsonl` and, optionally, `validation.jsonl`: chat records in the
//! form `dataset.project` writes (`messages`, `tools`, `metadata`, a
//! supervised message carrying `train: true`). Nothing is projected and no
//! record is rewritten: the lines are carried over byte for byte, so a
//! published dataset trains exactly as published. A record that is not a
//! supervised chat is refused with its line and the stage fails, never
//! dropped; a record the target's trainer counts past `max_block` is
//! excluded and listed, as `dataset.project` does. The files are
//! the stage's responsibility to vouch for: a task of both files, or
//! reasoning in a dataset for no-think mode, is refused, but whether a trial
//! was verified, or a task is examination material, is not something the
//! records can say.
//!
//! The product is the dataset product of [`super::dataset`]
//! (`train.jsonl`, `validation.jsonl`, `accepted.jsonl`, `excluded.jsonl`,
//! `supervision.jsonl`, `dataset.json`, `report.md`), its projection
//! `imported`.

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::json;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::{DatasetProduct, WaterfallStep, DATASET_SCHEMA};
use splinter_data::load::{read_records, ChatRecord, Record};
use splinter_store::factory::products::Staging;

use super::dataset::{file_ref, Accepted, Excluded, DATASET_FILE};
use super::{io, stage_error, write_json, write_lines, Factory, RunError};
use crate::ops::ImportDatasetParams;
use crate::plan::PlannedStage;
use crate::recipe::ModelRef;

/// The projection an imported dataset records.
pub const PROJECTION: &str = "imported";
/// The admission an imported dataset records: what its records were
/// admitted by is not known here.
pub const ADMISSION: &str = "supplied";
/// The training records, in the input directory and the product.
const TRAIN_FILE: &str = "train.jsonl";
/// The held-out records, in the input directory and the product.
const VALIDATION_FILE: &str = "validation.jsonl";

/// An `dataset.import` stage resolved.
struct Resolved {
    params: ImportDatasetParams,
    target: ModelRef,
    /// The input directory, as the recipe names it.
    records: String,
}

fn resolved(factory: &Factory<'_>, stage: &PlannedStage) -> Result<Resolved, RunError> {
    let params: ImportDatasetParams = stage.params().map_err(|e| stage_error(&stage.key, e))?;
    let target = factory
        .resolved
        .model(&params.target)
        .cloned()
        .ok_or_else(|| {
            stage_error(
                &stage.key,
                format!("model role {} is not bound", params.target),
            )
        })?;
    let records = stage
        .input("records")
        .and_then(|input| factory.resolved.compiled.inputs.get(input))
        .map(|input| input.path.clone())
        .ok_or_else(|| stage_error(&stage.key, "imports no records"))?;
    Ok(Resolved {
        params,
        target,
        records,
    })
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let import = resolved(factory, stage)?;
    if import.params.max_block.is_some() && factory.services.trainer.is_none() {
        return Err(stage_error(
            &stage.key,
            "max_block needs the trainer to count the target's tokens",
        ));
    }
    let digest = factory
        .resolved
        .lock
        .files
        .get(&import.records)
        .ok_or_else(|| stage_error(&stage.key, format!("{} was not resolved", import.records)))?;
    let trainer = factory
        .services
        .trainer
        .filter(|_| import.params.max_block.is_some())
        .map(|t| t.identity());
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &stage.key,
        json!({
            "records": [import.records, digest], "target": import.target,
            "max_block": import.params.max_block, "thinking": import.params.thinking,
        }),
        Vec::new(),
        trainer.as_deref().unwrap_or_default(),
        None,
    ))
}

/// One record line, checked.
struct Line {
    text: String,
    group: String,
    /// The messages the loss covers, by index.
    supervised: Vec<usize>,
}

/// The lines of `path`, each a supervised chat record, or why not.
fn read_lines(path: &Path, thinking: bool) -> Result<Vec<Line>, String> {
    let mut lines = Vec::new();
    for (number, record) in read_records::<ChatRecord>(path).map_err(|e| e.to_string())? {
        let at = |why: &str| format!("{} line {number}: {why}", path.display());
        let supervised = record.supervised();
        if supervised.is_empty() {
            return Err(at("supervises no message (`train: true`)"));
        }
        if !thinking && record.has_reasoning() {
            return Err(at(
                "keeps a turn's reasoning in a dataset for no-think mode (`thinking: false`)",
            ));
        }
        lines.push(Line {
            text: record.line().to_owned(),
            group: record
                .group()
                .map_or_else(|| format!("line {number}"), str::to_owned),
            supervised,
        });
    }
    Ok(lines)
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
) -> Result<(), RunError> {
    let import = resolved(factory, stage)?;
    let directory = factory.resolved.repository.join(&import.records);
    let read = |name: &str, required: bool| -> Result<Vec<Line>, RunError> {
        let path = directory.join(name);
        if !required && !path.exists() {
            return Ok(Vec::new());
        }
        read_lines(&path, import.params.thinking).map_err(|e| stage_error(&stage.key, e))
    };
    let mut splits = vec![
        ("train", read(TRAIN_FILE, true)?),
        ("validation", read(VALIDATION_FILE, false)?),
    ];
    let raw: u64 = splits.iter().map(|(_, lines)| lines.len() as u64).sum();
    if splits[0].1.is_empty() {
        return Err(stage_error(
            &stage.key,
            format!("{} holds no training record", import.records),
        ));
    }
    let shared: Vec<&String> = {
        let train: BTreeSet<&String> = splits[0].1.iter().map(|l| &l.group).collect();
        let held: BTreeSet<&String> = splits[1].1.iter().map(|l| &l.group).collect();
        train.intersection(&held).copied().collect()
    };
    if !shared.is_empty() {
        return Err(stage_error(
            &stage.key,
            format!("groups in both splits: {shared:?}"),
        ));
    }
    let mut excluded: Vec<Excluded> = Vec::new();
    if let (Some(limit), Some(trainer)) = (import.params.max_block, factory.services.trainer) {
        for (split, lines) in &mut splits {
            let texts: Vec<String> = lines.iter().map(|l| l.text.clone()).collect();
            let lengths = trainer
                .measure(&import.target, &texts)
                .map_err(|e| stage_error(&stage.key, e))?;
            let mut kept = Vec::new();
            for (line, tokens) in std::mem::take(lines).into_iter().zip(lengths) {
                if tokens > limit {
                    excluded.push(Excluded {
                        pass: (*split).to_owned(),
                        slot: String::new(),
                        task: line.group.clone(),
                        reason: "longer_than_max_block".into(),
                        detail: format!(
                            "{tokens} tokens of {} where max_block is {limit}",
                            import.target
                        ),
                    });
                } else {
                    kept.push(line);
                }
            }
            *lines = kept;
        }
    }
    let (mut accepted, mut supervision) = (Vec::new(), Vec::new());
    for (split, lines) in &splits {
        for (index, line) in lines.iter().enumerate() {
            accepted.push(Accepted {
                pass: import.records.clone(),
                slot: format!("{split}/{index}"),
                task: line.group.clone(),
                agent: "imported".into(),
                model: None,
                actions: line.supervised.len(),
                split: (*split).to_owned(),
                guided: false,
            });
            supervision.push(json!({
                "split": split, "line": index, "pass": import.records, "task": line.group,
                "supervised_messages": line.supervised,
            }));
        }
    }
    if splits[0].1.is_empty() {
        return Err(stage_error(
            &stage.key,
            format!("every training record is longer than max_block ({excluded:?})"),
        ));
    }
    for (split, name) in [("train", TRAIN_FILE), ("validation", VALIDATION_FILE)] {
        let lines = splits
            .iter()
            .find(|(s, _)| *s == split)
            .map_or(&[][..], |(_, lines)| lines);
        let mut text = lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        let path = staging.path().join(name);
        std::fs::write(&path, text).map_err(io(&path))?;
    }
    write_lines(&staging.path().join("accepted.jsonl"), &accepted)?;
    write_lines(&staging.path().join("excluded.jsonl"), &excluded)?;
    write_lines(&staging.path().join("supervision.jsonl"), &supervision)?;
    let digest = factory
        .resolved
        .lock
        .files
        .get(&import.records)
        .cloned()
        .ok_or_else(|| stage_error(&stage.key, format!("{} was not resolved", import.records)))?;
    let mut exclusions = std::collections::BTreeMap::new();
    for e in &excluded {
        *exclusions.entry(e.reason.clone()).or_insert(0u64) += 1;
    }
    let waterfall = vec![
        WaterfallStep {
            step: "supplied_records".into(),
            remaining: raw,
        },
        WaterfallStep {
            step: "accepted".into(),
            remaining: accepted.len() as u64,
        },
    ];
    let product = DatasetProduct {
        schema: DATASET_SCHEMA.into(),
        name: stage.key.clone(),
        objective: "sft".into(),
        projection: PROJECTION.into(),
        admission: ADMISSION.into(),
        target: import.target.to_string(),
        sources: vec![digest],
        train: file_ref(staging, TRAIN_FILE)?,
        validation: file_ref(staging, VALIDATION_FILE)?,
        accepted: file_ref(staging, "accepted.jsonl")?,
        excluded: file_ref(staging, "excluded.jsonl")?,
        supervision: file_ref(staging, "supervision.jsonl")?,
        waterfall,
        exclusions,
        thinking: import.params.thinking,
    };
    write_json(&staging.path().join(DATASET_FILE), &product)?;
    let held = splits[1].1.len();
    let report = format!(
        "# Dataset {}\n\nImported from {}: {} records supplied, {} train and {held} validation \
         accepted, {} excluded. Mode: {}.\n",
        stage.key,
        import.records,
        raw,
        splits[0].1.len(),
        excluded.len(),
        if import.params.thinking {
            "thinking"
        } else {
            "no-think"
        },
    );
    std::fs::write(staging.path().join("report.md"), report).map_err(io(staging.path()))
}
