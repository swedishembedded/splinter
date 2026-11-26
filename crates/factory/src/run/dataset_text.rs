// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continued pretraining on an organisation's
// own documents, for its clients. If your team needs expertise in adapting
// models to a domain's writing, you can procure our services by sending an
// email to info@swedishembedded.com.

//! `dataset.text`: documents as a continued-pretraining dataset.
//!
//! Every file under the stage's input (a recipe input naming a directory;
//! hidden entries left out) is captured into a source store of the stage's
//! own, and splinter-data's CPT view ([`splinter_data::Cpt`]) makes a record
//! of each one's text: a text two documents share is trained on once
//! (`duplicate`), a file that is not text or is empty is left out and
//! counted. A record is one supervised turn of the document's text
//! (`generic-messages-v2`), so the dataset trains like any chat dataset.
//! Validation holds out whole documents, drawn by the stage's seed.
//!
//! The product is the dataset product of [`super::dataset`], its objective
//! `cpt` and its projection the view's name.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::json;
use splinter_core::clock::SystemClock;
use splinter_core::digest::Digest;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::{DatasetProduct, WaterfallStep, DATASET_SCHEMA};
use splinter_core::source::{CapturedSource, Origin, PartContent};
use splinter_data::{Corpus, Cpt, Objective, View};
use splinter_store::factory::products::Staging;
use splinter_store::sources::SourceStore;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

use super::dataset::{file_ref, held_out, Accepted, Excluded, DATASET_FILE};
use super::{io, stage_error, write_json, write_lines, Factory, RunError};
use crate::ops::TextDatasetParams;
use crate::plan::PlannedStage;
use crate::recipe::ModelRef;

/// A `dataset.text` stage resolved.
struct Resolved {
    params: TextDatasetParams,
    target: ModelRef,
    /// The input directory, as the recipe names it.
    documents: String,
}

fn resolved(factory: &Factory<'_>, stage: &PlannedStage) -> Result<Resolved, RunError> {
    let params: TextDatasetParams = stage.params().map_err(|e| stage_error(&stage.key, e))?;
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
    let documents = stage
        .input("documents")
        .and_then(|input| factory.resolved.compiled.inputs.get(input))
        .map(|input| input.path.clone())
        .ok_or_else(|| stage_error(&stage.key, "reads no documents"))?;
    Ok(Resolved {
        params,
        target,
        documents,
    })
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let text = resolved(factory, stage)?;
    let digest = factory
        .resolved
        .lock
        .files
        .get(&text.documents)
        .ok_or_else(|| stage_error(&stage.key, format!("{} was not resolved", text.documents)))?;
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &stage.key,
        json!({
            "documents": [text.documents, digest], "target": text.target,
            "validation": text.params.validation,
        }),
        Vec::new(),
        "",
        Some(text.params.seed),
    ))
}

/// Every file under `dir`, by its path relative to `root`, in order;
/// hidden entries left out.
fn files(root: &Path, dir: &Path, out: &mut Vec<(String, Vec<u8>)>) -> Result<(), String> {
    let unreadable = |e: std::io::Error| format!("{}: {e}", dir.display());
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(unreadable)? {
        let entry = entry.map_err(unreadable)?;
        if !entry.file_name().to_string_lossy().starts_with('.') {
            entries.push(entry.path());
        }
    }
    entries.sort();
    for path in entries {
        if path.is_dir() {
            files(root, &path, out)?;
        } else {
            let relative = path
                .strip_prefix(root)
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .into_owned();
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            out.push((relative, bytes));
        }
    }
    Ok(())
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
) -> Result<(), RunError> {
    let fail = |why: String| stage_error(&stage.key, why);
    let text = resolved(factory, stage)?;
    let root = factory.resolved.repository.join(&text.documents);
    let mut found = Vec::new();
    files(&root, &root, &mut found).map_err(fail)?;
    let state = tempfile::tempdir().map_err(io(staging.path()))?;
    let store = SourceStore::new(&Workspace::at(&StateRoot::new(state.path().to_path_buf())));
    let mut corpus = Corpus::new();
    // Each part's content, by the document it is.
    let mut documents: BTreeMap<Digest, String> = BTreeMap::new();
    for (path, bytes) in &found {
        let media = if path.ends_with(".md") {
            "text/markdown"
        } else {
            "text/plain"
        };
        let name = path.rsplit('/').next().unwrap_or(path).to_owned();
        let captured = CapturedSource::new(
            Origin::Document { path: path.clone() },
            vec![PartContent {
                name,
                media_type: media.into(),
                bytes: bytes.clone(),
            }],
            &SystemClock,
        )
        .map_err(|e| fail(format!("{path}: {e}")))?;
        let id = store
            .put_source(&captured)
            .map_err(|e| fail(e.to_string()))?;
        for part in store
            .get_source(&id)
            .map_err(|e| fail(e.to_string()))?
            .parts
        {
            documents
                .entry(part.content)
                .or_insert_with(|| path.clone());
        }
        corpus.add_source(id);
    }
    let view = Cpt::new(&store);
    let projection = view.project(&corpus).map_err(|e| fail(e.to_string()))?;
    if projection.records.is_empty() {
        return Err(fail(format!(
            "no text among {} files of {}; left out {:?}",
            found.len(),
            text.documents,
            projection.excluded
        )));
    }
    let document_of = |record: &splinter_data::Record| -> String {
        record
            .metadata
            .sources
            .first()
            .and_then(|digest| documents.get(digest))
            .cloned()
            .unwrap_or_default()
    };
    let groups: BTreeSet<String> = projection.records.iter().map(document_of).collect();
    let held = text
        .params
        .validation
        .as_ref()
        .map(|v| held_out(&groups, v.percent, text.params.seed))
        .unwrap_or_default();
    let (mut train, mut validation) = (Vec::new(), Vec::new());
    let (mut accepted, mut supervision) = (Vec::new(), Vec::new());
    for record in &projection.records {
        let document = document_of(record);
        let split = if held.contains(&document) {
            "validation"
        } else {
            "train"
        };
        let line =
            splinter_data::dataset_line(Objective::Cpt, record).map_err(|e| fail(e.to_string()))?;
        let lines = if split == "train" {
            &mut train
        } else {
            &mut validation
        };
        supervision.push(json!({ "split": split, "line": lines.len(), "document": document }));
        lines.push(line);
        accepted.push(Accepted {
            pass: stage.key.clone(),
            slot: String::new(),
            task: document,
            agent: String::new(),
            model: None,
            actions: 1,
            split: split.into(),
            guided: false,
        });
    }
    let write_text = |name: &str, lines: &[String]| -> Result<(), RunError> {
        let path = staging.path().join(name);
        let mut text = lines.join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        std::fs::write(&path, text).map_err(io(&path))
    };
    write_text("train.jsonl", &train)?;
    write_text("validation.jsonl", &validation)?;
    write_lines(&staging.path().join("accepted.jsonl"), &accepted)?;
    write_lines(
        &staging.path().join("excluded.jsonl"),
        &Vec::<Excluded>::new(),
    )?;
    write_lines(&staging.path().join("supervision.jsonl"), &supervision)?;
    let exclusions: BTreeMap<String, u64> = projection
        .excluded
        .iter()
        .map(|(reason, count)| {
            let name = serde_json::to_value(reason)
                .ok()
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_else(|| format!("{reason:?}"));
            (name, *count as u64)
        })
        .collect();
    let waterfall = vec![
        WaterfallStep {
            step: "files".into(),
            remaining: found.len() as u64,
        },
        WaterfallStep {
            step: "records".into(),
            remaining: projection.records.len() as u64,
        },
    ];
    let product = DatasetProduct {
        schema: DATASET_SCHEMA.into(),
        name: stage.key.clone(),
        objective: "cpt".into(),
        projection: view.name().to_owned(),
        admission: "supplied".into(),
        target: text.target.to_string(),
        sources: Vec::new(),
        train: file_ref(staging, "train.jsonl")?,
        validation: file_ref(staging, "validation.jsonl")?,
        accepted: file_ref(staging, "accepted.jsonl")?,
        excluded: file_ref(staging, "excluded.jsonl")?,
        supervision: file_ref(staging, "supervision.jsonl")?,
        waterfall,
        exclusions: exclusions.clone(),
        thinking: false,
        control: None,
        composition: None,
    };
    write_json(&staging.path().join(DATASET_FILE), &product)?;
    let mut report = format!(
        "# {}\n\n{} texts of {} files of {} ({} train, {} validation documents).\n\n",
        stage.key,
        projection.records.len(),
        found.len(),
        text.documents,
        train.len(),
        validation.len()
    );
    report.push_str("| Left out | Count |\n|---|---|\n");
    for (reason, count) in &exclusions {
        report.push_str(&format!("| {reason} | {count} |\n"));
    }
    std::fs::write(staging.path().join("report.md"), report).map_err(io(staging.path()))
}
