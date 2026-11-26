// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment runs that can be forked, checked
// and reported on long after they ran, for its clients. If your team needs
// expertise in reproducible machine-learning experiments, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What can be done with a run directory besides running it: fork it,
//! verify its products, read its status and write its report.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use splinter_core::clock::SystemClock;
use splinter_core::digest::Digest;
use splinter_store::factory::products::{ProductManifest, ProductStore};
use splinter_store::factory::runstate::RunState;

use super::{io, require_run, RunError, RunManifest};

/// What [`fork`] brought over.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Forked {
    /// Products now committed in the new run.
    pub products: usize,
    /// Blobs shared with the source run as hard links (no bytes moved).
    pub blobs_linked: usize,
    /// Blobs copied because linking was not possible (another file system):
    /// non-zero means the fork moved bytes.
    pub blobs_copied: usize,
}

/// Reuse every product committed in the run `source` in the run `run`
/// (normally a new directory), so the stages a changed recipe leaves
/// unchanged are not computed again. Products are immutable, so their
/// blobs are linked, not copied, wherever the file system allows.
///
/// # Errors
/// A product cannot be imported.
pub fn fork(source: &Path, run: &Path) -> Result<Forked, RunError> {
    require_run(source)?;
    std::fs::create_dir_all(run).map_err(io(run))?;
    let from = ProductStore::new(source, Box::new(SystemClock));
    let into = ProductStore::new(run, Box::new(SystemClock));
    let mut forked = Forked::default();
    for manifest in from.list()? {
        let imported = into.import(&from, &manifest.stage, &run.join(&manifest.name))?;
        forked.products += 1;
        forked.blobs_linked += imported.blobs_linked;
        forked.blobs_copied += imported.blobs_copied;
    }
    Ok(forked)
}

/// Check every committed product of the run in `run` against its blobs and
/// its view; one line per problem, empty when the run is intact.
///
/// # Errors
/// The products cannot be listed.
pub fn verify(run: &Path) -> Result<Vec<String>, RunError> {
    require_run(run)?;
    let store = ProductStore::new(run, Box::new(SystemClock));
    let mut problems = Vec::new();
    for manifest in store.list()? {
        // Every product's files must match their blobs; a view is checked
        // against the product it shows (a stage key a fork reused holds
        // several products, and its view shows one of them).
        for file in &manifest.files {
            if let Err(e) = store.verify(&file.digest) {
                problems.push(format!("{} {}: {e}", manifest.name, file.path));
            }
        }
        let view = run.join(&manifest.name);
        let shown: Option<ProductManifest> = std::fs::read(view.join("manifest.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok());
        if shown.as_ref().is_some_and(|m| m.stage == manifest.stage) {
            if let Err(e) = store.verify_product(&manifest.stage, &view) {
                problems.push(format!("{}: {e}", manifest.name));
            }
        }
    }
    Ok(problems)
}

/// One stage's state as the run state records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StageStatus {
    /// The stage key.
    pub key: String,
    /// Its state.
    pub state: String,
    /// Its product, when committed.
    pub product: Option<Digest>,
    /// Why it failed, when it did.
    pub error: Option<String>,
    /// Its rollout slots by state, for a rollout or eval arm.
    pub slots: BTreeMap<String, usize>,
}

/// What the run in `run` has done, stage by stage, from its run state alone.
///
/// # Errors
/// The run state cannot be read.
pub fn status(run: &Path) -> Result<Vec<StageStatus>, RunError> {
    require_run(run)?;
    let state = RunState::open(run)?;
    let mut out = Vec::new();
    for record in state.list()? {
        let mut slots = BTreeMap::new();
        for pass in state.slot_namespaces(&record.stage_id)? {
            for slot in state.list_slots(&pass)? {
                *slots.entry(slot.state.to_string()).or_default() += 1;
            }
        }
        out.push(StageStatus {
            key: record.name.clone(),
            state: record.state.to_string(),
            product: record.product.clone(),
            error: record.error.clone(),
            slots,
        });
    }
    Ok(out)
}

/// The run's report: every stage with its product, then each stage's own
/// report, in plan order.
#[must_use]
pub fn report_of(run: &Path, manifest: &RunManifest) -> String {
    let mut out = format!(
        "# Run of {}\n\n| stage | product | this invocation |\n|---|---|---|\n",
        manifest.recipe
    );
    for stage in &manifest.stages {
        let short = &stage.product.hex()[..12];
        let how = if stage.computed { "computed" } else { "reused" };
        out.push_str(&format!("| {} | {short} | {how} |\n", stage.key));
    }
    for stage in &manifest.stages {
        if let Ok(text) = std::fs::read_to_string(run.join(&stage.key).join("report.md")) {
            // A stage's headings move one level down under the run's.
            let demoted: Vec<String> = text
                .lines()
                .map(|l| {
                    if l.starts_with('#') {
                        format!("#{l}")
                    } else {
                        l.to_owned()
                    }
                })
                .collect();
            out.push('\n');
            out.push_str(demoted.join("\n").trim_end());
            out.push('\n');
        }
    }
    out
}
