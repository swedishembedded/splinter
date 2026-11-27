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
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::clock::SystemClock;
use splinter_core::digest::Digest;
use splinter_store::factory::products::{ProductManifest, ProductStore};
use splinter_store::factory::runstate::RunState;

use super::{io, read_json, require_run, write_json, RunError, RunManifest};

/// The file of a forked run that names, for every product a fork brought
/// in, the run it came from.
pub const FORK_FILE: &str = "fork.json";

/// What [`fork_from`] brought over.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Forked {
    /// Products now committed in the new run.
    pub products: usize,
    /// Products brought in from each source run, in the order given.
    pub per_source: Vec<usize>,
    /// Blobs shared with the source runs as hard links (no bytes moved).
    pub blobs_linked: usize,
    /// Blobs copied because linking was not possible (another file system):
    /// non-zero means the fork moved bytes.
    pub blobs_copied: usize,
    /// Stages a later source holds another product of than the one taken
    /// (the same stage computed again in that run): `(stage key, source)`.
    /// A product's digest is an input of every stage downstream of it, so
    /// that source's downstream products are not reused.
    pub conflicts: Vec<(String, PathBuf)>,
}

/// Where a forked run's products came from ([`FORK_FILE`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkOrigin {
    /// Every product a fork brought in, by stage identity.
    pub products: Vec<ForkedProduct>,
}

/// One product a fork brought in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkedProduct {
    /// The stage identity it is the product of.
    pub stage: Digest,
    /// The stage key it was viewed at in its source run.
    pub name: String,
    /// The run it came from (absolute).
    pub source: PathBuf,
}

/// Reuse every product committed in the run `source` in the run `run`;
/// [`fork_from`] with one source.
///
/// # Errors
/// As [`fork_from`].
pub fn fork(source: &Path, run: &Path) -> Result<Forked, RunError> {
    fork_from(&[source], run)
}

/// Reuse every product committed in any of the runs `sources` in the run
/// `run` (normally a new directory), so the stages a changed recipe leaves
/// unchanged are not computed again. A stage identity committed in several
/// sources is taken from the first that holds it, in the order given; one
/// `run` already holds is left as it is. A product's digest (which names
/// when it was committed) is an input of the stages downstream of it, so the
/// sources should share their upstream products (be forks of a common run):
/// a stage a later source computed again is reported in
/// [`Forked::conflicts`], as that source's downstream products cannot be
/// reused. Products are immutable and
/// addressed by their stage identity, so a product of one run is the
/// product of that stage in every run; their blobs are linked, not copied,
/// wherever the file system allows. [`FORK_FILE`] records which source each
/// product came from.
///
/// # Errors
/// A source is not a run, or a product cannot be imported.
pub fn fork_from(sources: &[&Path], run: &Path) -> Result<Forked, RunError> {
    for source in sources {
        require_run(source)?;
    }
    std::fs::create_dir_all(run).map_err(io(run))?;
    let into = ProductStore::new(run, Box::new(SystemClock));
    let mut held: BTreeMap<Digest, Digest> = into
        .list()?
        .into_iter()
        .map(|m| (m.stage.clone(), m.digest()))
        .collect();
    let origin_path = run.join(FORK_FILE);
    let mut origin: ForkOrigin = if origin_path.exists() {
        read_json(&origin_path)?
    } else {
        ForkOrigin::default()
    };
    let mut forked = Forked::default();
    for source in sources {
        let from = ProductStore::new(*source, Box::new(SystemClock));
        let absolute = std::fs::canonicalize(source).map_err(io(source))?;
        let mut brought = 0;
        for manifest in from.list()? {
            if let Some(taken) = held.get(&manifest.stage) {
                if *taken != manifest.digest() {
                    forked.conflicts.push((manifest.name, absolute.clone()));
                }
                continue;
            }
            held.insert(manifest.stage.clone(), manifest.digest());
            let imported = into.import(&from, &manifest.stage, &run.join(&manifest.name))?;
            brought += 1;
            forked.blobs_linked += imported.blobs_linked;
            forked.blobs_copied += imported.blobs_copied;
            origin.products.push(ForkedProduct {
                stage: manifest.stage,
                name: manifest.name,
                source: absolute.clone(),
            });
        }
        forked.products += brought;
        forked.per_source.push(brought);
    }
    write_json(&origin_path, &origin)?;
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
