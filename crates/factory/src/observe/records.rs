// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning loops whose every generation can be
// explained from its own records, for its clients. If your team needs
// expertise in observing and steering agent training loops, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The records of one run directory, read and never written: the recipe it
//! ran (`recipe.yaml`, the graph that relates its stages), each stage's
//! committed product (its view's manifest, or any product by the digest of
//! its manifest, the way an evaluation names its arms), and when each stage
//! ran (`state/events.jsonl`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde_json::Value;
use splinter_core::factory::product::TrialResult;
use splinter_store::factory::products::ProductManifest;

use crate::graph::{self, Binding, RecipeV2, StageSpec};

use super::ObserveError;

/// The recipe a run directory keeps beside its products.
const RECIPE_FILE: &str = "recipe.yaml";
/// The manifest every view directory holds.
const MANIFEST_FILE: &str = "manifest.json";
/// Where committed product manifests are, by stage identity.
const PRODUCTS_DIR: &str = "products";
/// Where a product's file contents are, by digest.
const BLOBS_DIR: &str = "blobs";
/// The run's event log.
const EVENTS_FILE: &str = "state/events.jsonl";

/// A reference a stage's input port is bound to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Reference {
    /// An output of another stage, by its id.
    Stage(String),
    /// A recipe input, by its name.
    Input(String),
    /// Anything else (`$item`).
    Other(String),
}

impl Reference {
    fn parse(text: &str) -> Self {
        if let Some(rest) = text.strip_prefix("$stages.") {
            let id = rest.rsplit_once('.').map_or(rest, |(id, _)| id);
            return Self::Stage(id.to_owned());
        }
        if let Some(rest) = text.strip_prefix("$inputs.") {
            return Self::Input(rest.to_owned());
        }
        Self::Other(text.to_owned())
    }
}

/// A committed product's files: each path to the file that holds its bytes.
pub(super) struct Product {
    pub(super) manifest: ProductManifest,
    files: BTreeMap<String, PathBuf>,
}

impl Product {
    fn new(root: &Path, manifest: ProductManifest) -> Self {
        let files = manifest
            .files
            .iter()
            .map(|f| {
                let digest = f.digest.as_str();
                let (algorithm, hex) = digest.split_once(':').unwrap_or(("sha256", digest));
                (
                    f.path.clone(),
                    root.join(BLOBS_DIR).join(algorithm).join(hex),
                )
            })
            .collect();
        Self { manifest, files }
    }

    /// Where the product's file `path` is, if it has one.
    pub(super) fn file(&self, path: &str) -> Option<&Path> {
        self.files.get(path).map(PathBuf::as_path)
    }

    /// The product's file `path` parsed as JSON, if it has one that parses.
    pub(super) fn json<T: DeserializeOwned>(&self, path: &str) -> Option<T> {
        let bytes = std::fs::read(self.file(path)?).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Every line of the product's file `path` that parses, in order.
    pub(super) fn lines<T: DeserializeOwned>(&self, path: &str) -> Vec<T> {
        let Some(text) = self
            .file(path)
            .and_then(|f| std::fs::read_to_string(f).ok())
        else {
            return Vec::new();
        };
        text.lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }

    /// The final trial of every slot of an attempt product.
    pub(super) fn trials(&self) -> Vec<TrialResult> {
        self.lines(crate::run::rollout::TRIALS_FILE)
    }
}

/// One run directory's records.
pub(super) struct RunRecords {
    root: PathBuf,
    recipe: RecipeV2,
    /// Every committed product's manifest by the digest of the manifest,
    /// read on first need.
    by_manifest: std::sync::OnceLock<BTreeMap<String, ProductManifest>>,
}

impl RunRecords {
    /// The records of the run directory `root`.
    pub(super) fn open(root: &Path) -> Result<Self, ObserveError> {
        let path = root.join(RECIPE_FILE);
        let text = std::fs::read_to_string(&path).map_err(|e| ObserveError::Read {
            path: path.clone(),
            reason: e.to_string(),
        })?;
        let recipe = graph::parse(&text).map_err(|e| ObserveError::Read {
            path,
            reason: e.0.join("; "),
        })?;
        Ok(Self {
            root: root.to_path_buf(),
            recipe,
            by_manifest: std::sync::OnceLock::new(),
        })
    }

    /// The recipe's stages, in its order.
    pub(super) fn stages(&self) -> &[StageSpec] {
        &self.recipe.stages
    }

    /// The stage `id` of the recipe.
    pub(super) fn stage(&self, id: &str) -> Option<&StageSpec> {
        self.recipe.stages.iter().find(|s| s.id == id)
    }

    /// The references bound to the port `port` of the stage `id`, in order.
    pub(super) fn references(&self, id: &str, port: &str) -> Vec<Reference> {
        let Some(binding) = self.stage(id).and_then(|s| s.inputs.get(port)) else {
            return Vec::new();
        };
        match binding {
            Binding::One(r) => vec![Reference::parse(r)],
            Binding::Many(rs) => rs.iter().map(|r| Reference::parse(r)).collect(),
        }
    }

    /// The stages bound to the port `port` of the stage `id`, in order.
    pub(super) fn upstream(&self, id: &str, port: &str) -> Vec<String> {
        self.references(id, port)
            .into_iter()
            .filter_map(|r| match r {
                Reference::Stage(s) => Some(s),
                _ => None,
            })
            .collect()
    }

    /// The stage `id` and every stage that reads it, directly or through
    /// others.
    pub(super) fn downstream(&self, id: &str) -> std::collections::BTreeSet<String> {
        let mut out = std::collections::BTreeSet::from([id.to_owned()]);
        // A stage reads only stages declared above it, so one pass in order
        // closes the set.
        for stage in &self.recipe.stages {
            let reads = stage.inputs.keys().any(|port| {
                self.upstream(&stage.id, port)
                    .iter()
                    .any(|u| out.contains(u))
            });
            if reads {
                out.insert(stage.id.clone());
            }
        }
        out
    }

    /// The parameter `name` the recipe gives the stage `id`.
    pub(super) fn param(&self, id: &str, name: &str) -> Option<&Value> {
        self.stage(id).and_then(|s| s.params.get(name))
    }

    /// The product the run committed for the stage `id`: its view's manifest.
    pub(super) fn product(&self, id: &str) -> Option<Product> {
        let bytes = std::fs::read(self.root.join(id).join(MANIFEST_FILE)).ok()?;
        let manifest: ProductManifest = serde_json::from_slice(&bytes).ok()?;
        Some(Product::new(&self.root, manifest))
    }

    /// The committed product whose manifest has the digest `digest`.
    pub(super) fn product_by_manifest(&self, digest: &str) -> Option<Product> {
        let all = self.by_manifest.get_or_init(|| self.read_manifests());
        all.get(digest)
            .cloned()
            .map(|m| Product::new(&self.root, m))
    }

    fn read_manifests(&self) -> BTreeMap<String, ProductManifest> {
        let Ok(entries) = std::fs::read_dir(self.root.join(PRODUCTS_DIR)) else {
            return BTreeMap::new();
        };
        entries
            .filter_map(Result::ok)
            .filter_map(|e| std::fs::read(e.path()).ok())
            .filter_map(|b| serde_json::from_slice::<ProductManifest>(&b).ok())
            .map(|m| (m.digest().to_string(), m))
            .collect()
    }

    /// When each stage identity began and finished running, in Unix seconds,
    /// by the run's event log: the last `running` and the `completed` after it.
    pub(super) fn stage_windows(&self) -> BTreeMap<String, (u64, u64)> {
        let mut out = BTreeMap::new();
        let Ok(text) = std::fs::read_to_string(self.root.join(EVENTS_FILE)) else {
            return out;
        };
        let mut began: BTreeMap<String, u64> = BTreeMap::new();
        for event in text
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        {
            if event["kind"] != "stage.transition" {
                continue;
            }
            let (Some(subject), Some(at)) = (event["subject"].as_str(), event["at"].as_u64())
            else {
                continue;
            };
            match event["detail"]["to"].as_str() {
                Some("running") => {
                    began.insert(subject.to_owned(), at);
                }
                Some("completed") => {
                    if let Some(start) = began.get(subject) {
                        out.insert(subject.to_owned(), (*start, at));
                    }
                }
                _ => {}
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reference_names_its_stage_input_or_neither() {
        assert_eq!(
            Reference::parse("$stages.sft-g2.dataset"),
            Reference::Stage("sft-g2".into())
        );
        assert_eq!(
            Reference::parse("$inputs.course"),
            Reference::Input("course".into())
        );
        assert_eq!(Reference::parse("$item"), Reference::Other("$item".into()));
    }
}
