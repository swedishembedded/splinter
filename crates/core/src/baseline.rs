// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible-experiment infrastructure for
// learning systems, for its clients. If your team needs expertise in freezing
// the exact software, models and hardware an evaluation ran on, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `baseline-v1`: the machine-readable record of everything a registered
//! experiment ran on, written before the experiment starts: the three
//! repositories' commits and whether the tree was clean, the toolchain, the
//! serving binary, the model files and tokenizer, the seeds and the hardware.
//!
//! Like [`crate::build`], every field is a measurement that may be missing, and
//! a missing one is absent, never `0` or an empty string. [`Baseline::unmet`]
//! lists what a registered run may not start without.

use serde::{Deserialize, Serialize};

use crate::build::BuildInfo;
use crate::digest::Digest;

/// The schema name every record carries.
pub const SCHEMA: &str = "baseline-v1";

/// One file and its content digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDigest {
    /// The file's name inside its model directory.
    pub name: String,
    /// Its SHA-256.
    pub digest: Digest,
}

/// A model as served: its files and its tokenizer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// The model's name in the store (for example `Qwen/Qwen3-8B`).
    pub id: String,
    /// Every weight and configuration file, by digest.
    pub files: Vec<FileDigest>,
    /// The digest of the tokenizer file, absent when the model has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<Digest>,
}

/// The serving binary: brain's `serve`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrainBinary {
    /// Where it was read from.
    pub path: String,
    /// Its SHA-256.
    pub digest: Digest,
    /// What `--version` printed, when it could be asked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// One GPU as the driver reports it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gpu {
    /// The product name.
    pub name: String,
    /// Total memory in MiB.
    pub memory_mib: u64,
}

/// The machine the experiment ran on.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hardware {
    /// The kernel release.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
    /// The CPU model name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
    /// Logical CPUs available to the process.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_threads: Option<u32>,
    /// The NVIDIA driver version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_driver: Option<String>,
    /// The GPUs, in the driver's order; empty when none were found.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gpus: Vec<Gpu>,
}

/// The seeds the caller registered for the experiment.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Seeds {
    /// The seed that reserves the qualification families.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exam: Option<u64>,
    /// The seed of adapter initialisation and batch order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<u64>,
}

/// The whole record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Baseline {
    /// Always [`SCHEMA`].
    pub schema: String,
    /// The splinter binary: commits, tree state, digest.
    pub build: BuildInfo,
    /// The compiler the splinter binary was built with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rustc: Option<String>,
    /// The serving binary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brain_binary: Option<BrainBinary>,
    /// The models named on the command line.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<ModelIdentity>,
    /// The registered seeds.
    pub seeds: Seeds,
    /// The machine.
    pub hardware: Hardware,
}

impl Baseline {
    /// A record of `build` and nothing else yet.
    #[must_use]
    pub fn new(build: BuildInfo) -> Self {
        Self {
            schema: SCHEMA.to_string(),
            build,
            rustc: None,
            brain_binary: None,
            models: Vec::new(),
            seeds: Seeds::default(),
            hardware: Hardware::default(),
        }
    }

    /// What a registered run may not start without, each as a sentence: a
    /// tree with uncommitted changes, a commit of any of the three
    /// repositories that is not known, a binary that was not digested.
    #[must_use]
    pub fn unmet(&self) -> Vec<String> {
        let mut unmet = Vec::new();
        match self.build.dirty {
            Some(false) => {}
            Some(true) => unmet.push("the splinter tree has uncommitted changes".to_string()),
            None => unmet.push("whether the splinter tree was clean is not known".to_string()),
        }
        let revisions = &self.build.revisions;
        for (name, known) in [
            ("splinter", &revisions.splinter),
            ("sven", &revisions.sven),
            ("brain", &revisions.brain),
        ] {
            if known.is_none() {
                unmet.push(format!("the {name} commit is not known"));
            }
        }
        if self.build.binary_digest.is_none() {
            unmet.push("the splinter binary could not be digested".to_string());
        }
        unmet
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::Revisions;

    fn clean() -> Baseline {
        Baseline::new(BuildInfo {
            revisions: Revisions {
                splinter: Some("a".repeat(40)),
                sven: Some("b".repeat(40)),
                brain: Some("c".repeat(40)),
            },
            dirty: Some(false),
            binary_digest: Some(Digest::sha256_of(b"binary")),
            gpu_contexts: None,
        })
    }

    #[test]
    fn a_clean_pinned_build_has_nothing_unmet() {
        assert_eq!(clean().unmet(), Vec::<String>::new());
    }

    #[test]
    fn a_dirty_tree_an_unknown_commit_or_an_undigested_binary_is_unmet() {
        let mut dirty = clean();
        dirty.build.dirty = Some(true);
        assert_eq!(dirty.unmet().len(), 1, "{:?}", dirty.unmet());
        let mut unknown = clean();
        unknown.build.dirty = None;
        unknown.build.revisions.sven = None;
        unknown.build.binary_digest = None;
        let unmet = unknown.unmet();
        assert_eq!(unmet.len(), 3, "{unmet:?}");
        assert!(unmet.iter().any(|m| m.contains("sven")), "{unmet:?}");
    }

    #[test]
    fn the_record_names_its_schema_and_omits_what_was_not_measured() {
        let json = serde_json::to_value(clean()).unwrap();
        assert_eq!(json["schema"], "baseline-v1");
        assert!(json.get("brain_binary").is_none() && json.get("models").is_none());
        assert!(json["hardware"].as_object().unwrap().is_empty());
    }
}
