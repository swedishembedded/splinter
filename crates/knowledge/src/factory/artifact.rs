// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible, content-addressed training
// environments for its clients. If your team needs expertise in packaging
// agent tasks in a portable task-folder format with verifier provenance,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The environment artifact: a Harbor task folder with the provenance a
//! training campaign needs written into `task.toml`.
//!
//! ```text
//! task.toml          Harbor's keys, and [metadata.splinter_env] below
//! instruction.md     what the solver is asked
//! environment/       the workspace the solver works in (Harbor copies it to
//!                    /app) and the Dockerfile that builds the image
//! tests/test.sh      the verifier: writes /logs/verifier/reward.json, a flat
//!                    object of numbers in 0..=1 whose `reward` is the scalar
//! solution/solve.sh  the reference: solves the task from the workspace
//! ```
//!
//! `[metadata.splinter_env]` records the schema version, the environment
//! (generator) id, the plugin and its version, the [`VerifierClass`] and
//! verifier every reward from this folder comes from, the difficulty knobs,
//! the [`Split`] and the seed. The split is `train` or `dev` and nothing
//! else: a sealed exam is never produced here.
//!
//! Hidden material (`tests/`, `solution/`) is never part of what a solver is
//! shown: [`Instance::solver_view`] is the instruction and `environment/`,
//! and a solver runtime mounts nothing else.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::digest::{digest_of, Digest};
use splinter_core::verifiability::{RewardProvenance, Split, VerifierClass};

use super::folder::{Folder, FolderError};

/// The version of the `[metadata.splinter_env]` schema this build writes and
/// reads. A newer file is refused: its fields may mean something this build
/// would ignore.
pub const SCHEMA_VERSION: u32 = 1;

/// The workspace directory (Harbor's `environment/`).
pub const ENVIRONMENT_DIR: &str = "environment";

/// The Dockerfile of the image, inside [`ENVIRONMENT_DIR`].
pub const DOCKERFILE: &str = "environment/Dockerfile";

/// The verifier script.
pub const TEST_SCRIPT: &str = "tests/test.sh";

/// The reference solution script.
pub const SOLVE_SCRIPT: &str = "solution/solve.sh";

/// The script that rebuilds the workspace's state from its sources, when the
/// environment has state beyond its files (a database): the environment's
/// `reset`.
pub const RESET_SCRIPT: &str = "environment/reset.sh";

/// Where the verifier writes its reward, in the container.
pub const REWARD_PATH: &str = "/logs/verifier/reward.json";

/// What a solver may use while working.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SolverTools {
    /// Reading and writing files, and no way to run code: the instruction
    /// carries everything and the answer is a file. A harness that gives the
    /// solver an interpreter has changed the task.
    Files,
    /// A shell in the workspace.
    Shell,
}

/// What `[metadata.splinter_env]` records.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvMeta {
    /// [`SCHEMA_VERSION`] at the time of writing.
    pub schema_version: u32,
    /// The environment (generator configuration) this instance is drawn
    /// from: [`environment_id`].
    pub environment: String,
    /// The plugin that generated it.
    pub plugin: String,
    /// The plugin's version; a change of generation or verification logic
    /// is a new version.
    pub plugin_version: String,
    /// How every reward from this folder is established.
    pub verifier_class: VerifierClass,
    /// The verifier, with its version.
    pub verifier: String,
    /// The difficulty knobs the instance was generated with.
    pub difficulty: BTreeMap<String, u32>,
    /// The part of the generator it belongs to.
    pub split: Split,
    /// The seed it was drawn with. Written as decimal text: TOML integers
    /// stop at 2^63 and seeds are drawn over all of `u64`.
    #[serde(with = "seed_text")]
    pub seed: u64,
    /// What a solver may use.
    pub solver_tools: SolverTools,
}

mod seed_text {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(seed: &u64, out: S) -> Result<S::Ok, S::Error> {
        out.serialize_str(&seed.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(input: D) -> Result<u64, D::Error> {
        let text = String::deserialize(input)?;
        text.parse()
            .map_err(|_| serde::de::Error::custom(format!("{text:?} is not a seed")))
    }
}

/// The id of an environment: what a plugin is asked for, independent of the
/// split and seed that draw instances from it.
#[must_use]
pub fn environment_id(
    plugin: &str,
    plugin_version: &str,
    difficulty: &BTreeMap<String, u32>,
) -> String {
    #[derive(Serialize)]
    struct Body<'a> {
        schema: u32,
        plugin: &'a str,
        version: &'a str,
        difficulty: &'a BTreeMap<String, u32>,
    }
    let digest = digest_of(&Body {
        schema: SCHEMA_VERSION,
        plugin,
        version: plugin_version,
        difficulty,
    });
    format!("env-{}", &digest.hex()[..16])
}

#[derive(Serialize, Deserialize)]
struct TaskToml {
    version: String,
    metadata: Metadata,
    #[serde(default)]
    verifier: Timeout,
    #[serde(default)]
    agent: Timeout,
    #[serde(default)]
    environment: EnvironmentToml,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    #[serde(default)]
    author_name: String,
    #[serde(default)]
    category: String,
    #[serde(default)]
    tags: Vec<String>,
    splinter_env: EnvMeta,
}

#[derive(Serialize, Deserialize)]
struct Timeout {
    timeout_sec: f64,
}

impl Default for Timeout {
    fn default() -> Self {
        Self { timeout_sec: 600.0 }
    }
}

#[derive(Serialize, Deserialize)]
struct EnvironmentToml {
    build_timeout_sec: f64,
}

impl Default for EnvironmentToml {
    fn default() -> Self {
        Self {
            build_timeout_sec: 600.0,
        }
    }
}

/// Why an artifact is not one.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    /// A file every artifact has is missing.
    #[error("the artifact has no {0}")]
    Missing(&'static str),
    /// `task.toml` does not parse as an artifact's.
    #[error("task.toml: {0}")]
    TaskToml(String),
    /// The metadata is from another schema version.
    #[error("task.toml records schema version {found}; this build reads version {SCHEMA_VERSION}")]
    Schema {
        /// What the file says.
        found: u32,
    },
    /// The folder could not be built or read.
    #[error(transparent)]
    Folder(#[from] FolderError),
}

/// One instance of an environment: a task folder and what it records.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    meta: EnvMeta,
    folder: Folder,
}

/// The parts a plugin hands over; [`Instance::assemble`] lays them out.
pub struct Parts {
    /// What the solver is asked, as Markdown.
    pub instruction: String,
    /// The workspace, with paths relative to its root (`/app`).
    pub workspace: Folder,
    /// The packages the image needs, for the Dockerfile.
    pub packages: Vec<String>,
    /// The verifier: paths relative to `tests/`, with `test.sh` among them.
    pub tests: Folder,
    /// The reference solution: paths relative to `solution/`, with
    /// `solve.sh` among them.
    pub solution: Folder,
    /// The category Harbor lists the task under.
    pub category: String,
}

impl Instance {
    /// Lays `parts` out as a task folder with `meta` recorded in
    /// `task.toml`.
    pub fn assemble(meta: EnvMeta, parts: Parts) -> Result<Self, ArtifactError> {
        let mut folder = Folder::new();
        folder.put_text("instruction.md", &parts.instruction)?;
        for (path, entry) in parts.workspace.entries() {
            folder.put(
                &format!("{ENVIRONMENT_DIR}/{path}"),
                entry.bytes.clone(),
                entry.executable,
            )?;
        }
        folder.put_text(DOCKERFILE, &dockerfile(&parts.packages))?;
        for (path, entry) in parts.tests.entries() {
            folder.put(
                &format!("tests/{path}"),
                entry.bytes.clone(),
                entry.executable,
            )?;
        }
        for (path, entry) in parts.solution.entries() {
            folder.put(
                &format!("solution/{path}"),
                entry.bytes.clone(),
                entry.executable,
            )?;
        }
        for required in [TEST_SCRIPT, SOLVE_SCRIPT] {
            if folder.get(required).is_none() {
                return Err(ArtifactError::Missing(required));
            }
        }
        let toml = TaskToml {
            version: "1.0".into(),
            metadata: Metadata {
                author_name: "splinter".into(),
                category: parts.category,
                tags: vec!["splinter-env".into(), meta.plugin.clone()],
                splinter_env: meta.clone(),
            },
            verifier: Timeout::default(),
            agent: Timeout {
                timeout_sec: 1800.0,
            },
            environment: EnvironmentToml::default(),
        };
        let text =
            toml::to_string_pretty(&toml).map_err(|e| ArtifactError::TaskToml(e.to_string()))?;
        folder.put_text("task.toml", &text)?;
        Ok(Self { meta, folder })
    }

    /// Reads an artifact back from a folder (as read from disk).
    pub fn from_folder(folder: Folder) -> Result<Self, ArtifactError> {
        let text = folder
            .text("task.toml")
            .ok_or(ArtifactError::Missing("task.toml"))?;
        let parsed: TaskToml =
            toml::from_str(text).map_err(|e| ArtifactError::TaskToml(e.to_string()))?;
        let meta = parsed.metadata.splinter_env;
        if meta.schema_version != SCHEMA_VERSION {
            return Err(ArtifactError::Schema {
                found: meta.schema_version,
            });
        }
        for required in ["instruction.md", TEST_SCRIPT, SOLVE_SCRIPT] {
            if folder.get(required).is_none() {
                return Err(ArtifactError::Missing(required));
            }
        }
        Ok(Self { meta, folder })
    }

    /// What `task.toml` records.
    #[must_use]
    pub fn meta(&self) -> &EnvMeta {
        &self.meta
    }

    /// The whole task folder.
    #[must_use]
    pub fn folder(&self) -> &Folder {
        &self.folder
    }

    /// The content version of this instance: the address of every file.
    /// Identical for two builds of the same environment, split and seed.
    #[must_use]
    pub fn version(&self) -> Digest {
        self.folder.digest()
    }

    /// The instruction.
    #[must_use]
    pub fn instruction(&self) -> &str {
        self.folder.text("instruction.md").unwrap_or_default()
    }

    /// The workspace: `environment/` without the Dockerfile, paths relative
    /// to the workspace root.
    #[must_use]
    pub fn workspace(&self) -> Folder {
        let mut workspace = Folder::new();
        for (path, entry) in self.folder.under(ENVIRONMENT_DIR) {
            let relative = &path[ENVIRONMENT_DIR.len() + 1..];
            if relative != "Dockerfile" {
                workspace
                    .put(relative, entry.bytes.clone(), entry.executable)
                    .unwrap_or_default();
            }
        }
        workspace
    }

    /// Everything a solver is shown: the instruction and the workspace, and
    /// never `tests/` or `solution/`.
    #[must_use]
    pub fn solver_view(&self) -> Folder {
        let mut view = Folder::new();
        view.put_text("instruction.md", self.instruction())
            .unwrap_or_default();
        for (path, entry) in self.workspace().entries() {
            view.put(
                &format!("{ENVIRONMENT_DIR}/{path}"),
                entry.bytes.clone(),
                entry.executable,
            )
            .unwrap_or_default();
        }
        view
    }

    /// Where every reward of this instance came from.
    #[must_use]
    pub fn provenance(&self) -> RewardProvenance {
        RewardProvenance {
            class: self.meta.verifier_class,
            verifier: self.meta.verifier.clone(),
            environment: self.meta.environment.clone(),
            split: self.meta.split,
            seed: self.meta.seed,
        }
    }
}

fn dockerfile(packages: &[String]) -> String {
    let mut text = String::from("FROM ubuntu:24.04\n");
    if !packages.is_empty() {
        text.push_str(&format!(
            "RUN apt-get update && apt-get install -y --no-install-recommends {} && rm -rf /var/lib/apt/lists/*\n",
            packages.join(" ")
        ));
    }
    text.push_str("WORKDIR /app\nCOPY . /app\n");
    text.push_str("RUN if [ -f /app/reset.sh ]; then sh /app/reset.sh; fi\n");
    text
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;
    use crate::factory::testing::{meta, parts};

    #[test]
    fn an_artifact_round_trips_through_its_task_folder_with_its_provenance() {
        let instance = Instance::assemble(meta(), parts()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("t");
        instance.folder().write_to(&target).unwrap();
        let back = Instance::from_folder(Folder::read_from(&target).unwrap()).unwrap();
        assert_eq!(back.meta(), instance.meta());
        assert_eq!(back.version(), instance.version());
        let provenance = back.provenance();
        assert_eq!(provenance.class, VerifierClass::Objective);
        assert_eq!(provenance.split, Split::Train);
        assert_eq!(provenance.seed, 5);
    }

    #[test]
    fn the_solver_is_shown_the_instruction_and_workspace_and_never_the_tests_or_solution() {
        let instance = Instance::assemble(meta(), parts()).unwrap();
        let view = instance.solver_view();
        let shown: Vec<_> = view.paths().collect();
        assert_eq!(shown, ["environment/answer.txt", "instruction.md"]);
        assert!(!shown
            .iter()
            .any(|p| p.starts_with("tests") || p.starts_with("solution")));
    }

    #[test]
    fn a_split_other_than_train_or_dev_cannot_be_written_or_read() {
        let instance = Instance::assemble(meta(), parts()).unwrap();
        let text = instance.folder().text("task.toml").unwrap();
        assert!(text.contains("split = \"train\""), "{text}");
        let mut folder = instance.folder().clone();
        folder
            .put_text(
                "task.toml",
                &text.replace("split = \"train\"", "split = \"exam\""),
            )
            .unwrap();
        assert!(matches!(
            Instance::from_folder(folder),
            Err(ArtifactError::TaskToml(_))
        ));
    }

    #[test]
    fn a_newer_schema_is_refused_and_a_missing_verifier_is_named() {
        let instance = Instance::assemble(meta(), parts()).unwrap();
        let text = instance
            .folder()
            .text("task.toml")
            .unwrap()
            .replace("schema_version = 1", "schema_version = 9");
        let mut folder = instance.folder().clone();
        folder.put_text("task.toml", &text).unwrap();
        assert!(matches!(
            Instance::from_folder(folder),
            Err(ArtifactError::Schema { found: 9 })
        ));

        let mut broken = parts();
        broken.tests = Folder::new();
        assert!(matches!(
            Instance::assemble(meta(), broken),
            Err(ArtifactError::Missing("tests/test.sh"))
        ));
    }

    #[test]
    fn the_environment_id_ignores_split_and_seed_and_follows_difficulty_and_version() {
        let a = BTreeMap::from([("depth".to_string(), 2)]);
        let b = BTreeMap::from([("depth".to_string(), 3)]);
        assert_eq!(environment_id("p", "1", &a), environment_id("p", "1", &a));
        assert_ne!(environment_id("p", "1", &a), environment_id("p", "1", &b));
        assert_ne!(environment_id("p", "1", &a), environment_id("p", "2", &a));
    }
}
