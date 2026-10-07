// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Where a loop run keeps its files: `<home>/.sven/loop/runs/<run>/`.
//!
//! The run's records are written atomically (a temporary file in the same
//! directory, then a rename), so a reader never sees half a record and a
//! crash leaves the previous version in place.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// The contract a run was started with.
pub const CONTRACT_FILE: &str = "contract.json";
/// The last durable point of a run.
pub const CHECKPOINT_FILE: &str = "checkpoint.json";
/// The structured result of a run.
pub const OUTCOME_FILE: &str = "outcome.json";
/// The patch of the accepted or final candidate.
pub const PATCH_FILE: &str = "patch.diff";
/// The isolated checkout the worker edits.
pub const WORK_DIR: &str = "work";

/// The loop's namespace under a home directory.
#[derive(Clone, Debug)]
pub struct LoopHome {
    root: PathBuf,
}

impl LoopHome {
    /// `<home>/.sven/loop`, beside sven's own files and splinter's.
    #[must_use]
    pub fn under(home: &Path) -> Self {
        Self {
            root: home.join(".sven").join("loop"),
        }
    }

    /// The namespace itself.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The directory of run `run`.
    #[must_use]
    pub fn run_dir(&self, run: &str) -> PathBuf {
        self.root.join("runs").join(run)
    }

    /// The operator's override of the system prompt, if one is kept.
    #[must_use]
    pub fn system_prompt_override(&self) -> PathBuf {
        self.root.join("system.md")
    }
}

/// Writes `bytes` to `path` atomically.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().context("a record needs a directory")?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .context("a record needs a file name")?;
    let temporary = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    std::fs::write(&temporary, bytes)
        .with_context(|| format!("writing {}", temporary.display()))?;
    std::fs::rename(&temporary, path).with_context(|| format!("replacing {}", path.display()))
}

/// Writes `value` to `path` as pretty JSON, atomically.
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value).context("encoding a record")?;
    text.push('\n');
    write_atomic(path, text.as_bytes())
}

/// The JSON record at `path`.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("{} is not a valid record", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_is_replaced_whole_and_leaves_no_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("record.json");
        write_json(&path, &serde_json::json!({"n": 1})).unwrap();
        write_json(&path, &serde_json::json!({"n": 2})).unwrap();
        let back: serde_json::Value = read_json(&path).unwrap();
        assert_eq!(back["n"], 2);
        let names: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names.len(), 1, "{names:?}");
    }

    #[test]
    fn the_namespace_sits_beside_sven_and_splinter() {
        let home = LoopHome::under(Path::new("h"));
        assert_eq!(home.run_dir("r"), Path::new("h/.sven/loop/runs/r"));
    }
}
