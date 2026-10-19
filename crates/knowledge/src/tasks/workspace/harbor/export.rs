// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchange of agent benchmark tasks between
// task formats for its clients. If your team needs expertise in importing
// containerised benchmark tasks into reproducible, verifiable task sets, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A workspace task written as a Harbor task directory.

use std::path::Path;

use splinter_core::experience::{PrivilegedKind, Task};
use splinter_eval::verifiers::workspace::WorkspaceCheck;

use super::super::pack::lay_over;
use super::{io, refuse, script, SplinterMeta, SPLINTER_SCHEMA};
use crate::tasks::workspace::{WorkspaceError, WorkspaceGenerator};

/// The working directory an export copies the repository into.
const WORKDIR: &str = "/app";

fn write(path: &Path, bytes: impl AsRef<[u8]>) -> Result<(), WorkspaceError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
    }
    std::fs::write(path, bytes).map_err(io(path))
}

fn argvs(calls: &[splinter_sandbox::CommandCall]) -> Vec<Vec<String>> {
    calls.iter().map(|c| c.argv.clone()).collect()
}

impl WorkspaceGenerator {
    /// Writes the workspace `task` into the new directory `out` as a Harbor task: the
    /// request, the repository under `environment/workspace` with a
    /// Dockerfile that copies it (from `base_image`), the hidden files under
    /// `tests/overlay` with a `test.sh` that runs the check's commands and
    /// writes the reward, the reference change as `solution/reference.patch`
    /// with a `solve.sh` that applies it, and everything splinter needs to
    /// rebuild the task exactly in `[metadata.splinter]`.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError`] when `out` exists, an archive of the task is not in
    /// the store, or a file cannot be written.
    pub fn export_harbor(
        &self,
        task: &Task,
        out: &Path,
        base_image: &str,
    ) -> Result<(), WorkspaceError> {
        if out.exists() {
            return refuse(out, "the export directory already exists");
        }
        let Some((_, check)) =
            WorkspaceCheck::of_task(task).map_err(|e| WorkspaceError::Unsupported {
                what: format!("a task whose check cannot be read: {e}"),
            })?
        else {
            return refuse(out, "the task has no workspace check");
        };
        let workspace = out.join("environment/workspace");
        std::fs::create_dir_all(&workspace).map_err(io(&workspace))?;
        // The environment's own `snapshot` addresses the environment record;
        // the archive of the tree is the digest its spec names.
        let Some(snapshot) = task
            .environment
            .spec
            .get("snapshot")
            .and_then(serde_json::Value::as_str)
            .and_then(|text| splinter_core::digest::Digest::parse(text).ok())
        else {
            return refuse(out, "the task's environment names no snapshot archive");
        };
        lay_over(&self.store.path(&snapshot)?, &workspace)?;
        if let Some(hidden) = &check.hidden {
            let overlay = out.join("tests/overlay");
            std::fs::create_dir_all(&overlay).map_err(io(&overlay))?;
            lay_over(&self.store.path(hidden)?, &overlay)?;
        }
        write(
            &out.join("environment/Dockerfile"),
            format!("FROM {base_image}\nWORKDIR {WORKDIR}\nCOPY workspace/ {WORKDIR}/\n"),
        )?;
        write(
            &out.join("instruction.md"),
            format!("{}\n", task.instruction),
        )?;
        write(
            &out.join("tests/test.sh"),
            script::verifier(
                WORKDIR,
                &argvs(&check.fail_to_pass),
                &argvs(&check.pass_to_pass),
            ),
        )?;
        let reference = task.privileged.iter().find_map(|p| match p.kind {
            PrivilegedKind::Reference => Some(p.content.as_str()),
            _ => None,
        });
        if let Some(patch) = reference {
            write(&out.join("solution/reference.patch"), patch)?;
            write(
                &out.join("solution/solve.sh"),
                format!("#!/bin/sh\ncd {WORKDIR} && patch -p1 < /solution/reference.patch\n"),
            )?;
        }
        let meta = SplinterMeta {
            schema: SPLINTER_SCHEMA.to_string(),
            family: task.concepts.first().cloned().unwrap_or_default(),
            fail_to_pass: argvs(&check.fail_to_pass),
            pass_to_pass: argvs(&check.pass_to_pass),
            protected: check.protected.clone(),
            pinned: check.pinned.clone(),
            expectations: check.expectations.clone(),
            hidden_strength: check.hidden_strength,
        };
        let mut config = toml::Table::new();
        config.insert("schema_version".into(), "1.4".into());
        let mut named = toml::Table::new();
        named.insert(
            "name".into(),
            format!("splinter/{}", task.task.id.hex()).into(),
        );
        config.insert("task".into(), named.into());
        let mut metadata = toml::Table::new();
        metadata.insert(
            "splinter".into(),
            toml::Value::try_from(meta).map_err(|e| WorkspaceError::Unsupported {
                what: format!("metadata that is not TOML: {e}"),
            })?,
        );
        config.insert("metadata".into(), metadata.into());
        write(
            &out.join("task.toml"),
            toml::to_string(&config).map_err(|e| WorkspaceError::Unsupported {
                what: format!("a task.toml: {e}"),
            })?,
        )
    }
}
