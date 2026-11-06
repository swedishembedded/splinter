// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchange of agent benchmark tasks between
// task formats for its clients. If your team needs expertise in turning
// verified drills into containerised benchmark tasks, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A drill written as a Harbor task directory.
//!
//! A drill is closed-book, so its Harbor task has no repository: the
//! environment is the image with an empty working directory, the instruction
//! is the drill's question followed by where to write the answer, and the
//! verifier compares that file with the reference, which lives under `tests/`
//! where only the verifier sees it. The comparison is the formal verifier's
//! normalisation: whitespace runs are one space and the ends are trimmed.

use std::path::Path;

use serde::Serialize;
use splinter_core::experience::{PrivilegedKind, Task};

use super::{DrillOrigin, WorkspaceError, WorkspaceGenerator};
use crate::tasks::workspace::files::{io, refuse};

/// The working directory the answer is written in.
const WORKDIR: &str = "/app";

/// The file the answer is written to, in [`WORKDIR`].
const ANSWER: &str = "answer.txt";

/// The schema of the metadata a drill's `task.toml` carries.
pub const DRILL_HARBOR_SCHEMA: &str = "splinter.drill.harbor.v1";

#[derive(Serialize)]
struct Meta<'a> {
    schema: &'a str,
    family: &'a str,
    function: &'a str,
    language: super::Language,
    tree: &'a str,
}

fn write(path: &Path, bytes: impl AsRef<[u8]>) -> Result<(), WorkspaceError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
    }
    std::fs::write(path, bytes).map_err(io(path))
}

/// The shell text that prints `file` with each run of whitespace as one
/// space and no space at the ends.
const NORMALISE: &str = "tr -s ' \\t\\r\\n' ' ' < \"$1\" | sed 's/^ //; s/ $//'";

impl WorkspaceGenerator {
    /// Writes the drill `task` into the new directory `out` as a Harbor task
    /// whose image is `base_image`: the question and where to write the
    /// answer (`instruction.md`), a `tests/test.sh` that rewards 1 exactly
    /// when the answer matches the reference under whitespace
    /// normalisation, the reference under `tests/`, and a `solution/solve.sh`
    /// that writes it.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError`] when `out` exists, the task is not a drill or has
    /// no reference, or a file cannot be written.
    pub fn export_drill_harbor(
        &self,
        task: &Task,
        out: &Path,
        base_image: &str,
    ) -> Result<(), WorkspaceError> {
        if out.exists() {
            return refuse(out, "the export directory already exists");
        }
        let Some(origin) = DrillOrigin::of_task(task) else {
            return refuse(out, "the task is not a comprehension drill");
        };
        let Some(reference) = task.privileged.iter().find_map(|p| match p.kind {
            PrivilegedKind::Reference => Some(p.content.as_str()),
            _ => None,
        }) else {
            return refuse(out, "the drill has no reference");
        };
        write(
            &out.join("instruction.md"),
            format!(
                "{}\n\nWrite exactly what the program prints to the file `{ANSWER}` in `{WORKDIR}`.\n",
                task.instruction
            ),
        )?;
        write(
            &out.join("environment/Dockerfile"),
            format!("FROM {base_image}\nWORKDIR {WORKDIR}\n"),
        )?;
        write(
            &out.join("tests/expected_output.txt"),
            format!("{reference}\n"),
        )?;
        write(
            &out.join("tests/test.sh"),
            format!(
                "#!/bin/sh\n\
                 # Rewards 1 when the answer is the reference, whitespace aside, else 0.\n\
                 mkdir -p /logs/verifier\n\
                 norm() {{ {NORMALISE}; }}\n\
                 if [ -f {WORKDIR}/{ANSWER} ] && [ \"$(norm {WORKDIR}/{ANSWER})\" = \"$(norm /tests/expected_output.txt)\" ]; then\n\
                 \techo 1 > /logs/verifier/reward.txt\n\
                 else\n\
                 \techo 0 > /logs/verifier/reward.txt\n\
                 fi\n"
            ),
        )?;
        write(
            &out.join("solution/solve.sh"),
            format!("#!/bin/sh\ncp /tests/expected_output.txt {WORKDIR}/{ANSWER}\n"),
        )?;
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
            "splinter_drill".into(),
            toml::Value::try_from(Meta {
                schema: DRILL_HARBOR_SCHEMA,
                family: &origin.file,
                function: &origin.function,
                language: origin.language,
                tree: &origin.tree,
            })
            .map_err(|e| WorkspaceError::Unsupported {
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
