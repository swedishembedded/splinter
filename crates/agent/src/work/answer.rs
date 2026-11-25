// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that grade an agent's answer with
// the task's own checks, whether it changed files or only replied, for its
// clients. If your team needs expertise in verifiable agent evaluation, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A task answered in the reply ([`super::WorkProtocol::answer_file`]): where
//! in the tree the reply goes, and writing it there.

use std::path::{Component, Path, PathBuf};

use splinter_sandbox::SandboxError;

use crate::solve::SolveError;

/// The answer file `path` names, relative to the tree; refused when it is
/// empty or would leave the tree.
pub(super) fn answer_path(path: &str) -> Result<PathBuf, SolveError> {
    let relative = Path::new(path);
    let inside = !path.is_empty()
        && relative
            .components()
            .all(|c| matches!(c, Component::Normal(_)));
    inside
        .then(|| relative.to_path_buf())
        .ok_or_else(|| SolveError::AnswerFile(path.to_string()))
}

/// Writes `reply` to `path` in `tree`, its directories made as needed.
pub(super) fn write_answer(tree: &Path, path: &Path, reply: &str) -> Result<(), SolveError> {
    let file = tree.join(path);
    let io = |source| SandboxError::Io {
        path: file.clone(),
        source,
    };
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    std::fs::write(&file, reply).map_err(io)?;
    Ok(())
}
