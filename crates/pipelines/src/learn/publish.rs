// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents whose trained adapters are
// selected by name like any other model, for its clients. If your team needs
// expertise in managing fine-tuned adapters across a model store, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The publish stage: the candidate a named run trained becomes the adapter of
//! that name in the model store, where `local:<checkpoint>+<name>` finds it and
//! the next run of the same name continues it.

use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::model_ref::is_alias_name;
use splinter_orchestrator::model_ref::named_adapter_path;
use splinter_orchestrator::pipeline::StageEnd;
use splinter_orchestrator::OrchestratorError;

use super::stages::{to_value, Done, LearnState};

/// The adapter a run is to leave in the store.
#[derive(Clone, Debug)]
pub(super) struct NamedAdapter {
    /// Its name.
    pub(super) name: String,
    /// The file it is kept as, in the checkpoint's adapter store.
    pub(super) path: PathBuf,
}

/// Where a named adapter was published.
#[derive(Clone, Debug, Serialize)]
pub struct Published {
    /// The adapter's name.
    pub adapter: String,
    /// The file in the model store.
    pub path: PathBuf,
}

/// Refuses an adapter name that is not one: the name is a path component of
/// the store, so it is held to the alias rule.
pub(super) fn check_name(name: &str) -> Result<(), OrchestratorError> {
    if is_alias_name(name) {
        Ok(())
    } else {
        Err(OrchestratorError::Refused(format!(
            "{name:?} is not an adapter name: a lowercase letter, then up to 63 lowercase \
             letters, digits, - or _"
        )))
    }
}

/// The adapter `name` on the checkpoint at `base`.
pub(super) fn named(base: &Path, name: &str) -> NamedAdapter {
    NamedAdapter {
        name: name.to_string(),
        path: named_adapter_path(base, name, "latest"),
    }
}

/// Copies `from` to `to` so that a reader sees the old file or the new one,
/// never a part of it.
pub(super) fn install(from: &Path, to: &Path) -> Result<(), OrchestratorError> {
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |source| OrchestratorError::Io { path, source }
    };
    let dir = to.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    let partial = to.with_extension(format!("partial-{}", std::process::id()));
    std::fs::copy(from, &partial).map_err(io(&partial))?;
    std::fs::rename(&partial, to).map_err(io(to))
}

pub(super) fn publish_stage(
    _: &splinter_orchestrator::Context,
    _: &mut splinter_orchestrator::runs::Recorder<'_>,
    st: &mut LearnState<'_>,
) -> Done {
    let (Some(named), Some(candidate)) = (st.learn.adapter.as_ref(), st.report.candidate.as_ref())
    else {
        unreachable!("the publish stage runs for a named adapter, after the train stage")
    };
    install(&candidate.adapter, &named.path)?;
    let published = Published {
        adapter: named.name.clone(),
        path: named.path.clone(),
    };
    let summary = to_value(&published)?;
    st.report.published = Some(published);
    Ok(StageEnd::done(summary))
}
