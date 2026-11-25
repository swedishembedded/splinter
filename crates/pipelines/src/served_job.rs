// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that run any training job
// a model service offers across every card of a cluster without hand-made
// launch scripts, for its clients. If your team needs expertise in
// orchestrating distributed fine-tuning through a serving API, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `jobs run-served`: one `brain/train` action, its params written by the
//! caller in a file, run on the brain servers of the configuration (the
//! fleet this process starts on the allocation's nodes, a server it starts
//! here, or the endpoints it was given) and spread as far as the action
//! allows (`splinter_model::served_train::run_served_job`).

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Map, Value};
use splinter_agent::CancelToken;
use splinter_model::served_train::{run_served_job, ServedJob, ServedJobReport};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::OrchestratorError;

/// The most bytes a params file may hold: params name datasets by digest,
/// they never carry them.
pub const MAX_PARAMS_BYTES: u64 = 1 << 20;

/// What `jobs run-served` is asked.
#[derive(Clone, Debug, Serialize)]
pub struct ServedJobRequest {
    /// The `brain/train` action.
    pub action: String,
    /// The file holding its params, one JSON object.
    pub params: PathBuf,
    /// `(param, file)`: each file published to the artifact store and the
    /// param set to its digest.
    pub artifacts: Vec<(String, PathBuf)>,
    /// Where the files the job publishes are fetched to, with its result.
    pub out_dir: Option<PathBuf>,
}

/// Reads the params file: one JSON object of at most [`MAX_PARAMS_BYTES`].
///
/// # Errors
/// The file cannot be read, is too large, or is not a JSON object.
pub fn read_params(file: &Path) -> Result<Map<String, Value>, OrchestratorError> {
    let io = |source| OrchestratorError::Io {
        path: file.to_path_buf(),
        source,
    };
    let length = std::fs::metadata(file).map_err(io)?.len();
    if length > MAX_PARAMS_BYTES {
        return Err(OrchestratorError::Refused(format!(
            "{}: {length} bytes of params; at most {MAX_PARAMS_BYTES} (a dataset is named by \
             --artifact, never inlined)",
            file.display()
        )));
    }
    let text = std::fs::read(file).map_err(io)?;
    match serde_json::from_slice::<Value>(&text) {
        Ok(Value::Object(params)) => Ok(params),
        Ok(_) => Err(OrchestratorError::Refused(format!(
            "{}: the params are one JSON object",
            file.display()
        ))),
        Err(source) => Err(OrchestratorError::Json {
            what: file.display().to_string(),
            source,
        }),
    }
}

/// Runs `request` on `ctx`'s brain, handing rank 0's step frames to
/// `on_frame`; `cancel` stops it on every server.
///
/// # Errors
/// The params cannot be read, no server can be reached or started, or the
/// job was refused, failed or was cancelled.
pub fn run(
    ctx: &Context,
    request: &ServedJobRequest,
    cancel: &CancelToken,
    on_frame: &mut dyn FnMut(&Value),
) -> Result<ServedJobReport, OrchestratorError> {
    let params = read_params(&request.params)?;
    if let Some((_, file)) = request.artifacts.iter().find(|(_, f)| !f.is_file()) {
        return Err(OrchestratorError::Refused(format!(
            "{} is not a file to publish",
            file.display()
        )));
    }
    run_served_job(
        &ctx.brain()?,
        &ServedJob {
            action: &request.action,
            params: &params,
            artifacts: &request.artifacts,
            out_dir: request.out_dir.as_deref(),
            cancel: Some(cancel),
        },
        on_frame,
    )
    .map_err(crate::train::trainer_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_are_one_bounded_json_object() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, text: &str| {
            let path = dir.path().join(name);
            std::fs::write(&path, text).unwrap();
            path
        };
        let params = read_params(&write("ok.json", r#"{"steps": 3}"#)).unwrap();
        assert_eq!(params["steps"], 3);
        assert!(read_params(&write("list.json", "[1]")).is_err());
        assert!(read_params(&write("bad.json", "{")).is_err());
        let huge = "x".repeat(usize::try_from(MAX_PARAMS_BYTES).unwrap() + 1);
        let refused = read_params(&write("huge.json", &format!("{{\"a\":\"{huge}\"}}")))
            .unwrap_err()
            .to_string();
        assert!(refused.contains("at most"), "{refused}");
    }
}
