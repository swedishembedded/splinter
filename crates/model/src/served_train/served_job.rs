// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that run any training job
// a model service offers across every card of a cluster without hand-made
// launch scripts, for its clients. If your team needs expertise in
// orchestrating distributed fine-tuning through a serving API, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Any `brain/train` action as one served job ([`run_served_job`]): the
//! request a caller writes (an action and its params), with its datasets
//! published to the artifact store and named by digest, spread as far as
//! the action allows, followed to its end, and its checkpoint fetched.
//!
//! * **The action decides the spread.** The servers' `GET /v1/capabilities`
//!   says which params the action takes. One that takes `cards` and `job`
//!   runs over the endpoints' whole training span
//!   ([`BrainEndpoints::train_span`]: every card of every server); one that
//!   takes `cards` only runs on the first server's cards; any other on one
//!   card. A narrower spread than the endpoints offer is said on stderr.
//!   `cards` and `job` are the span's, so a request that sets them is
//!   refused.
//! * **One seed for every rank.** An action that takes `seed` and was given
//!   none gets a drawn one, reported, so the run can be repeated.
//! * **Datasets by digest.** Each `(param, file)` artifact is published to
//!   the store the servers share and the param set to its digest; a param
//!   the action does not take is refused before anything is published.
//! * **Everything comes back verified.** The report's `files` (a checkpoint)
//!   and its adapter and training record are fetched by digest into the
//!   output directory, each checked against its digest, with the result
//!   beside them as `result.json`; the `parallel` report is checked against
//!   the spread asked for.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Map, Value};
use splinter_core::training::ParallelRecord;

use super::artifacts::{Fetched, ListedFile};
use crate::endpoint::{BrainEndpoints, TrainSpan};
use crate::error::PolicyError;

/// The file the result is written to in the output directory.
pub const RESULT_FILE: &str = "result.json";

/// The params a spread sets, never the caller.
const SPREAD_PARAMS: [&str; 2] = ["cards", "job"];

/// Single artifacts a training report names by digest, and the file each is
/// fetched to.
const REPORTED_DIGESTS: [(&str, &str); 2] = [
    ("adapter_digest", super::ADAPTER_FILE),
    ("record_digest", super::RECORD_FILE),
];

/// One served job, as its caller asks for it.
pub struct ServedJob<'a> {
    /// The `brain/train` action (`speech_sft`, `full_train`, `train`, ...).
    pub action: &'a str,
    /// Its params, as brain takes them.
    pub params: &'a Map<String, Value>,
    /// Files published to the artifact store, each param set to its file's
    /// digest (`dataset_digest`, `held_out_digest`, ...).
    pub artifacts: &'a [(String, PathBuf)],
    /// Where the files the job publishes are fetched to, with the result;
    /// `None` fetches nothing.
    pub out_dir: Option<&'a Path>,
    /// Fired, the job is cancelled on every server.
    pub cancel: Option<&'a sven_sdk::CancelToken>,
}

/// The seed the job's ranks start from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SeedUsed {
    /// Its value.
    pub value: u64,
    /// Whether it was drawn here because the request gave none.
    pub drawn: bool,
}

/// A file published for the job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Published {
    /// The param that names it.
    pub param: String,
    /// The file.
    pub file: PathBuf,
    /// Its BLAKE3 digest, in hex: the param's value.
    pub digest: String,
}

/// What a served job came to.
#[derive(Clone, Debug, Serialize)]
pub struct ServedJobReport {
    /// The action.
    pub action: String,
    /// The seed, when the action takes one.
    pub seed: Option<SeedUsed>,
    /// The files published for it.
    pub artifacts: Vec<Published>,
    /// The spread asked for, what brain reported and where they differ.
    pub parallel: ParallelRecord,
    /// The files fetched into the output directory.
    pub fetched: Vec<Fetched>,
    /// Rank 0's server's result (`outputs`).
    pub result: Value,
    /// From the first job created to the result, in seconds.
    pub seconds: f64,
}

/// The names of the params `action` of `brain/train` takes in a capability
/// listing (`{"data": [{"model", "action", "params": [{"name"}]}]}`), or
/// `None` when the listing has no such action.
pub(crate) fn action_params(listing: &Value, action: &str) -> Option<Vec<String>> {
    listing["data"]
        .as_array()?
        .iter()
        .find(|entry| entry["model"] == "brain/train" && entry["action"] == action)
        .map(|entry| {
            entry["params"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|param| param["name"].as_str().map(str::to_string))
                .collect()
        })
}

/// The `brain/train` actions a capability listing names.
fn train_actions(listing: &Value) -> Vec<String> {
    listing["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|entry| entry["model"] == "brain/train")
        .filter_map(|entry| entry["action"].as_str().map(str::to_string))
        .collect()
}

/// The span `wanted` narrowed to what an action that takes `params` can be
/// spread over, and why when it was narrowed.
fn span_for(wanted: &TrainSpan, params: &[String]) -> (TrainSpan, Option<&'static str>) {
    let takes = |name: &str| params.iter().any(|p| p == name);
    match wanted {
        TrainSpan::OneCard => (TrainSpan::OneCard, None),
        TrainSpan::Cards(_) if takes("cards") => (wanted.clone(), None),
        TrainSpan::Servers(_) if takes("cards") && takes("job") => (wanted.clone(), None),
        TrainSpan::Servers(job) if takes("cards") => (
            TrainSpan::Cards(job.cards.first().copied().unwrap_or(1)),
            Some("it takes cards but no job: it runs on the first server's cards"),
        ),
        _ => (
            TrainSpan::OneCard,
            Some("it takes no cards: it runs on one card"),
        ),
    }
}

/// Runs `job` on `brain`'s endpoints as the module says, handing every step
/// frame of rank 0's server to `on_frame`.
///
/// # Errors
/// The request names an action or a param the servers do not take, sets
/// `cards` or `job`, needs an artifact store none is configured for, or the
/// job was refused, failed or was cancelled, or a file it published does not
/// arrive as its digest says.
pub fn run_served_job(
    brain: &BrainEndpoints,
    job: &ServedJob<'_>,
    on_frame: &mut dyn FnMut(&Value),
) -> Result<ServedJobReport, PolicyError> {
    let place = job
        .out_dir
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    let refused = |reason: String| PolicyError::Train {
        dir: place.clone(),
        reason,
    };
    if let Some(set) = SPREAD_PARAMS.iter().find(|p| job.params.contains_key(**p)) {
        return Err(refused(format!(
            "{set} is set by how the job is spread over the servers' cards, not by the request"
        )));
    }
    let listing = brain.get(0, "capabilities")?;
    let takes = action_params(&listing, job.action).ok_or_else(|| {
        refused(format!(
            "the servers' brain/train has no action {:?}; it has: {}",
            job.action,
            train_actions(&listing).join(", ")
        ))
    })?;
    if let Some((param, _)) = job.artifacts.iter().find(|(p, _)| !takes.contains(p)) {
        return Err(refused(format!(
            "brain/train {} takes no param {param:?} to name a file by; it takes: {}",
            job.action,
            takes.join(", ")
        )));
    }
    let (span, narrowed) = span_for(brain.train_span(), &takes);
    if let Some(why) = narrowed {
        eprintln!("brain: {} {why}", job.action);
    }
    let spread = brain.clone().with_train_span(span);
    let mut params = job.params.clone();
    let seed = takes.iter().any(|p| p == "seed").then(|| {
        match params.get("seed").and_then(Value::as_u64) {
            Some(value) => SeedUsed {
                value,
                drawn: false,
            },
            None => {
                let value = crate::seed::draw().value();
                eprintln!(
                    "brain: {} seed {value} (drawn; pass \"seed\" to repeat it)",
                    job.action
                );
                params.insert("seed".into(), Value::from(value));
                SeedUsed { value, drawn: true }
            }
        }
    });
    let mut artifacts = Vec::with_capacity(job.artifacts.len());
    if !job.artifacts.is_empty() {
        let store = brain.artifact_store()?;
        for (param, file) in job.artifacts {
            let digest = store.publish(file)?.hex().to_string();
            params.insert(param.clone(), Value::from(digest.clone()));
            artifacts.push(Published {
                param: param.clone(),
                file: file.clone(),
                digest,
            });
        }
    }
    let started = std::time::Instant::now();
    let result = super::run_spread_action(
        &spread,
        &place,
        // Only which endpoint is asked first, for an action that takes no seed.
        seed.map_or_else(|| crate::seed::draw().value(), |s| s.value),
        job.action,
        &Value::Object(params),
        job.cancel,
        on_frame,
    )?;
    let seconds = started.elapsed().as_secs_f64();
    let parallel = super::parallel::observe(&spread, &result).map_err(refused)?;
    let fetched = match job.out_dir {
        Some(dir) => fetch(brain, &result, dir)?,
        None => Vec::new(),
    };
    Ok(ServedJobReport {
        action: job.action.to_string(),
        seed,
        artifacts,
        parallel,
        fetched,
        result,
        seconds,
    })
}

/// The files `result` names (its checkpoint's `files`, its adapter and its
/// training record) fetched into `dir`, and the result written beside them.
fn fetch(brain: &BrainEndpoints, result: &Value, dir: &Path) -> Result<Vec<Fetched>, PolicyError> {
    let unusable = |reason: String| PolicyError::Train {
        dir: dir.to_path_buf(),
        reason,
    };
    let mut files: Vec<ListedFile> = match &result["files"] {
        Value::Null => Vec::new(),
        listed => serde_json::from_value(listed.clone())
            .map_err(|e| unusable(format!("the result's files: {e}")))?,
    };
    for (field, name) in REPORTED_DIGESTS {
        if let Some(digest) = result[field].as_str() {
            files.push(ListedFile {
                name: name.to_string(),
                digest: digest.to_string(),
            });
        }
    }
    std::fs::create_dir_all(dir).map_err(|source| PolicyError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    let fetched = if files.is_empty() {
        Vec::new()
    } else {
        brain.artifact_store()?.fetch_listed(&files, dir)?
    };
    let path = dir.join(RESULT_FILE);
    let text = serde_json::to_vec_pretty(result).map_err(|e| unusable(e.to_string()))?;
    std::fs::write(&path, text).map_err(|source| PolicyError::Io { path, source })?;
    Ok(fetched)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoint::ServersJob;

    #[test]
    fn an_action_is_spread_only_as_far_as_its_params_allow() {
        let servers = TrainSpan::Servers(ServersJob {
            cards: vec![2, 4],
            coordinator: "10.0.0.1:1".into(),
            secret_sha256: "ab".repeat(32),
        });
        let names = |list: &[&str]| list.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        assert_eq!(
            span_for(&servers, &names(&["cards", "job"])),
            (servers.clone(), None)
        );
        assert_eq!(
            span_for(&servers, &names(&["cards"])).0,
            TrainSpan::Cards(2)
        );
        assert_eq!(span_for(&servers, &names(&["seed"])).0, TrainSpan::OneCard);
        assert_eq!(
            span_for(&TrainSpan::Cards(4), &names(&["job"])).0,
            TrainSpan::OneCard
        );
        assert_eq!(
            span_for(&TrainSpan::OneCard, &names(&["cards"])),
            (TrainSpan::OneCard, None)
        );
    }
}
