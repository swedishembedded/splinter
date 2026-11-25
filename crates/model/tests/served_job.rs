// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that run any training job
// a model service offers across every card of a cluster without hand-made
// launch scripts, for its clients. If your team needs expertise in
// orchestrating distributed fine-tuning through a serving API, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: any `brain/train` action runs as one served job. Its datasets go
//! into the artifact store and are named by digest; it is spread over every
//! card of every server when the action takes `cards` and `job` (as the
//! servers' `GET /v1/capabilities` lists it), else it runs on one card; rank
//! 0's step frames reach the caller; a failure or a cancel on any server
//! stops every server's run; the checkpoint it publishes is fetched by
//! digest, verified, into the output directory; and the `parallel` report is
//! checked against what was asked.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::{json, Map, Value};
use splinter_core::digest::Digest;
use splinter_model::endpoint::{BrainEndpoints, ServersJob, TrainSpan};
use splinter_model::scripted_endpoint::{ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::scripted_jobs::JobScript;
use splinter_model::served_train::{run_served_job, ArtifactStore, ServedJob};

/// `speech_sft` as brain lists it, with `cards` and `job` when `spreads`.
fn capabilities(spreads: bool) -> Value {
    let mut params = vec!["base", "dataset_digest", "held_out_digest", "steps", "seed"];
    if spreads {
        params.extend(["cards", "job"]);
    }
    json!([{"model": "brain/train", "action": "speech_sft",
            "params": params.iter().map(|p| json!({"name": p})).collect::<Vec<_>>()}])
}

/// What the scripted servers report for a world of 8 ranks, rank 0's server holding `[0, 4)`.
fn parallel() -> Value {
    json!({"world_size": 8, "ranks": [0, 4],
           "devices": (0..4).map(|r| json!({"rank": r, "device": format!("gpu{r}"), "name": "GH200"})).collect::<Vec<_>>(),
           "collective": "nccl", "transport": "tcp", "network": null, "sharding": "replicated"})
}

struct Fleet {
    dir: tempfile::TempDir,
    store: PathBuf,
    servers: Vec<ScriptedEndpoint>,
}

impl Fleet {
    /// Two servers over one artifact store, as a fleet shares it.
    fn start(spreads: bool, script: impl Fn(JobScript) -> JobScript) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("artifacts");
        std::fs::create_dir_all(&store).unwrap();
        let servers = (0..2)
            .map(|_| {
                ScriptedEndpoint::start(vec![])
                    .with_capabilities(capabilities(spreads))
                    .with_training(script(JobScript {
                        parallel: Some(parallel()),
                        ..JobScript::over(store.clone())
                    }))
            })
            .collect();
        Self {
            dir,
            store,
            servers,
        }
    }

    fn brain(&self) -> BrainEndpoints {
        BrainEndpoints::new(
            self.servers.iter().map(ScriptedEndpoint::url).collect(),
            SCRIPTED_KEY,
        )
        .unwrap()
        .with_artifact_store(ArtifactStore::open(&self.store).unwrap())
        .with_refused_wait(std::time::Duration::from_millis(10))
        .with_train_span(TrainSpan::Servers(ServersJob {
            cards: vec![4, 4],
            coordinator: "127.0.0.1:29500".into(),
            secret_sha256: "ab".repeat(32),
        }))
    }

    fn shard(&self, name: &str) -> PathBuf {
        let path = self.dir.path().join(name);
        std::fs::write(&path, format!("the {name} shard")).unwrap();
        path
    }

    fn artifacts(&self) -> Vec<(String, PathBuf)> {
        vec![
            ("dataset_digest".into(), self.shard("train")),
            ("held_out_digest".into(), self.shard("held-out")),
        ]
    }
}

fn params() -> Map<String, Value> {
    json!({"base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "steps": 3})
        .as_object()
        .unwrap()
        .clone()
}

#[test]
fn an_action_that_takes_cards_and_job_spreads_over_every_server_with_its_datasets_by_digest() {
    let fleet = Fleet::start(true, |script| script);
    let out = fleet.dir.path().join("out");
    let artifacts = fleet.artifacts();
    let mut frames = Vec::new();
    let report = run_served_job(
        &fleet.brain(),
        &ServedJob {
            action: "speech_sft",
            params: &params(),
            artifacts: &artifacts,
            out_dir: Some(&out),
            cancel: None,
        },
        &mut |frame| frames.push(frame.clone()),
    )
    .unwrap();

    let digest = |file: &Path| Digest::of(&std::fs::read(file).unwrap()).hex().to_string();
    let mut seeds = Vec::new();
    for (server, first_rank) in fleet.servers.iter().zip([0, 4]) {
        let created = server.jobs().created();
        assert_eq!(created.len(), 1, "one run per server");
        let run = &created[0];
        assert_eq!(run["cards"], 4);
        let part: Value = serde_json::from_str(run["job"].as_str().unwrap()).unwrap();
        assert_eq!(part["first_rank"], first_rank);
        assert_eq!(part["plan"]["world_size"], 8);
        assert_eq!(run["dataset_digest"], digest(&artifacts[0].1));
        assert_eq!(run["held_out_digest"], digest(&artifacts[1].1));
        assert_eq!(run["steps"], 3);
        seeds.push(run["seed"].as_u64().expect("a drawn seed"));
    }
    assert_eq!(seeds[0], seeds[1], "every rank starts from one seed");
    assert_eq!(
        report.seed.as_ref().map(|s| (s.value, s.drawn)),
        Some((seeds[0], true))
    );
    assert!(
        frames.iter().any(|f| f["type"] == "step"),
        "rank 0's step frames reach the caller: {frames:?}"
    );
    assert_eq!(report.parallel.requested.world_size, 8);
    assert!(
        report.parallel.warnings().is_empty(),
        "{:?}",
        report.parallel
    );

    // The checkpoint, fetched by digest into the output directory and checked.
    let files = report.result["files"].as_array().unwrap();
    assert!(!files.is_empty());
    for file in files {
        let fetched = out.join(file["name"].as_str().unwrap());
        assert_eq!(digest(&fetched), file["digest"].as_str().unwrap());
    }
    assert_eq!(report.fetched.len(), files.len());
    let written: Value =
        serde_json::from_slice(&std::fs::read(out.join("result.json")).unwrap()).unwrap();
    assert_eq!(
        written, report.result,
        "the result is kept beside the files"
    );
}

#[test]
fn an_action_that_takes_no_job_runs_on_one_card_of_one_server() {
    let fleet = Fleet::start(false, |script| script);
    let artifacts = fleet.artifacts();
    let report = run_served_job(
        &fleet.brain(),
        &ServedJob {
            action: "speech_sft",
            params: &params(),
            artifacts: &artifacts,
            out_dir: None,
            cancel: None,
        },
        &mut |_| {},
    )
    .unwrap();
    let created: Vec<Value> = fleet
        .servers
        .iter()
        .flat_map(|s| s.jobs().created())
        .collect();
    assert_eq!(created.len(), 1, "one server runs it");
    assert!(created[0].get("cards").is_none() && created[0].get("job").is_none());
    assert_eq!(report.parallel.requested.world_size, 1);
    assert!(report.fetched.is_empty(), "nothing fetched unless asked");
}

#[test]
fn a_failure_on_one_server_fails_the_job_and_stops_the_other() {
    let fleet = Fleet::start(true, |script| script);
    // The second server's run fails; the first hangs until it is cancelled.
    let fleet = Fleet {
        servers: vec![
            ScriptedEndpoint::start(vec![])
                .with_capabilities(capabilities(true))
                .with_training(JobScript {
                    hang: true,
                    ..JobScript::over(fleet.store.clone())
                }),
            ScriptedEndpoint::start(vec![])
                .with_capabilities(capabilities(true))
                .with_training(JobScript {
                    fail_with: Some("rank 5 lost its card".into()),
                    ..JobScript::over(fleet.store.clone())
                }),
        ],
        ..fleet
    };
    let artifacts = fleet.artifacts();
    let error = run_served_job(
        &fleet.brain(),
        &ServedJob {
            action: "speech_sft",
            params: &params(),
            artifacts: &artifacts,
            out_dir: None,
            cancel: None,
        },
        &mut |_| {},
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("rank 5 lost its card"), "{error}");
    assert!(
        fleet.servers[0].jobs().was_cancelled(),
        "the other run is stopped"
    );
}

#[test]
fn a_cancel_stops_every_servers_run() {
    let fleet = Fleet::start(true, |script| JobScript {
        hang: true,
        ..script
    });
    let artifacts = fleet.artifacts();
    let cancel = sven_sdk::CancelToken::new();
    let firing = cancel.clone();
    let fired = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&fired);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(800));
        seen.store(true, Ordering::SeqCst);
        firing.cancel();
    });
    let error = run_served_job(
        &fleet.brain(),
        &ServedJob {
            action: "speech_sft",
            params: &params(),
            artifacts: &artifacts,
            out_dir: None,
            cancel: Some(&cancel),
        },
        &mut |_| {},
    )
    .unwrap_err()
    .to_string();
    assert!(fired.load(Ordering::SeqCst));
    assert!(error.contains("cancel"), "{error}");
    assert!(fleet.servers.iter().all(|s| s.jobs().was_cancelled()));
}

#[test]
fn a_request_brain_could_not_run_as_asked_is_refused_before_any_job() {
    let fleet = Fleet::start(true, |script| script);
    let brain = fleet.brain();
    let run = |action: &str, params: Map<String, Value>, artifacts: Vec<(String, PathBuf)>| {
        run_served_job(
            &brain,
            &ServedJob {
                action,
                params: &params,
                artifacts: &artifacts,
                out_dir: None,
                cancel: None,
            },
            &mut |_| {},
        )
        .unwrap_err()
        .to_string()
    };
    let unknown = run("speech_dpo", params(), vec![]);
    assert!(
        unknown.contains("speech_dpo") && unknown.contains("speech_sft"),
        "{unknown}"
    );
    let stray = run(
        "speech_sft",
        params(),
        vec![("voices_digest".into(), fleet.shard("voices"))],
    );
    assert!(stray.contains("voices_digest"), "{stray}");
    let mut spread_by_hand = params();
    spread_by_hand.insert("job".into(), json!("{}"));
    let manual = run("speech_sft", spread_by_hand, vec![]);
    assert!(manual.contains("job"), "{manual}");
    assert!(
        fleet.servers.iter().all(|s| s.jobs().created().is_empty()),
        "no job was started"
    );
}
