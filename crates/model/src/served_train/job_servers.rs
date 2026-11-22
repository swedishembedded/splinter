// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that train one model on
// the cards of several machines through their model servers' APIs, for its
// clients. If your team needs expertise in data-parallel fine-tuning across
// a cluster allocation, you can procure our services by sending an email to
// info@swedishembedded.com.

//! One training run as one data-parallel job over several servers (brain's
//! "one job over several servers"): the same `train` request goes to every
//! server with that server's `cards` and a `job` document naming its ranks,
//! the plan every server shares and the digest of the secret they were
//! started with. Every server's run is followed; the result is rank 0's
//! server's; a failure or a cancel on any server cancels the others and
//! fails the run.

use std::sync::mpsc::{channel, Sender};
use std::time::Duration;

use futures::stream::{FuturesUnordered, StreamExt};
use serde_json::{json, Value};

use super::{Api, Cancel, JobError};
use crate::endpoint::ServersJob;

/// How long every rank may take to join the job: each server's run first
/// waits for its cards.
const HANDSHAKE_MS: u64 = 600_000;

/// How long one collective may take.
const OP_MS: u64 = 600_000;

/// The request each server is sent: `params` with its `cards` and its part
/// of the job, ranks numbered over the servers in order.
pub(super) fn server_params(params: &Value, job: &ServersJob, name: &str) -> Vec<Value> {
    let world: u32 = job.cards.iter().sum();
    let plan = json!({
        "job": name,
        "world_size": world,
        "tp": 1,
        "pp": 1,
        "dp": world,
        "coordinator": job.coordinator,
        "handshake_timeout_ms": HANDSHAKE_MS,
        "op_timeout_ms": OP_MS,
    });
    let mut first_rank = 0u32;
    job.cards
        .iter()
        .map(|&cards| {
            let mut mine = params.clone();
            mine["cards"] = json!(cards);
            // brain takes the document as a string parameter.
            mine["job"] = json!(json!({
                "plan": plan,
                "first_rank": first_rank,
                "secret_sha256": job.secret_sha256,
            })
            .to_string());
            first_rank += cards;
            mine
        })
        .collect()
}

/// Runs `action` as one job over every server of `urls`, frames from rank 0's
/// server to `frames`, and returns rank 0's server's report.
pub(super) fn drive(
    urls: &[String],
    key: &str,
    refused_wait: Duration,
    job: &ServersJob,
    (action, params): (&str, &Value),
    cancel: Option<&dyn Cancel>,
    frames: &Sender<Value>,
) -> Result<Value, JobError> {
    if params.get("seed").is_none_or(Value::is_null) {
        return Err(JobError::Failed(
            "a job over several servers needs an explicit seed: every rank starts from it".into(),
        ));
    }
    if job.cards.len() != urls.len() {
        return Err(JobError::Failed(format!(
            "the job names cards for {} servers, and there are {}",
            job.cards.len(),
            urls.len()
        )));
    }
    let name = format!(
        "splinter-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    );
    let mut params = params.clone();
    if params.get("resume_key").is_some_and(|key| !key.is_null()) {
        // brain refuses to resume a job across servers; it runs from its start.
        eprintln!("brain: a job over several servers does not resume: running it from its start");
        params.as_object_mut().map(|p| p.remove("resume_key"));
    }
    let bodies = server_params(&params, job, &name);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| JobError::Failed(format!("starting a runtime: {e}")))?;
    runtime.block_on(async {
        let client = sven_sdk::http::model_client_builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(15))
            .build()
            .map_err(|e| JobError::Failed(e.to_string()))?;
        let api = Api {
            client,
            key,
            cancel,
        };
        // Every server at once: a server's run waits for the others' ranks.
        let created = futures::future::join_all(urls.iter().zip(&bodies).map(|(url, body)| {
            api.create(std::slice::from_ref(url), refused_wait, 0, action, body)
        }))
        .await;
        let mut placed = Vec::with_capacity(created.len());
        let mut failure = None;
        for outcome in created {
            match outcome {
                Ok(job) => placed.push(job),
                Err(error) => failure = failure.or(Some(error)),
            }
        }
        if let Some(error) = failure {
            let stopped = cancel_all(&api, &placed).await;
            forget_all(&api, &placed).await;
            return Err(with_statement(error, &stopped));
        }
        // Only rank 0's server's frames are the run's; the others' are dropped.
        let (others, _dropped) = channel::<Value>();
        let api = &api;
        let mut following: FuturesUnordered<_> = placed
            .iter()
            .enumerate()
            .map(|(at, (url, id))| {
                let sink = if at == 0 {
                    frames.clone()
                } else {
                    others.clone()
                };
                async move { (at, api.follow(url, id, &sink).await) }
            })
            .collect();
        let mut done = vec![false; placed.len()];
        while let Some((at, outcome)) = following.next().await {
            match outcome {
                Ok(()) => done[at] = true,
                Err(error) => {
                    drop(following);
                    let running: Vec<(String, String)> = placed
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !done[*i])
                        .map(|(_, job)| job.clone())
                        .collect();
                    let stopped = cancel_all(api, &running).await;
                    forget_all(api, &placed).await;
                    let (url, _) = &placed[at];
                    let error = match error {
                        JobError::Failed(why) => {
                            JobError::Failed(format!("the job's run on {url} failed: {why}"))
                        }
                        other => other,
                    };
                    return Err(with_statement(error, &stopped));
                }
            }
        }
        let (url, id) = &placed[0];
        let report = api.result(url, id).await;
        forget_all(api, &placed).await;
        report
    })
}

/// Forgets every ended job of `jobs` on its server.
async fn forget_all(api: &Api<'_>, jobs: &[(String, String)]) {
    futures::future::join_all(jobs.iter().map(|(url, id)| api.forget(url, id))).await;
}

/// Asks every server of `jobs` to stop its run; what each answered.
async fn cancel_all(api: &Api<'_>, jobs: &[(String, String)]) -> String {
    let answers = futures::future::join_all(jobs.iter().map(|(url, id)| api.cancel(url, id))).await;
    answers.join("; ")
}

/// `error` with what stopping the other servers' runs came to.
fn with_statement(error: JobError, stopped: &str) -> JobError {
    if stopped.is_empty() {
        return error;
    }
    match error {
        JobError::Cancelled(_) => JobError::Cancelled(Some(stopped.to_string())),
        JobError::Failed(why) => JobError::Failed(format!("{why}; {stopped}")),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_server_gets_its_cards_and_its_ranks_of_one_plan() {
        let job = ServersJob {
            cards: vec![2, 1],
            coordinator: "10.0.0.1:29500".into(),
            secret_sha256: "ab".repeat(32),
        };
        let params = json!({"seed": 7, "steps": 4});
        let bodies = server_params(&params, &job, "run-1");
        assert_eq!(bodies.len(), 2);
        for (body, (cards, first)) in bodies.iter().zip([(2, 0), (1, 2)]) {
            assert_eq!(body["seed"], 7);
            assert_eq!(body["cards"], cards);
            let part: Value = serde_json::from_str(body["job"].as_str().unwrap()).unwrap();
            assert_eq!(part["first_rank"], first);
            assert_eq!(part["secret_sha256"], "ab".repeat(32));
            assert_eq!(
                part["plan"],
                json!({"job": "run-1", "world_size": 3, "tp": 1, "pp": 1, "dp": 3,
                       "coordinator": "10.0.0.1:29500", "handshake_timeout_ms": HANDSHAKE_MS,
                       "op_timeout_ms": OP_MS})
            );
        }
    }

    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use crate::scripted_endpoint::{ScriptedEndpoint, SCRIPTED_KEY};
    use crate::scripted_jobs::JobScript;

    struct Flag(Arc<AtomicBool>);

    impl Cancel for Flag {
        fn cancelled(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
    }

    fn servers(scripts: Vec<JobScript>) -> (Vec<ScriptedEndpoint>, Vec<String>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let endpoints: Vec<ScriptedEndpoint> = scripts
            .into_iter()
            .map(|script| ScriptedEndpoint::start(vec![]).with_training(script))
            .collect();
        let urls = endpoints.iter().map(ScriptedEndpoint::url).collect();
        (endpoints, urls, dir)
    }

    fn job() -> ServersJob {
        ServersJob {
            cards: vec![2, 1],
            coordinator: "127.0.0.1:29500".into(),
            secret_sha256: "cd".repeat(32),
        }
    }

    fn script(store: &std::path::Path) -> JobScript {
        std::fs::create_dir_all(store).unwrap();
        JobScript::over(store.to_path_buf())
    }

    #[test]
    fn every_server_runs_its_ranks_and_rank_0s_server_answers() {
        let dir = tempfile::tempdir().unwrap();
        let (endpoints, urls, _keep) = servers(vec![
            script(&dir.path().join("a")),
            script(&dir.path().join("b")),
        ]);
        let (frames, received) = channel();
        let params = json!({"objective": "sft", "seed": 7, "resume_key": "r"});
        let report = drive(
            &urls,
            SCRIPTED_KEY,
            Duration::from_millis(10),
            &job(),
            ("train", &params),
            None,
            &frames,
        )
        .unwrap();
        drop(frames);
        assert!(report["adapter_digest"].is_string(), "{report}");
        assert!(
            received.try_iter().count() > 0,
            "rank 0's frames reach the caller"
        );
        for (endpoint, (cards, first)) in endpoints.iter().zip([(2, 0), (1, 2)]) {
            let created = endpoint.jobs().created();
            assert_eq!(created.len(), 1, "one run per server");
            assert_eq!(created[0]["cards"], cards);
            assert_eq!(created[0]["seed"], 7);
            assert!(
                created[0].get("resume_key").is_none(),
                "a job over servers does not resume"
            );
            let part: Value = serde_json::from_str(created[0]["job"].as_str().unwrap()).unwrap();
            assert_eq!(part["first_rank"], first);
            assert_eq!(part["secret_sha256"], "cd".repeat(32));
            assert_eq!(part["plan"]["world_size"], 3);
        }
    }

    #[test]
    fn a_failure_on_any_server_fails_the_run_and_stops_the_others() {
        let dir = tempfile::tempdir().unwrap();
        let hangs = JobScript {
            hang: true,
            ..script(&dir.path().join("a"))
        };
        let fails = JobScript {
            fail_with: Some("rank 2 lost its card".into()),
            ..script(&dir.path().join("b"))
        };
        let (endpoints, urls, _keep) = servers(vec![hangs, fails]);
        let (frames, _received) = channel();
        let params = json!({"objective": "sft", "seed": 7});
        let error = drive(
            &urls,
            SCRIPTED_KEY,
            Duration::from_millis(10),
            &job(),
            ("train", &params),
            None,
            &frames,
        )
        .unwrap_err();
        let JobError::Failed(why) = error else {
            panic!("{error:?}")
        };
        assert!(
            why.contains(&urls[1]) && why.contains("rank 2 lost its card"),
            "{why}"
        );
        assert!(
            endpoints[0].jobs().was_cancelled(),
            "the other server's run is stopped"
        );
    }

    #[test]
    fn a_cancel_stops_every_servers_run() {
        let dir = tempfile::tempdir().unwrap();
        let hang = |name: &str| JobScript {
            hang: true,
            ..script(&dir.path().join(name))
        };
        let (endpoints, urls, _keep) = servers(vec![hang("a"), hang("b")]);
        let (frames, _received) = channel();
        let flag = Arc::new(AtomicBool::new(false));
        let setter = Arc::clone(&flag);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(800));
            setter.store(true, Ordering::SeqCst);
        });
        let params = json!({"objective": "sft", "seed": 7});
        let error = drive(
            &urls,
            SCRIPTED_KEY,
            Duration::from_millis(10),
            &job(),
            ("train", &params),
            Some(&Flag(flag)),
            &frames,
        )
        .unwrap_err();
        assert!(matches!(error, JobError::Cancelled(Some(_))), "{error:?}");
        assert!(
            endpoints.iter().all(|e| e.jobs().was_cancelled()),
            "every server was told"
        );
    }

    #[test]
    fn a_job_without_a_seed_or_with_a_server_too_many_is_refused() {
        let (frames, _received) = channel();
        let urls = vec!["http://127.0.0.1:1/v1".to_string()];
        let unseeded = drive(
            &urls,
            "k",
            Duration::ZERO,
            &job(),
            ("train", &json!({})),
            None,
            &frames,
        );
        assert!(matches!(unseeded, Err(JobError::Failed(why)) if why.contains("seed")));
        let mismatched = drive(
            &urls,
            "k",
            Duration::ZERO,
            &job(),
            ("train", &json!({"seed": 1})),
            None,
            &frames,
        );
        assert!(matches!(mismatched, Err(JobError::Failed(why)) if why.contains("2 servers")));
    }
}
