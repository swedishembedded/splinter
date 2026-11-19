// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn an examiner's practice
// material into graded agent experience for its clients. If your team needs
// expertise in on-policy experience collection for agent training you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the examiner's staged practice material, run under a policy brain
//! serves, comes back as admitted experience with typed outcomes.
//!
//! * The policy is named to the examiner as a profile derived from the one
//!   the material was staged under: only the main role's upstream changes,
//!   to the endpoint and id brain lists the policy under.
//! * The material is staged again under that profile and refused unless it
//!   is the material `practice.json` names, digest for digest.
//! * Each trial's outcome is read from the examiner's score record; a trial
//!   the infrastructure stopped has no reward and no experience.
//! * A rerun of a finished job runs nothing and admits nothing twice.
//! * Material that is not the train side is never run.
//!
//! The examiner and Harbor are scripted stand-ins (shell scripts): the
//! gateway announces it serves and waits to be stopped, the staging copies
//! the practice manifest, Harbor lays out the trials of a finished job, and
//! the export copies an export written by the bench fixture.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

use common::bench::{context, export, Trial};
use common::Scratch;
use serde_json::json;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::Context;
use splinter_pipelines::bench::artifact::sha256_hex;
use splinter_pipelines::bench::practice::{
    run_practice, BenchCheckout, Practice, PracticeDraw, PracticeRequest, PracticeSettings,
};
use splinter_pipelines::episodes::{EpisodeError, EpisodeOutcome};

/// The practice manifest of three staged tasks.
fn practice_manifest(set: &str) -> serde_json::Value {
    json!({
        "schema": "bench.practice/2", "collection": "tasks/support", "set": set,
        "manifest_sha256": "m1", "content_sha256": "c1", "environment": "userns",
        "bench": {"version": "0.1.0", "commit": "abc"}, "job": "job.yaml", "draw": null,
        "tasks": [
            {"name": "alpha", "digest": "da", "source_digest": "sa"},
            {"name": "beta", "digest": "db", "source_digest": "sb"},
            {"name": "gamma", "digest": "dg", "source_digest": "sg"},
        ],
    })
}

const PROFILE: &str = "[sven]\nbuild = \"${BENCH_SVEN_BUILD}\"\n[run]\nenvironment = \"userns\"\n\
attempts = 3\nsession = \"teacher\"\n[roles]\nmain = \"brain\"\n[upstreams.brain]\n\
base_url = \"http://127.0.0.1:1/v1\"\nmodel = \"teacher-27b\"\napi_key_env = \"BRAIN_KEY\"\n";

/// The practice directory the examiner staged under the teacher's profile.
fn practice_dir(root: &Path, set: &str) -> PathBuf {
    let profile = root.join("teacher.toml");
    fs::write(&profile, PROFILE).unwrap();
    let dir = root.join("practice").join("support");
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("practice.json"),
        serde_json::to_string_pretty(&practice_manifest(set)).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("bench-run.json"),
        json!({"profile": profile, "profile_sha256": sha256_hex(PROFILE.as_bytes()), "tools": {}})
            .to_string(),
    )
    .unwrap();
    dir
}

/// The stand-in examiner and Harbor in `root/bench`; every call is logged
/// to `root/calls.log`, and the profile each staging was given to
/// `root/profiles.log`.
fn checkout(root: &Path, export_fixture: &Path, staged_practice: &Path) -> BenchCheckout {
    let dir = root.join("bench");
    fs::create_dir_all(&dir).unwrap();
    let calls = root.join("calls.log");
    let bench = format!(
        r#"#!/bin/sh
echo "bench $*" >> '{calls}'
[ "$1" = "--output" ] && shift 2
case "$1" in
gateway)
    echo "key=${{BRAIN_KEY:+set}}" >> '{calls}'
    echo "bench gateway: serving brain of $4 at http://127.0.0.1:$6; state x" >&2
    exec sleep 600 ;;
task)
    while [ $# -gt 0 ]; do
        case "$1" in --profile) profile="$2"; shift ;; --out) out="$2"; shift ;; esac
        shift
    done
    cat "$profile" >> '{root}/profiles.log'
    mkdir -p "$out/support/tasks"
    cp '{staged}' "$out/support/practice.json"
    echo "jobs_dir: $out/support" > "$out/support/job.yaml" ;;
export)
    out="$4"
    [ -d "$out" ] || cp -r '{export}' "$out"
    echo '{{"ok": true, "records": [], "summary": {{"exported": 2, "skipped": [{{"trial": "gamma__3", "reason": "model_error: the endpoint answered 502"}}]}}}}' ;;
esac
"#,
        calls = calls.display(),
        root = root.display(),
        staged = staged_practice.display(),
        export = export_fixture.display(),
    );
    let harbor = format!(
        r#"#!/bin/sh
echo "harbor $*" >> '{calls}'
while [ $# -gt 0 ]; do
    case "$1" in -c) config="$2"; shift ;; --job-name) name="$2"; shift ;; esac
    shift
done
job="$(dirname "$config")/$name"
for t in alpha__1:alpha beta__2:beta gamma__3:gamma; do
    trial="${{t%%:*}}"; task="${{t##*:}}"
    mkdir -p "$job/$trial"
    echo "{{\"task\": {{\"path\": \"/x/tasks/$task\"}}}}" > "$job/$trial/config.json"
done
echo '{{"finished_at": "2026-11-18T00:00:00"}}' > "$job/result.json"
"#,
        calls = calls.display()
    );
    for (name, script) in [("bench", bench), ("harbor", harbor)] {
        let path = dir.join(name);
        fs::write(&path, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        }
    }
    BenchCheckout {
        harbor: dir.join("harbor"),
        dir,
        sven_build: None,
    }
}

/// A brain endpoint that lists `model` to every request.
fn endpoint(model: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut request = [0u8; 4096];
            let _ = stream.read(&mut request);
            let body = json!({"object": "list", "data": [{"id": model}]}).to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    format!("http://{address}/v1")
}

/// A context whose brain endpoint serves the student, and the student.
fn served(test: &str) -> (Scratch, Context, ModelRef) {
    let (scratch, ctx) = context(test);
    let mut config = ctx.config().clone();
    config.brain =
        Some(splinter_model::BrainEndpoints::new(vec![endpoint("Qwen/Qwen3-4B")], "k").unwrap());
    let ctx = Context::new(config, false).unwrap();
    let policy = ModelRef::Local {
        checkpoint: "Qwen/Qwen3-4B".into(),
        adapter: None,
        context_tokens: None,
    };
    (scratch, ctx, policy)
}

fn request<'a>(
    practice: &'a Practice,
    checkout: &'a BenchCheckout,
    policy: &'a ModelRef,
) -> PracticeRequest<'a> {
    PracticeRequest {
        practice,
        checkout,
        profile: None,
        policy,
        adapter: None,
        // seed: the stand-in examiner ignores it; any seed names a job.
        settings: PracticeSettings {
            attempts: 1,
            seed: 7,
            jobs: 2,
        },
        tasks: None,
        cancel: CancelToken::new(),
    }
}

/// The export the stand-in examiner writes: alpha passed, beta failed,
/// both graded; gamma's model server failed, so it is not exported.
fn graded_export(root: &Path) -> PathBuf {
    export(
        &root.join("fixture-export"),
        &[
            Trial::new("alpha__1", "org/alpha", Some("train"), true).scored(),
            Trial::new("beta__2", "org/beta", Some("train"), false).scored(),
        ],
    )
}

#[test]
fn material_selected_by_capability_is_staged_again_with_its_selection_and_a_draw_asks_for_it() {
    let (scratch, ctx, policy) = served("practice-capability");
    let dir = practice_dir(&scratch.0, "train");
    let mut manifest = practice_manifest("train");
    manifest["selection"] = json!({"capabilities": ["exact-arithmetic"], "offered": 5});
    for task in manifest["tasks"].as_array_mut().unwrap() {
        task["capabilities"] = json!(["apply-rules", "exact-arithmetic"]);
    }
    fs::write(dir.join("practice.json"), manifest.to_string()).unwrap();
    let checkout = checkout(
        &scratch.0,
        &graded_export(&scratch.0),
        &dir.join("practice.json"),
    );
    let practice = Practice::read(&dir).unwrap();
    assert_eq!(
        practice.selected_capabilities().unwrap(),
        ["exact-arithmetic"]
    );
    assert_eq!(
        practice.capabilities()["beta"],
        ["apply-rules", "exact-arithmetic"]
    );
    run_practice(&ctx, &request(&practice, &checkout, &policy)).unwrap();
    let calls = fs::read_to_string(scratch.0.join("calls.log")).unwrap();
    let staging = calls
        .lines()
        .find(|l| l.starts_with("bench task practice"))
        .unwrap();
    assert!(
        staging.ends_with("--capability exact-arithmetic"),
        "{staging}"
    );

    let seed = splinter_model::seed::draw().value();
    let out = scratch.0.join("transfer");
    let drawn = checkout
        .draw_practice(
            &PracticeDraw {
                collection: "tasks/support",
                environment: "userns",
                count: 40,
                seed,
                capabilities: &["design-review".to_string(), "exact-arithmetic".to_string()],
            },
            &out,
            &scratch.0.join("draw.log"),
        )
        .unwrap();
    assert_eq!(drawn, out.join("support"));
    let calls = fs::read_to_string(scratch.0.join("calls.log")).unwrap();
    let draw = calls.lines().last().unwrap();
    assert_eq!(
        draw,
        format!(
            "bench task practice tasks/support --environment userns --count 40 --seed {seed} --out {} \
             --capability design-review --capability exact-arithmetic",
            out.display()
        )
    );
}

#[test]
fn practice_runs_under_the_policy_and_comes_back_as_experience_with_typed_outcomes() {
    let (scratch, ctx, policy) = served("practice-run");
    let dir = practice_dir(&scratch.0, "train");
    let checkout = checkout(
        &scratch.0,
        &graded_export(&scratch.0),
        &dir.join("practice.json"),
    );
    let practice = Practice::read(&dir).unwrap();

    let report = run_practice(&ctx, &request(&practice, &checkout, &policy)).unwrap();

    // Only the main role's upstream became the policy, at the endpoint that
    // serves it, under the id it lists; the key reaches the gateway alone.
    let profile = fs::read_to_string(scratch.0.join("profiles.log")).unwrap();
    assert!(profile.contains("model = \"Qwen/Qwen3-4B\""), "{profile}");
    assert!(!profile.contains("teacher-27b"), "{profile}");
    assert!(
        profile.contains("build = \"${BENCH_SVEN_BUILD}\""),
        "{profile}"
    );
    let calls = fs::read_to_string(scratch.0.join("calls.log")).unwrap();
    assert!(calls.contains("key=set"), "{calls}");
    assert!(calls.contains("--n-attempts 1 --n-concurrent 2"), "{calls}");
    assert_eq!(report.served_as, "Qwen/Qwen3-4B");
    assert_eq!(report.seed, 7);
    assert_eq!(report.provenance.practice_sha256, practice.sha256);
    assert!(report.ran);

    let outcome = |task: &str| {
        report
            .trials
            .iter()
            .find(|t| t.task == task)
            .unwrap_or_else(|| panic!("no trial of {task}"))
    };
    assert_eq!(outcome("alpha").outcome, EpisodeOutcome::Passed);
    assert_eq!(outcome("alpha").reward, Some(1.0));
    assert_eq!(outcome("alpha").experiences.len(), 1);
    assert_eq!(outcome("beta").outcome, EpisodeOutcome::Failed);
    assert_eq!(outcome("beta").reward, Some(0.0));
    let gamma = outcome("gamma");
    assert!(
        matches!(&gamma.outcome, EpisodeOutcome::Infrastructure { detail } if detail.contains("502")),
        "{:?}",
        gamma.outcome
    );
    assert_eq!(
        gamma.reward, None,
        "an infrastructure failure is never a reward of 0"
    );
    assert!(gamma.experiences.is_empty());
    assert_eq!(report.ingested.admitted.get("train"), Some(&2));

    // A rerun finds the finished job: nothing runs, nothing is admitted twice.
    let again = run_practice(&ctx, &request(&practice, &checkout, &policy)).unwrap();
    assert!(!again.ran);
    assert_eq!(again.ingested.admitted.values().sum::<usize>(), 0);
    assert_eq!(again.ingested.already_held, 2);
    assert_eq!(again.trials, report.trials);
    let calls = fs::read_to_string(scratch.0.join("calls.log")).unwrap();
    assert_eq!(calls.matches("harbor run").count(), 1, "{calls}");
    assert_eq!(ctx.experiences().list().unwrap().len(), 2);
}

#[test]
fn material_the_examiner_stages_otherwise_is_refused_before_anything_runs() {
    let (scratch, ctx, policy) = served("practice-drift");
    let dir = practice_dir(&scratch.0, "train");
    let mut drifted = practice_manifest("train");
    drifted["tasks"][1]["digest"] = json!("changed");
    let staged = scratch.0.join("drifted.json");
    fs::write(&staged, drifted.to_string()).unwrap();
    let checkout = checkout(&scratch.0, &graded_export(&scratch.0), &staged);
    let practice = Practice::read(&dir).unwrap();

    let error = run_practice(&ctx, &request(&practice, &checkout, &policy)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("task beta is not staged with digest db"),
        "{error}"
    );
    let calls = fs::read_to_string(scratch.0.join("calls.log")).unwrap();
    assert!(!calls.contains("harbor"), "{calls}");
    assert!(ctx.experiences().list().unwrap().is_empty());
}

#[test]
fn material_that_is_not_the_train_side_is_never_practised_on() {
    let scratch = Scratch::new("practice-sealed");
    let dir = practice_dir(&scratch.0, "eval");
    assert!(matches!(
        Practice::read(&dir),
        Err(EpisodeError::Sealed { .. })
    ));
}

#[test]
fn a_run_of_some_tasks_names_them_to_harbor_over_the_staged_tasks_and_is_its_own_job() {
    let (scratch, ctx, policy) = served("practice-subset");
    let dir = practice_dir(&scratch.0, "train");
    let checkout = checkout(
        &scratch.0,
        &graded_export(&scratch.0),
        &dir.join("practice.json"),
    );
    let practice = Practice::read(&dir).unwrap();
    let all = run_practice(&ctx, &request(&practice, &checkout, &policy)).unwrap();
    let chosen = ["alpha".to_string()];
    let some = run_practice(
        &ctx,
        &PracticeRequest {
            tasks: Some(&chosen),
            ..request(&practice, &checkout, &policy)
        },
    )
    .unwrap();
    assert_ne!(all.job, some.job);
    let calls = fs::read_to_string(scratch.0.join("calls.log")).unwrap();
    assert!(
        calls.contains("/staged/support/tasks --include-task-name alpha"),
        "{calls}"
    );
    let unknown = ["omega".to_string()];
    let refused = run_practice(
        &ctx,
        &PracticeRequest {
            tasks: Some(&unknown),
            ..request(&practice, &checkout, &policy)
        },
    )
    .unwrap_err();
    assert!(
        refused.to_string().contains("omega is not a task"),
        "{refused}"
    );
}

#[test]
fn as_a_campaigns_executor_each_trial_is_an_episode_and_only_a_graded_one_has_a_reward() {
    use splinter_pipelines::bench::practice::BenchPractice;
    use splinter_pipelines::campaign::endpoint::EpisodeExecutor;

    let (scratch, ctx, policy) = served("practice-executor");
    let dir = practice_dir(&scratch.0, "train");
    let checkout = checkout(
        &scratch.0,
        &graded_export(&scratch.0),
        &dir.join("practice.json"),
    );
    let mut executor = BenchPractice::new(&ctx, Practice::read(&dir).unwrap(), checkout, policy, 2);
    let tasks = executor.tasks();
    assert_eq!(tasks, ["alpha", "beta", "gamma"]);
    // seed: the stand-in examiner ignores it; any seed names a job.
    let episodes = executor.run(&tasks, 1, 3).unwrap();
    let rewards: Vec<(&str, Option<f64>, bool)> = episodes
        .iter()
        .map(|e| (e.task.as_str(), e.reward, e.passed))
        .collect();
    assert_eq!(
        rewards,
        [
            ("alpha", Some(1.0), true),
            ("beta", Some(0.0), false),
            ("gamma", None, false)
        ]
    );
    assert!(episodes[0].prompt[0]
        .content
        .contains("Repair the firmware"));
}

#[test]
fn as_a_campaigns_rollouts_the_tokens_are_summed_from_each_trials_usage_so_a_token_budget_binds() {
    use splinter_pipelines::campaign::practice::{PracticeRollouts, PracticeSpec};
    use splinter_pipelines::campaign::{Model, RolloutRequest, Rollouts};

    let (scratch, ctx, policy) = served("practice-rollout-tokens");
    let dir = practice_dir(&scratch.0, "train");
    let checkout = checkout(
        &scratch.0,
        &graded_export(&scratch.0),
        &dir.join("practice.json"),
    );
    let mut rollouts = PracticeRollouts::new(&PracticeSpec {
        dir: dir.clone(),
        bench: checkout.dir.clone(),
        harbor: Some(checkout.harbor.clone()),
        sven_build: checkout.sven_build.clone(),
        profile: None,
        jobs: 2,
    })
    .unwrap();
    let champion = Model {
        generation: 0,
        model: policy.to_string(),
        candidate: None,
        adapter: None,
    };
    let stage = scratch.0.join("rollout");
    fs::create_dir_all(&stage).unwrap();
    let tasks: Vec<String> = ["alpha", "beta", "gamma"].map(String::from).to_vec();
    let rolled = rollouts
        .rollout(
            &ctx,
            &RolloutRequest {
                generation: 1,
                champion: &champion,
                tasks: &tasks,
                attempts: 1,
                // seed: the scripted examiner ignores it; any seed names a job.
                seed: 3,
                transfer: None,
                dir: &stage,
            },
            &CancelToken::new(),
        )
        .unwrap();
    // alpha and beta were graded, each 10 prompt and 5 completion tokens by
    // its score record; gamma never reached the model and spent nothing
    // counted.
    assert_eq!(rolled.tokens, Some(30));
    assert_eq!(rolled.trials.len(), 3);
}
