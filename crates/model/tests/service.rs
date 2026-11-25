// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The server a run owns: started with the store, the adapter directory and
//! the training workload, reachable through endpoints that offer adapters by
//! writing them into the directory it serves, and gone when dropped.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use splinter_model::service::{ManagedBrain, ServiceSettings};
use sven_sdk::CancelToken;

/// A stand-in `brain`: records its arguments, writes the key file and the
/// ready marker the way `brain serve` does, then waits to be stopped.
fn stand_in(dir: &Path) -> std::path::PathBuf {
    let script = dir.join("brain");
    std::fs::write(
        &script,
        r#"#!/bin/sh
echo "$@" > "$0.args"
while [ $# -gt 0 ]; do
  case "$1" in
    --api-keys-out) printf '{"openai":"sk-stand-in"}' > "$2"; shift ;;
    --ready-file) : > "$2"; shift ;;
  esac
  shift
done
exec sleep 600
"#,
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    script
}

fn settings(dir: &Path, binary: std::path::PathBuf) -> ServiceSettings {
    ServiceSettings {
        binary,
        models: dir.join("models"),
        state: dir.join("state"),
        device: Some("gpu1".into()),
        backend: None,
        max_batch: Some(24),
        serve_adapters: true,
        adapter_base: None,
    }
}

#[test]
fn the_server_a_run_owns_serves_the_store_the_adapters_and_training_and_is_asked_as_wide_as_it_decodes(
) {
    let dir = tempfile::tempdir().unwrap();
    let binary = stand_in(dir.path());

    let base = dir.path().join("models/Qwen/Qwen3-4B");
    let brain = ManagedBrain::start(
        &ServiceSettings {
            adapter_base: Some(base.clone()),
            ..settings(dir.path(), binary.clone())
        },
        &CancelToken::default(),
    )
    .unwrap();

    let args = std::fs::read_to_string(dir.path().join("brain.args")).unwrap();
    // The adapters sit on the base the run trains, on every layer it trains.
    let checkpoint = format!("--checkpoint {}", base.display());
    for wanted in [
        checkpoint.as_str(),
        "--adapter-targets all",
        "--device gpu1",
        "serve --openai 127.0.0.1:",
        "--max-batch 24",
        "--models-dir",
        "--serve-adapters",
        "--training",
        "--artifact-store",
    ] {
        assert!(args.contains(wanted), "{wanted} in {args}");
    }
    assert_eq!(brain.endpoints().key(), "sk-stand-in");
    assert_eq!(brain.endpoints().attempt_concurrency(), 24);
    assert!(brain.endpoints().artifact_store().is_ok());
    assert!(brain.endpoints().urls()[0].starts_with("http://127.0.0.1:"));
}

#[test]
fn dropping_the_server_removes_its_scratch_directory_and_keys() {
    let dir = tempfile::tempdir().unwrap();
    let binary = stand_in(dir.path());
    let brain =
        ManagedBrain::start(&settings(dir.path(), binary), &CancelToken::default()).unwrap();
    let run = dir
        .path()
        .join("state")
        .join(format!("run-{}", std::process::id()));
    assert!(run.join("keys.json").is_file());

    drop(brain);

    assert!(!run.exists(), "the keys do not outlive the server");
    assert!(
        dir.path().join("state/adapters").is_dir(),
        "the adapters stay"
    );
}

#[test]
fn a_binary_that_cannot_start_is_refused_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("no-such-brain");

    let refused = ManagedBrain::start(
        &settings(dir.path(), missing.clone()),
        &CancelToken::default(),
    )
    .err()
    .expect("refused")
    .to_string();

    assert!(refused.contains("no-such-brain"), "{refused}");
    assert!(!dir
        .path()
        .join("state")
        .join(format!("run-{}", std::process::id()))
        .exists());
}

mod fleet {
    use super::*;
    use splinter_model::service::FleetSettings;
    use splinter_sandbox::step::NodeSteps;

    /// A stand-in `srun`: records its arguments and runs the step's script
    /// here once for every node of `--nodelist`, as the task of that node
    /// (`SLURMD_NODENAME`, `SLURM_PROCID` in list order) of a step whose id
    /// is its pid, each task in a process group of its own; `SIGTERM` ends
    /// every task, as the scheduler forwards it.
    fn srun(dir: &Path) -> std::path::PathBuf {
        let script = dir.join("srun");
        std::fs::write(
            &script,
            r#"#!/bin/sh
echo "$@" >> "$0.args"
for word in "$@"; do case "$word" in --nodelist=*) nodes=${word#--nodelist=} ;; esac; done
while [ "$1" != sh ]; do shift; done
shift
groups=
trap 'for g in $groups; do kill -TERM -$g 2>/dev/null; done; exit 143' TERM
rank=0
for node in $(echo "$nodes" | tr , ' '); do
  SLURM_JOB_ID=42 SLURM_STEP_ID=$$ SLURM_PROCID=$rank SLURMD_NODENAME=$node setsid sh "$@" &
  groups="$groups $!"
  rank=$((rank + 1))
done
wait
"#,
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script
    }

    /// A stand-in `brain serve` on a node: checks it was given the fleet's key
    /// file, publishes a connection record with the port its system "picked"
    /// and the ready marker, then waits to be stopped. The one on the node
    /// `FAIL_ON` names exits first.
    fn node_brain(dir: &Path) -> std::path::PathBuf {
        let script = dir.join("brain");
        std::fs::write(
            &script,
            r#"#!/bin/sh
all="$*"
while [ $# -gt 0 ]; do
  case "$1" in
    --openai) addr=$2; shift ;;
    --runtime-dir) runtime=$2; shift ;;
    --ready-file) ready=$2; shift ;;
    --api-key-file) key=$2; shift ;;
    --tls-cert) cert=$2; shift ;;
  esac
  shift
done
echo "$all" > "$runtime/args"
echo "$VIA_PREFIX" > "$runtime/prefix"
echo $$ > "$runtime/pid"
echo $$ >> "$0.pids"
if [ -n "$FAIL_ON" ] && [ "${runtime##*/}" = "$FAIL_ON" ]; then echo "no card" >&2; exit 3; fi
test -s "$key" || exit 4
host=${addr%:0}
scheme=http
if [ -n "$cert" ]; then grep -q 'BEGIN CERTIFICATE' "$cert" || exit 5; scheme=https; fi
printf '{"pid":%s,"surfaces":[{"provider":"openai","url":"%s://%s:%s"}]}' $$ "$scheme" "$host" $((40000 + $$ % 1000)) > "$runtime/serve-$$.json"
: > "$ready"
exec sleep 600
"#,
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script
    }

    fn fleet(dir: &Path, nodes: &[&str]) -> FleetSettings {
        FleetSettings {
            service: settings(dir, node_brain(dir)),
            steps: NodeSteps {
                srun: srun(dir),
                scancel: None,
                args: vec!["--mem=8G".into()],
                single_node_args: vec!["--network=single_node_vni".into()],
                exec_prefix: vec!["env".into(), "VIA_PREFIX=yes".into()],
            },
            nodes: nodes.iter().map(|n| (*n).to_string()).collect(),
            gpus_per_node: 4,
            node_cards: std::collections::BTreeMap::new(),
            insecure_plaintext: false,
            stage_models: None,
        }
    }

    fn alive(pid_file: &Path) -> bool {
        let pid: i32 = std::fs::read_to_string(pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // SAFETY: signal 0 only asks whether the process exists.
        unsafe { libc::kill(pid, 0) == 0 }
    }

    #[test]
    fn a_fleet_serves_every_card_of_every_node_behind_one_key_and_leaves_nothing_running() {
        let dir = tempfile::tempdir().unwrap();
        let mut settings = fleet(dir.path(), &["localhost", "127.0.0.1"]);
        settings.service.device = None;
        let brain = ManagedBrain::start_fleet(&settings, &CancelToken::default()).unwrap();

        let endpoints = brain.endpoints();
        assert_eq!(endpoints.urls().len(), 2, "{:?}", endpoints.urls());
        for url in endpoints.urls() {
            assert!(
                url.starts_with("https://127.0.0.1:4"),
                "TLS by default: {url}"
            );
        }
        assert_eq!(endpoints.attempt_concurrency(), 48, "24 slots on each node");
        assert!(endpoints.artifact_store().is_ok());

        let work = dir
            .path()
            .join(format!("state/fleet-{}", std::process::id()));
        let listed = std::fs::read_to_string(work.join("endpoints")).unwrap();
        assert_eq!(listed.lines().collect::<Vec<_>>(), endpoints.urls());
        assert_eq!(brain.work_dir(), work);
        let authority = std::fs::read_to_string(work.join("ca.pem")).unwrap();
        assert!(
            authority.contains("BEGIN CERTIFICATE"),
            "clients outside find the CA"
        );
        let key = std::fs::read_to_string(work.join("key")).unwrap();
        assert_eq!(
            endpoints.key(),
            key.trim(),
            "every server answers to the fleet's key"
        );
        let mode = std::fs::metadata(work.join("key"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);

        let srun_args = std::fs::read_to_string(dir.path().join("srun.args")).unwrap();
        assert_eq!(
            srun_args.lines().count(),
            1,
            "one step over every node, so their tasks share the fabric: {srun_args}"
        );
        for wanted in [
            "--overlap",
            "--nodelist=localhost,127.0.0.1",
            "--nodes=2",
            "--ntasks-per-node=1",
            "--gpus-per-node=4",
            "--mem=8G",
        ] {
            assert!(srun_args.contains(wanted), "{wanted} in {srun_args}");
        }
        assert!(
            !srun_args.contains("--network=single_node_vni"),
            "a step over several nodes has its fabric access: {srun_args}"
        );
        for node in ["localhost", "127.0.0.1"] {
            let args = std::fs::read_to_string(work.join(node).join("args")).unwrap();
            for wanted in [
                "--device gpu0,gpu1,gpu2,gpu3",
                "serve --openai 127.0.0.1:0",
                "--api-key-file",
                "--tls-cert",
                "--tls-key",
                "--serve-adapters",
                "--artifact-store",
            ] {
                assert!(args.contains(wanted), "{node}: {wanted} in {args}");
            }
            assert!(!args.contains("--insecure-plaintext"), "{args}");
            for private in ["tls.key", "tls.crt"] {
                let mode = std::fs::metadata(work.join(node).join(private))
                    .unwrap()
                    .permissions()
                    .mode();
                assert_eq!(mode & 0o777, 0o600, "{private}");
            }
            assert!(args.contains("--train-job-secret"), "{args}");
            assert_eq!(
                args.contains("--train-job-coordinator"),
                node == "localhost",
                "only the first server coordinates: {args}"
            );
            let mode = std::fs::metadata(work.join(node))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o700, "brain publishes its record only there");
            let prefix = std::fs::read_to_string(work.join(node).join("prefix")).unwrap();
            assert_eq!(prefix.trim(), "yes", "started under the exec prefix");
            assert!(alive(&work.join(node).join("pid")));
        }
        let pids: Vec<_> = ["localhost", "127.0.0.1"]
            .iter()
            .map(|node| {
                let pid = work.join(node).join("pid");
                let copy = dir.path().join(format!("{node}.pid"));
                std::fs::copy(&pid, &copy).unwrap();
                copy
            })
            .collect();

        drop(brain);

        for pid in &pids {
            assert!(
                !alive(pid),
                "no server outlives the fleet: {}",
                pid.display()
            );
        }
        assert!(!work.exists(), "the key does not outlive the fleet");
    }

    #[test]
    fn a_node_restricted_to_named_cards_serves_only_those_and_the_others_every_card() {
        let dir = tempfile::tempdir().unwrap();
        let mut settings = fleet(dir.path(), &["localhost", "127.0.0.1"]);
        settings.service.device = None;
        settings.node_cards.insert("localhost".into(), vec![0, 1]);
        assert_eq!(
            settings.server_cards(),
            [2, 4],
            "the cards each server trains on"
        );
        let brain = ManagedBrain::start_fleet(&settings, &CancelToken::default()).unwrap();
        let args =
            |node: &str| std::fs::read_to_string(brain.work_dir().join(node).join("args")).unwrap();
        assert!(
            args("localhost").contains("--device gpu0,gpu1 "),
            "{}",
            args("localhost")
        );
        assert!(
            args("127.0.0.1").contains("--device gpu0,gpu1,gpu2,gpu3 "),
            "{}",
            args("127.0.0.1")
        );
        let srun_args = std::fs::read_to_string(dir.path().join("srun.args")).unwrap();
        assert!(
            srun_args.contains("--gpus-per-node=4"),
            "the step takes the most any node serves: {srun_args}"
        );
        drop(brain);

        // A device list given for every server counts as its cards.
        let mut listed = fleet(dir.path(), &["localhost", "127.0.0.1"]);
        listed.service.device = Some("gpu1,gpu3".into());
        assert_eq!(listed.server_cards(), [2, 2]);
        listed.node_cards.insert("127.0.0.1".into(), vec![2]);
        assert_eq!(listed.server_cards(), [2, 1], "a node's own cards win");

        // A card the node does not have, a node the fleet does not use, none at all.
        for (node, cards) in [
            ("localhost", vec![4]),
            ("elsewhere", vec![0]),
            ("localhost", vec![]),
        ] {
            let mut wrong = fleet(dir.path(), &["localhost", "127.0.0.1"]);
            wrong.node_cards.insert(node.into(), cards.clone());
            let refused = ManagedBrain::start_fleet(&wrong, &CancelToken::default())
                .err()
                .unwrap_or_else(|| panic!("{node} {cards:?} is refused"))
                .to_string();
            assert!(refused.contains(node), "{refused}");
        }
    }

    #[test]
    fn plaintext_between_nodes_is_served_only_when_asked_for_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let settings = FleetSettings {
            insecure_plaintext: true,
            ..fleet(dir.path(), &["localhost"])
        };
        let brain = ManagedBrain::start_fleet(&settings, &CancelToken::default()).unwrap();
        assert!(brain.endpoints().urls()[0].starts_with("http://127.0.0.1:"));
        let work = brain.work_dir().to_path_buf();
        let args = std::fs::read_to_string(work.join("localhost/args")).unwrap();
        assert!(args.contains("--insecure-plaintext") && !args.contains("--tls-cert"));
        assert!(!work.join("ca.pem").exists());
    }

    #[test]
    fn a_fleet_given_a_node_local_directory_stages_the_store_there() {
        let dir = tempfile::tempdir().unwrap();
        let settings = FleetSettings {
            stage_models: Some("/scratch/local/models".into()),
            ..fleet(dir.path(), &["localhost"])
        };
        let brain = ManagedBrain::start_fleet(&settings, &CancelToken::default()).unwrap();
        let args = std::fs::read_to_string(brain.work_dir().join("localhost/args")).unwrap();
        let store = dir.path().join("models");
        let wanted = format!(
            "--stage-models /scratch/local/models --stage-from {} ",
            store.display()
        );
        assert!(args.starts_with(&wanted), "global flags first: {args}");
        drop(brain);

        let plain =
            ManagedBrain::start_fleet(&fleet(dir.path(), &["localhost"]), &CancelToken::default())
                .unwrap();
        let args = std::fs::read_to_string(plain.work_dir().join("localhost/args")).unwrap();
        assert!(
            !args.contains("--stage-models"),
            "none unless asked: {args}"
        );
    }

    #[test]
    fn a_node_whose_server_does_not_start_is_named_and_the_others_are_stopped() {
        let dir = tempfile::tempdir().unwrap();
        // `fail.invalid.` resolves nowhere: RFC 6761 reserves `.invalid`, and the
        // final dot keeps a resolver's search domains (whose wildcard records
        // can answer any name) from being tried. The refusal names it before
        // any step starts.
        let settings = fleet(dir.path(), &["localhost", "fail.invalid."]);
        let refused = ManagedBrain::start_fleet(&settings, &CancelToken::default())
            .err()
            .expect("refused")
            .to_string();
        assert!(refused.contains("fail.invalid."), "{refused}");

        // A node that resolves but whose server exits before it is ready,
        // after the other node's server started.
        let mut failing = fleet(dir.path(), &["localhost", "127.0.0.1"]);
        failing.steps.exec_prefix = vec!["env".into(), "FAIL_ON=127.0.0.1".into()];
        let work = dir
            .path()
            .join(format!("state/fleet-{}", std::process::id()));
        let refused = ManagedBrain::start_fleet(&failing, &CancelToken::default())
            .err()
            .expect("refused")
            .to_string();
        assert!(
            refused.contains("127.0.0.1") && refused.contains("no card"),
            "{refused}"
        );
        let pids = std::fs::read_to_string(dir.path().join("brain.pids")).unwrap();
        assert_eq!(pids.lines().count(), 2, "both servers were started: {pids}");
        for pid in pids.lines() {
            // SAFETY: signal 0 only asks whether the process exists.
            let gone = unsafe { libc::kill(pid.parse().unwrap(), 0) } != 0;
            assert!(gone, "the server on the node that started is stopped too");
        }
        assert!(!work.exists(), "a fleet that did not start leaves no key");
    }
}
