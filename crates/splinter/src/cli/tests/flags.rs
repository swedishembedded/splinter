// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for the flags that reach the models: the remote
//! request width, the served temperature, the solve deadline and the training
//! row's bound, the tier the frozen base is held at, the seed a dataset's
//! held-out split is drawn by, and the settings of preference and feedback
//! datasets and runs (the pairing bound and seed, the pair loss and its
//! margin, KTO's class weights), and the curation of passes per task.

use super::{command, parse, Command};

#[test]
fn the_remote_knobs_are_global_flags_and_bounded() {
    for args in [
        &[
            "--remote-concurrency",
            "8",
            "--temperature",
            "0.7",
            "status",
        ][..],
        &[
            "status",
            "--remote-concurrency",
            "8",
            "--temperature",
            "0.7",
        ][..],
    ] {
        let cli = parse(args).unwrap_or_else(|e| panic!("{args:?}: {e}"));
        assert_eq!(cli.global.remote_concurrency, Some(8));
        assert_eq!(cli.global.temperature, Some(0.7));
    }
    let none = parse(&["status"]).unwrap();
    assert_eq!(
        (none.global.remote_concurrency, none.global.temperature),
        (None, None)
    );
    assert!(parse(&["--remote-concurrency", "0", "status"]).is_err());
    assert!(parse(&["--temperature", "2.5", "status"]).is_err());
    assert!(parse(&["--temperature", "-0.1", "status"]).is_err());
}

#[test]
fn the_job_launcher_is_a_global_flag_of_the_configured_kinds() {
    use splinter_sdk::config::LauncherKind;

    let cli = parse(&["--job-launcher", "slurm", "status"]).unwrap();
    assert_eq!(cli.global.job_launcher, Some(LauncherKind::Slurm));
    assert_eq!(parse(&["status"]).unwrap().global.job_launcher, None);
    assert!(parse(&["--job-launcher", "kubernetes", "status"]).is_err());

    let cli = parse(&[
        "--slurm-sbatch",
        "/sl/sbatch",
        "--slurm-squeue",
        "/sl/squeue",
        "--slurm-scancel",
        "/sl/scancel",
        "--slurm-arg",
        "--partition=debug",
        "--slurm-arg",
        "--account=abc",
        "status",
    ])
    .unwrap();
    assert_eq!(cli.global.slurm_sbatch, Some("/sl/sbatch".into()));
    assert_eq!(cli.global.slurm_squeue, Some("/sl/squeue".into()));
    assert_eq!(cli.global.slurm_scancel, Some("/sl/scancel".into()));
    assert_eq!(cli.global.slurm_arg, ["--partition=debug", "--account=abc"]);
}

#[test]
fn the_cluster_is_used_automatically_and_its_flags_are_global() {
    use splinter_sdk::cluster::ClusterMode;

    let plain = parse(&["status"]).unwrap();
    assert_eq!(plain.global.device.cluster, None, "auto unless told");
    let cli = parse(&[
        "doctor",
        "--cluster",
        "auto",
        "--cluster-nodes",
        "n[442,445]",
        "--cluster-exec-prefix",
        "apptainer exec --nv /i.sif",
        "--slurm-srun",
        "/sl/srun",
        "--slurm-step-arg",
        "--mem=64G",
        "--slurm-single-node-step-arg",
        "--network=single_node_vni",
    ])
    .unwrap();
    let device = &cli.global.device;
    assert_eq!(device.cluster, Some(ClusterMode::Auto));
    assert_eq!(device.cluster_nodes.as_deref(), Some("n[442,445]"));
    assert_eq!(device.slurm_srun, Some("/sl/srun".into()));
    assert_eq!(device.slurm_step_arg, ["--mem=64G"]);
    assert_eq!(
        device.slurm_single_node_step_arg,
        ["--network=single_node_vni"]
    );
    assert!(parse(&["--cluster", "on", "status"]).is_err());
}

#[test]
fn the_solve_deadline_is_a_global_flag_of_whole_seconds() {
    for args in [
        &["--solve-deadline", "1800", "status"][..],
        &["status", "--solve-deadline", "1800"][..],
    ] {
        let cli = parse(args).unwrap_or_else(|e| panic!("{args:?}: {e}"));
        assert_eq!(cli.global.solve_deadline, Some(1800));
    }
    assert_eq!(parse(&["status"]).unwrap().global.solve_deadline, None);
    assert!(parse(&["--solve-deadline", "0", "status"]).is_err());
    assert!(parse(&["--solve-deadline", "1.5", "status"]).is_err());
}

#[test]
fn baseline_takes_the_binary_models_and_seeds_as_given_and_no_default_seed() {
    let Command::Baseline(args) = command(&[
        "baseline",
        "--brain-bin",
        "/b/brain",
        "--model",
        "/m/a",
        "--model",
        "/m/b",
        "--exam-seed",
        "20261017",
        "--require-clean",
    ]) else {
        panic!("not baseline");
    };
    assert_eq!(args.models.len(), 2);
    assert_eq!(
        (args.exam_seed, args.training_seed),
        (Some(20_261_017), None)
    );
    assert!(args.require_clean);
}
#[test]
fn the_sampling_of_a_served_model_is_global_flags_and_bounded() {
    let cli = parse(&[
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--sampling-seed",
        "16022413115216019911",
        "--thinking",
        "--thinking-budget",
        "1024",
        "status",
    ])
    .unwrap();
    assert_eq!(cli.global.thinking_budget, Some(1024));
    // A budget bounds thinking, so it needs it; and it is positive.
    assert!(parse(&["--thinking-budget", "1024", "status"]).is_err());
    assert!(parse(&["--thinking", "--thinking-budget", "0", "status"]).is_err());
    assert_eq!(
        (cli.global.top_p, cli.global.top_k, cli.global.sampling_seed),
        (Some(0.95), Some(20), Some(16_022_413_115_216_019_911))
    );
    assert!(cli.global.thinking);
    let none = parse(&["status"]).unwrap();
    assert_eq!(
        (
            none.global.top_p,
            none.global.top_k,
            none.global.sampling_seed
        ),
        (None, None, None)
    );
    assert!(!none.global.thinking);
    for bad in [["--top-p", "0"], ["--top-p", "1.5"], ["--top-k", "0"]] {
        assert!(parse(&[bad[0], bad[1], "status"]).is_err(), "{bad:?}");
    }
}

#[test]
fn a_project_variable_is_name_equals_value_and_not_one_the_sandbox_sets() {
    let cli = parse(&[
        "--workspace-env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--workspace-env",
        "A_B=x=y",
        "status",
    ])
    .unwrap();
    assert_eq!(
        cli.global.workspace_env,
        [
            ("PYTHONDONTWRITEBYTECODE".to_string(), "1".to_string()),
            ("A_B".to_string(), "x=y".to_string())
        ]
    );
    for bad in ["NOVALUE", "=1", "1A=1", "A-B=1", "HOME=/x", "TMPDIR=/x"] {
        assert!(parse(&["--workspace-env", bad, "status"]).is_err(), "{bad}");
    }
}

#[test]
fn the_coding_prompt_is_a_global_flag_naming_a_registered_version() {
    use splinter_sdk::vocabulary::prompt::CodingPrompt;
    let chosen = parse(&["--coding-prompt", "coding-agent-2", "status"]).unwrap();
    assert_eq!(chosen.global.coding_prompt, Some(CodingPrompt::V2));
    // Not given: the registered protocol's prompt stays the default.
    assert_eq!(parse(&["status"]).unwrap().global.coding_prompt, None);
    assert!(parse(&["--coding-prompt", "coding-agent-9", "status"]).is_err());
}

#[test]
fn brain_endpoints_are_global_flags_and_repeat() {
    let cli = parse(&[
        "--brain-endpoint",
        "http://127.0.0.1:8805/v1",
        "--brain-endpoint",
        "http://127.0.0.1:8806/v1",
        "--brain-key-file",
        "/keys/brain",
        "--brain-adapter-manifest",
        "/run/brain/release.json",
        "--attempt-concurrency",
        "48",
        "--brain-artifact-store",
        "/shared/artifacts",
        "status",
    ])
    .unwrap();
    assert_eq!(
        cli.global.device.brain_endpoint,
        ["http://127.0.0.1:8805/v1", "http://127.0.0.1:8806/v1"]
    );
    assert_eq!(
        cli.global.device.brain_key_file.as_deref(),
        Some(std::path::Path::new("/keys/brain"))
    );
    assert_eq!(cli.global.attempt_concurrency, Some(48));
    assert_eq!(
        cli.global.brain_artifact_store.as_deref(),
        Some(std::path::Path::new("/shared/artifacts"))
    );
    let none = parse(&["status"]).unwrap();
    assert!(
        none.global.device.brain_endpoint.is_empty() && none.global.attempt_concurrency.is_none()
    );
    assert!(parse(&["--attempt-concurrency", "0", "status"]).is_err());
}
