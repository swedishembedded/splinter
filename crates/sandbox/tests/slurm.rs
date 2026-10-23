// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements scheduler-backed execution of multi-node
// jobs for its clients. If your team needs expertise in batch-system
// integration or scheduler-independent job control, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A job submitted to a batch system, observed through it and stopped by it,
//! against stand-in programs that record what the launcher asked them and
//! answer with whatever the test says the scheduler knows.

#![allow(clippy::unwrap_used)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use splinter_core::job::{JobSpec, WorkerPlan};
use splinter_sandbox::job::{JobLauncher, Launch, Observation, SlurmLauncher, SlurmState};

fn write_program(path: &Path, body: &str) {
    std::fs::write(path, body).unwrap();
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).unwrap();
}

/// Stand-in `sbatch`, `squeue` and `scancel`, and the state they read.
struct Fixture {
    _root: tempfile::TempDir,
    state: PathBuf,
    sbatch: PathBuf,
    squeue: PathBuf,
    scancel: PathBuf,
}

impl Fixture {
    fn new() -> Fixture {
        let root = tempfile::tempdir().unwrap();
        let bin = root.path().join("bin");
        let state = root.path().join("state");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        let state_text = state.display().to_string();
        write_program(
            &bin.join("sbatch"),
            &format!(
                "#!/bin/sh\n\
                 printf '%s\\n' \"$@\" > '{state_text}/sbatch.args'\n\
                 printf 'call\\n' >> '{state_text}/sbatch.calls'\n\
                 for last do :; done\n\
                 cp \"$last\" '{state_text}/script'\n\
                 cat '{state_text}/sbatch.output'\n"
            ),
        );
        write_program(
            &bin.join("squeue"),
            &format!(
                "#!/bin/sh\n\
                 printf '%s\\n' \"$@\" > '{state_text}/squeue.args'\n\
                 cat '{state_text}/squeue.state'\n"
            ),
        );
        write_program(
            &bin.join("scancel"),
            &format!(
                "#!/bin/sh\n\
                 printf '%s\\n' \"$@\" > '{state_text}/scancel.args'\n"
            ),
        );
        let fixture = Fixture {
            _root: root,
            state,
            sbatch: bin.join("sbatch"),
            squeue: bin.join("squeue"),
            scancel: bin.join("scancel"),
        };
        fixture.write("sbatch.output", "77;test\n");
        fixture.write("squeue.state", "RUNNING\n");
        fixture
    }

    fn launcher(&self) -> SlurmLauncher {
        SlurmLauncher::new(&self.sbatch, &self.squeue, &self.scancel)
    }

    fn write(&self, name: &str, body: &str) {
        std::fs::write(self.state.join(name), body).unwrap();
    }

    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(self.state.join(name)).unwrap()
    }

    fn attempt_dir(&self, name: &str) -> PathBuf {
        let dir = self.state.join("attempts").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn launch(&self, spec: &JobSpec, attempt: &str) -> Launch {
        self.launcher()
            .launch(spec, 1, &self.attempt_dir(attempt))
            .unwrap()
    }
}

fn distributed_spec() -> JobSpec {
    let mut spec = JobSpec::single(vec![
        "brain".into(),
        "train".into(),
        "--rank={rank}".into(),
        "--world={world}".into(),
    ]);
    spec.workers = WorkerPlan {
        nodes: 2,
        ranks_per_node: 2,
        gpus_per_rank: 1,
    };
    spec.wall_clock_seconds = 3_723;
    spec
}

#[test]
fn the_batch_script_carries_the_plan() {
    let fixture = Fixture::new();
    let launcher = fixture
        .launcher()
        .with_sbatch_args(vec!["--partition=debug".into()]);
    let dir = fixture.attempt_dir("script");
    let launch = launcher.launch(&distributed_spec(), 1, &dir).unwrap();
    assert_eq!(launch.world_size, 4);

    let script = fixture.read("script");
    assert!(script.contains("#SBATCH --nodes=2\n"), "{script}");
    assert!(script.contains("#SBATCH --ntasks-per-node=2\n"), "{script}");
    assert!(script.contains("#SBATCH --gpus-per-task=1\n"), "{script}");
    assert!(script.contains("#SBATCH --time=01:02:03\n"), "{script}");
    assert!(script.contains("#SBATCH --job-name=splinter-"), "{script}");
    assert!(script.contains("exec srun "), "{script}");
    // The one script gives every rank its own placeholders.
    assert!(script.contains("${SLURM_PROCID}"), "{script}");
    assert!(script.contains("${SLURM_NTASKS}"), "{script}");

    // The caller's arguments reach sbatch before the script.
    let arguments: Vec<String> = fixture
        .read("sbatch.args")
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(arguments[0], "--partition=debug", "{arguments:?}");
    assert_eq!(arguments[1], "--parsable", "{arguments:?}");
    assert_eq!(Path::new(&arguments[2]), dir.join("job.sbatch"));
}

#[test]
fn submission_records_the_scheduler_job_id() {
    let fixture = Fixture::new();
    let launcher = fixture.launcher();
    let launch = fixture.launch(&distributed_spec(), "submitted");

    assert_eq!(launcher.job_id(&launch).unwrap(), "77");
    assert_eq!(
        Launch::open(&launch.dir).unwrap().expect("a launch record"),
        launch
    );

    // Observing asks the scheduler about the recorded id.
    assert_eq!(
        launcher.observe(&launch),
        Observation::Running {
            ranks_alive: vec![0, 1, 2, 3]
        }
    );
    let arguments: Vec<String> = fixture
        .read("squeue.args")
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(arguments, ["-h", "-j", "77", "-o", "%T"]);
}

#[test]
fn every_scheduler_state_maps_to_an_observation() {
    let fixture = Fixture::new();
    let launcher = fixture.launcher();
    let spec = distributed_spec();
    let launch = fixture.launch(&spec, "states");
    let world: Vec<u32> = (0..spec.workers.world_size()).collect();
    let cases = [
        (
            "PENDING",
            SlurmState::Pending,
            Observation::Running {
                ranks_alive: vec![],
            },
        ),
        (
            "RUNNING",
            SlurmState::Running,
            Observation::Running {
                ranks_alive: world.clone(),
            },
        ),
        (
            "COMPLETING",
            SlurmState::Completing,
            Observation::Running { ranks_alive: world },
        ),
        (
            "COMPLETED",
            SlurmState::Completed,
            Observation::Exited { exit_code: Some(0) },
        ),
        (
            "FAILED",
            SlurmState::Failed,
            Observation::Exited { exit_code: None },
        ),
        (
            "CANCELLED",
            SlurmState::Cancelled,
            Observation::Exited { exit_code: None },
        ),
        (
            "TIMEOUT",
            SlurmState::Timeout,
            Observation::Exited { exit_code: None },
        ),
        (
            "PREEMPTED",
            SlurmState::Preempted,
            Observation::Exited { exit_code: None },
        ),
        (
            "NODE_FAIL",
            SlurmState::NodeFail,
            Observation::Exited { exit_code: None },
        ),
    ];
    for (text, state, observation) in cases {
        fixture.write("squeue.state", &format!("{text}\n"));
        assert!(
            matches!(launcher.state(&launch), Ok(observed) if observed == state),
            "{text}"
        );
        assert_eq!(launcher.observe(&launch), observation, "{text}");
    }
}

#[test]
fn an_unknown_or_empty_state_is_not_a_guess() {
    let fixture = Fixture::new();
    let launcher = fixture.launcher();
    let launch = fixture.launch(&distributed_spec(), "unknown");

    fixture.write("squeue.state", "SOMETHING_NEW\n");
    let error = launcher.state(&launch).unwrap_err();
    assert!(error.to_string().contains("unknown job state"), "{error}");
    assert_eq!(
        launcher.observe(&launch),
        Observation::Exited { exit_code: None }
    );

    // A job the scheduler no longer lists has left the queue; the scheduler
    // does not say how it ended.
    fixture.write("squeue.state", "");
    let error = launcher.state(&launch).unwrap_err();
    assert!(
        error.to_string().contains("no longer lists job 77"),
        "{error}"
    );
    assert_eq!(
        launcher.observe(&launch),
        Observation::Exited { exit_code: None }
    );
}

#[test]
fn stop_runs_scancel() {
    let fixture = Fixture::new();
    let launcher = fixture.launcher();
    let launch = fixture.launch(&distributed_spec(), "stopped");

    launcher.stop(&launch, None).unwrap();
    let arguments: Vec<String> = fixture
        .read("scancel.args")
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(arguments, ["77"]);

    // The scheduler cancels a job as a whole; a single-rank stop is refused
    // rather than silently widened.
    let error = launcher.stop(&launch, Some(1)).unwrap_err();
    assert!(error.to_string().contains("stop rank 1"), "{error}");
}

#[test]
fn a_failed_sbatch_is_an_error_and_the_attempt_stays_claimed() {
    let fixture = Fixture::new();
    let launcher = fixture.launcher();
    let dir = fixture.attempt_dir("failed");
    let calls = fixture.state.join("sbatch.calls");
    write_program(
        &fixture.sbatch,
        &format!(
            "#!/bin/sh\n\
             printf 'call\\n' >> '{}'\n\
             echo 'no partition available' >&2\n\
             exit 3\n",
            calls.display()
        ),
    );

    let error = launcher.launch(&distributed_spec(), 1, &dir).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("sbatch failed"), "{message}");
    assert!(message.contains("no partition available"), "{message}");

    // The attempt was claimed before sbatch ran, so a retry cannot submit it
    // a second time.
    let again = launcher.launch(&distributed_spec(), 1, &dir).unwrap_err();
    assert!(
        again.to_string().contains("already holds a launch"),
        "{again}"
    );
    assert_eq!(fixture.read("sbatch.calls").lines().count(), 1);
}

#[test]
fn a_second_submit_of_the_same_attempt_is_refused() {
    let fixture = Fixture::new();
    let launcher = fixture.launcher();
    let dir = fixture.attempt_dir("twice");
    launcher.launch(&distributed_spec(), 1, &dir).unwrap();
    let error = launcher.launch(&distributed_spec(), 1, &dir).unwrap_err();
    assert!(
        error.to_string().contains("already holds a launch"),
        "{error}"
    );
    assert_eq!(fixture.read("sbatch.calls").lines().count(), 1);
}

#[test]
fn a_missing_scheduler_program_is_an_error() {
    let fixture = Fixture::new();
    let launcher = SlurmLauncher::new(
        fixture.state.join("no-such-sbatch"),
        &fixture.squeue,
        &fixture.scancel,
    );
    let error = launcher
        .launch(&distributed_spec(), 1, &fixture.attempt_dir("missing"))
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("could not run sbatch"), "{message}");
    assert!(message.contains("no-such-sbatch"), "{message}");
}

#[test]
fn unreadable_sbatch_output_is_an_error() {
    let fixture = Fixture::new();
    fixture.write("sbatch.output", "Submitted batch job 77\n");
    let error = fixture
        .launcher()
        .launch(&distributed_spec(), 1, &fixture.attempt_dir("unreadable"))
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("a scheduler job cannot be read from"),
        "{message}"
    );
    assert!(message.contains("Submitted batch job 77"), "{message}");
}
