// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements detached multi-process jobs for its clients.
// If your team needs expertise in process launch and failure simulation, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A job launched, observed and stopped across process boundaries, including
//! the failure a scheduler causes: one worker killed while the others run, and
//! a controller that restarts and reopens the launch instead of launching it
//! again.

use std::ffi::OsString;
use std::path::Path;
use std::time::{Duration, Instant};

use splinter_core::job::{JobSpec, WorkerPlan};
use splinter_sandbox::job::{
    supervise, JobLauncher, Launch, LocalLauncher, Observation, SUPERVISOR_MARKER, TIMEOUT_EXIT,
};

fn two_rank_spec() -> JobSpec {
    let mut spec = JobSpec::single(vec![
        "/bin/sh".into(),
        "-c".into(),
        "echo {rank} of {world} > rank-{rank}.txt; sleep 30".into(),
    ]);
    spec.workers = WorkerPlan {
        nodes: 1,
        ranks_per_node: 2,
        gpus_per_rank: 0,
    };
    spec
}

fn wait_for(path: &Path, limit: Duration) {
    let deadline = Instant::now() + limit;
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "{} never appeared",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn wait_until<F: Fn() -> bool>(what: &str, limit: Duration, condition: F) {
    let deadline = Instant::now() + limit;
    while !condition() {
        assert!(Instant::now() < deadline, "{what} never happened");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Stops the job however the test ends.
struct Guard(Launch);

impl Drop for Guard {
    fn drop(&mut self) {
        self.0.stop(None).ok();
    }
}

#[test]
fn a_two_rank_job_runs_as_separate_processes() {
    let dir = tempfile::tempdir().unwrap();
    let launcher = LocalLauncher::default();
    let launch = launcher
        .launch(&two_rank_spec(), 1, dir.path())
        .expect("launch");
    assert_eq!(launch.world_size, 2);
    assert_eq!(launch.pids.len(), 2);
    assert_ne!(
        launch.pids[0], launch.pids[1],
        "ranks must be separate processes"
    );
    let guard = Guard(launch.clone());

    // Both workers filled their own placeholder and said what world they were.
    for rank in 0..2 {
        let path = dir.path().join(format!("rank-{rank}.txt"));
        wait_for(&path, Duration::from_secs(10));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap().trim(),
            format!("{rank} of 2")
        );
    }
    assert_eq!(
        launcher.observe(&launch),
        Observation::Running {
            ranks_alive: vec![0, 1]
        }
    );
    guard.0.stop(None).unwrap();
    wait_until("both ranks exit", Duration::from_secs(10), || {
        matches!(launcher.observe(&launch), Observation::Exited { .. })
    });
}

#[test]
fn a_second_launch_of_the_same_attempt_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let launcher = LocalLauncher::default();
    let first = launcher
        .launch(&two_rank_spec(), 1, dir.path())
        .expect("first launch");
    let guard = Guard(first.clone());
    let error = launcher
        .launch(&two_rank_spec(), 1, dir.path())
        .expect_err("the attempt was already launched");
    assert!(
        error.to_string().contains("already holds a launch"),
        "{error}"
    );
    // The refused submission did not start another set of workers.
    assert_eq!(first.pids.len(), 2);
    guard.0.stop(None).unwrap();
}

#[test]
fn one_worker_can_die_while_the_others_keep_running() {
    let dir = tempfile::tempdir().unwrap();
    let launcher = LocalLauncher::default();
    let launch = launcher
        .launch(&two_rank_spec(), 1, dir.path())
        .expect("launch");
    let guard = Guard(launch.clone());
    wait_until("both ranks are running", Duration::from_secs(10), || {
        launch.ranks_alive() == vec![0, 1]
    });
    launch.stop(Some(1)).unwrap();
    wait_until("rank 1 is gone", Duration::from_secs(10), || {
        launch.ranks_alive() == vec![0]
    });
    assert_eq!(
        launcher.observe(&launch),
        Observation::Running {
            ranks_alive: vec![0]
        }
    );
    guard.0.stop(None).unwrap();
}

#[test]
fn a_supervisor_runs_the_worker_and_records_its_exit() {
    let dir = tempfile::tempdir().unwrap();
    let args = [
        OsString::from(SUPERVISOR_MARKER),
        dir.path().as_os_str().to_owned(),
        OsString::from("0"),
        OsString::from("0"),
        OsString::from("/bin/sh"),
        OsString::from("-c"),
        OsString::from("exit 7"),
    ];
    assert_eq!(supervise(args.into_iter()), Some(7));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("rank-0.exit"))
            .unwrap()
            .trim(),
        "7"
    );
    assert_eq!(
        supervise([OsString::from("something-else")].into_iter()),
        None,
        "anything but the marker is not a supervisor invocation"
    );
}

#[test]
fn a_worker_that_outlives_its_wall_clock_is_stopped_and_recorded_as_timed_out() {
    let dir = tempfile::tempdir().unwrap();
    let started = Instant::now();
    let args = [
        OsString::from(SUPERVISOR_MARKER),
        dir.path().as_os_str().to_owned(),
        OsString::from("0"),
        OsString::from("1"),
        OsString::from("/bin/sh"),
        OsString::from("-c"),
        OsString::from("sleep 30"),
    ];
    assert_eq!(supervise(args.into_iter()), Some(TIMEOUT_EXIT));
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the bound was enforced, not waited out"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("rank-0.exit"))
            .unwrap()
            .trim(),
        TIMEOUT_EXIT.to_string()
    );
}

/// A stand-in supervisor that runs the worker and records its exit the way the
/// real one does, so the launcher's spawn and its reading are tested without
/// the `splinter` binary.
fn scripted_supervisor(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("supervisor.sh");
    assert!(std::fs::write(
        &path,
        "#!/bin/sh\n\
         dir=$2\n\
         rank=$3\n\
         shift 4\n\
         \"$@\"\n\
         code=$?\n\
         printf '%s\\n' \"$code\" > \"$dir/rank-$rank.exit\"\n\
         exit \"$code\"\n",
    )
    .is_ok());
    use std::os::unix::fs::PermissionsExt;
    assert!(std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).is_ok());
    path
}

#[test]
fn a_supervised_launch_reports_the_exit_the_supervisor_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let launcher = LocalLauncher::supervised(scripted_supervisor(dir.path()));
    let mut spec = JobSpec::single(vec!["/bin/sh".into(), "-c".into(), "exit 5".into()]);
    spec.workers = WorkerPlan::default();
    let launch = launcher.launch(&spec, 1, dir.path()).expect("launch");
    wait_until("the worker exits", Duration::from_secs(10), || {
        matches!(launcher.observe(&launch), Observation::Exited { .. })
    });
    assert_eq!(
        launcher.observe(&launch),
        Observation::Exited { exit_code: Some(5) }
    );
    // A controller that restarted holds no children and reads the same exit.
    let reopened = Launch::open(dir.path()).unwrap().unwrap();
    assert_eq!(
        reopened.observe(),
        Observation::Exited { exit_code: Some(5) }
    );
}

#[test]
fn a_launch_without_a_recorded_exit_says_so_instead_of_inventing_one() {
    let dir = tempfile::tempdir().unwrap();
    let spec = JobSpec::single(vec!["/bin/true".into()]);
    let launch = Launch {
        digest: spec.digest(),
        attempt: 1,
        world_size: 1,
        pids: vec![u32::MAX],
        dir: dir.path().to_path_buf(),
    };
    assert_eq!(
        launch.observe(),
        Observation::Exited { exit_code: None },
        "no supervisor recorded an exit, so none is known"
    );
    std::fs::write(dir.path().join("rank-0.exit"), "3\n").unwrap();
    assert_eq!(launch.observe(), Observation::Exited { exit_code: Some(3) });
}

#[test]
fn a_restarted_controller_reopens_the_launch_instead_of_launching_again() {
    let dir = tempfile::tempdir().unwrap();
    {
        let launcher = LocalLauncher::default();
        launcher
            .launch(&two_rank_spec(), 1, dir.path())
            .expect("launch");
    } // the controller is gone; its workers are not

    let reopened = Launch::open(dir.path())
        .expect("read the launch")
        .expect("a launch was written");
    let guard = Guard(reopened.clone());
    wait_until(
        "the reopened launch sees its workers",
        Duration::from_secs(10),
        || reopened.ranks_alive() == vec![0, 1],
    );
    assert!(matches!(reopened.observe(), Observation::Running { .. }));

    // A fresh controller must refuse to submit the same attempt again.
    let launcher = LocalLauncher::default();
    let error = launcher
        .launch(&two_rank_spec(), 1, dir.path())
        .expect_err("the attempt exists on disk");
    assert!(
        error.to_string().contains("already holds a launch"),
        "{error}"
    );
    guard.0.stop(None).unwrap();
    wait_until("the workers exit", Duration::from_secs(10), || {
        reopened.ranks_alive().is_empty()
    });
}
