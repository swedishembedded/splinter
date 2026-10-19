// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements audits that prove an agent never saw the
// material it is graded against, for its clients. If your team needs
// expertise in contamination-free evaluation of tool-using agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a trajectory in which a tool call or output names a hidden path,
//! a tool output quotes hidden content, or a call names the state root is a
//! fail from the exposure producer; a clean trajectory is an abstention,
//! and text the solver may legitimately see is not exposure.

#![allow(clippy::unwrap_used)]

mod common;

use atif::{
    AgentProfile, ObservationEntry, StepObservation, StepOrigin, ToolInvocation, TraceStep,
    Trajectory,
};
use common::{task, verdict};
use serde_json::json;
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Experience, Provenance, Task};
use splinter_eval::exposure::{audit_exposure, ExposureAudit, HiddenMaterial, EXPOSURE_PRODUCER};
use splinter_eval::verifiers::Verifier;

const HIDDEN_TEST: &str = "import unittest\nfrom calc import add\n\nclass Hidden(unittest.TestCase):\n    def test_secret_sum_of_twelve(self):\n        self.assertEqual(add(5, 7), 12)\n        self.assertEqual(add(-3, 3), 0)\n";
const STATE_ROOT: &str = "/var/lib/splinter-state";

fn hidden() -> HiddenMaterial {
    HiddenMaterial {
        paths: vec!["tests/test_hidden.py".into()],
        contents: vec![HIDDEN_TEST.into()],
        public_contents: vec!["import unittest\nfrom calc import add\n".into()],
        forbidden: vec![STATE_ROOT.into()],
        confine_to_workspace: true,
    }
}

fn the_task() -> Task {
    task("workspace-change", Environment::closed_book(), vec![])
}

/// An experience of `task` with one tool call and its output.
fn run_of(task: &Task, command: &str, output: &str) -> Experience {
    run_of_parallel(task, &[(command, output)])
}

/// An experience of `task` whose one agent step made all of `calls` in
/// parallel, each with the output it returned. Every call has the id `call_0`,
/// as the recorded trajectories do.
fn run_of_parallel(task: &Task, calls: &[(&str, &str)]) -> Experience {
    let profile = AgentProfile {
        name: "t".into(),
        version: "1".into(),
        model_name: None,
        tool_definitions: None,
        extra: None,
    };
    let mut trajectory = Trajectory::new("ATIF-v1.7", profile);
    trajectory
        .steps
        .push(TraceStep::new(1, StepOrigin::User, "Fix the bug."));
    let mut act = TraceStep::new(2, StepOrigin::Agent, "Looking.");
    act.tool_calls = Some(
        calls
            .iter()
            .map(|(command, _)| {
                ToolInvocation::new("call_0", "shell").with_arguments(json!({ "command": command }))
            })
            .collect(),
    );
    let mut results = calls
        .iter()
        .map(|(_, output)| ObservationEntry::for_call("call_0", *output));
    let mut observation = StepObservation::single(results.next().unwrap());
    observation.results.extend(results);
    act.observation = Some(observation);
    trajectory.steps.push(act);
    Experience::new(
        task.clone(),
        trajectory,
        Some("done".into()),
        Provenance::new("s", &FixedClock::new("2026-09-30T00:00:00.000Z")),
    )
    .unwrap()
}

#[test]
fn a_call_or_output_naming_a_hidden_path_or_the_state_root_is_exposure() {
    let task = the_task();
    let cases = [
        run_of(&task, "cat tests/test_hidden.py", "No such file"),
        run_of(&task, "ls", "calc.py\ntests/test_hidden.py\n"),
        run_of(&task, &format!("ls {STATE_ROOT}/artifacts"), ""),
    ];
    for exp in &cases {
        let finding = audit_exposure(&task, exp, &hidden());
        assert_eq!(finding.outcome, Outcome::Fail, "{}", finding.evidence);
    }
}

#[test]
fn a_tool_output_quoting_hidden_content_is_exposure_without_leaking_it() {
    let task = the_task();
    // A line-numbered listing still carries the content.
    let numbered: String = HIDDEN_TEST
        .lines()
        .enumerate()
        .map(|(i, l)| format!("{:>6}\t{l}\n", i + 1))
        .collect();
    let exp = run_of(&task, "cat x", &numbered);
    let finding = audit_exposure(&task, &exp, &hidden());
    assert_eq!(finding.outcome, Outcome::Fail);
    assert!(!finding.evidence.to_string().contains("test_secret_sum"));

    let (outcome, strength, evidence) = verdict(&ExposureAudit::new(hidden()), &task, &exp);
    assert_eq!(outcome, Outcome::Fail, "{evidence}");
    assert_eq!(strength, Strength::Constraint);
    assert_eq!(
        ExposureAudit::new(hidden()).producer().name,
        EXPOSURE_PRODUCER
    );
}

#[test]
fn a_clean_trajectory_abstains_and_visible_boilerplate_is_not_exposure() {
    let task = the_task();
    let clean = run_of(&task, "python3 -m unittest", "Ran 1 test\nOK\n");
    assert_eq!(
        audit_exposure(&task, &clean, &hidden()).outcome,
        Outcome::Abstain
    );
    // The visible tests start the same way the hidden one does.
    let shared = run_of(
        &task,
        "cat tests/test_calc.py",
        "import unittest\nfrom calc import add\n",
    );
    assert_eq!(
        audit_exposure(&task, &shared, &hidden()).outcome,
        Outcome::Abstain
    );
}

fn outside(command: &str) -> bool {
    let task = the_task();
    let finding = audit_exposure(&task, &run_of(&task, command, "SECRET = 1"), &hidden());
    finding.outcome == Outcome::Fail && finding.evidence.to_string().contains("outside_workspace")
}

#[test]
fn a_command_naming_a_path_outside_the_tree_is_exposure_but_the_system_and_the_tree_are_not() {
    assert!(outside("cat /srv/x/y.py"));
    assert!(outside("find / -name iniparse"));
    assert!(outside("git -C /srv/x/repo log"));
    assert!(outside("ls /work/a/tree && cat /work/b/tree/a.py"));
    assert!(!outside("cat /usr/lib/python3/os.py"));
    assert!(!outside("python3 /bin/true && ls /dev/null"));
    assert!(!outside("python3 -c 'print(1 / 2)'"));
    assert!(!outside("curl https://example.com/x"));
    assert!(!outside("ls /work/a/tree && cat /work/a/tree/a.py"));
}

#[test]
fn the_recorded_tree_is_the_one_that_counts() {
    let task = the_task();
    let mut exp = run_of(
        &task,
        "cat /work/mine/a.py && cat /work/other/a.py",
        "x = 1",
    );
    exp.trajectory.agent.extra = Some(json!({ "workspace_tree": "/work/mine" }));
    let finding = audit_exposure(&task, &exp, &hidden());
    assert_eq!(finding.outcome, Outcome::Fail);
    let mut own_only = run_of(&task, "cat /work/mine/a.py", "");
    own_only.trajectory.agent.extra = Some(json!({ "workspace_tree": "/work/mine" }));
    assert_eq!(
        audit_exposure(&task, &own_only, &hidden()).outcome,
        Outcome::Abstain
    );
}

/// The kinds the audit reports for `command` whose output was `output`.
fn kinds_of(command: &str, output: &str) -> Vec<String> {
    let task = the_task();
    let finding = audit_exposure(&task, &run_of(&task, command, output), &hidden());
    finding.evidence["exposed"]
        .as_array()
        .or_else(|| finding.evidence["detail"]["outside_probes"].as_array())
        .into_iter()
        .flatten()
        .filter_map(|f| f["kind"].as_str().map(str::to_string))
        .collect()
}

#[test]
fn a_path_outside_the_tree_is_reported_by_what_came_of_it_and_none_is_dropped() {
    // The sandbox refused it.
    assert_eq!(
        kinds_of("cat /srv/x/y.py", "cat: /srv/x/y.py: Permission denied"),
        ["outside_workspace_denied"]
    );
    // A path that is not there, and a search that returned nothing.
    assert_eq!(
        kinds_of(
            "cd /nowhere/user",
            "bash: cd: /nowhere/user: No such file or directory"
        ),
        ["outside_workspace_absent"]
    );
    assert_eq!(
        kinds_of("find / -name iniparse", ""),
        ["outside_workspace_absent"]
    );
    // Something came back: an access that may have succeeded.
    assert_eq!(
        kinds_of("cat /srv/x/y.py", "SECRET = 1"),
        ["outside_workspace"]
    );
    // A refusal next to a result is not a refusal.
    assert_eq!(
        kinds_of(
            "find / -name a.py",
            "find: '/root': Permission denied\n/srv/x/a.py"
        ),
        ["outside_workspace"]
    );
    // Only an access that returned something fails the audit; a refused or
    // empty probe is recorded as a soft flag and the attempt stays clean.
    let task = the_task();
    let denied = run_of(
        &task,
        "ls /srv",
        "ls: cannot open directory '/srv': Permission denied",
    );
    let finding = audit_exposure(&task, &denied, &hidden());
    assert_eq!(finding.outcome, Outcome::Abstain);
    assert_eq!(
        finding.evidence["detail"]["outside_probes"][0]["kind"],
        "outside_workspace_denied"
    );
}

#[test]
fn the_shells_exit_code_line_is_not_something_that_came_back() {
    // `cd` into a path that is not there ends the shell tool's output with its
    // own `[exit code 1]` line; that line is not a result.
    let task = the_task();
    let exp = run_of(
        &task,
        "cd /nowhere/repos/x && python3 -m unittest",
        "bash: line 1: cd: /nowhere/repos/x: No such file or directory\n\n[exit code 1]",
    );
    let finding = audit_exposure(&task, &exp, &hidden());
    assert_eq!(finding.outcome, Outcome::Abstain);
    assert_eq!(
        finding.evidence["detail"]["outside_probes"][0]["kind"],
        "outside_workspace_absent"
    );
}

#[test]
fn a_probe_is_judged_by_its_own_result_not_by_its_parallel_siblings() {
    // One step, two parallel calls: a failed `cd` and a file read that
    // returned text. The read's output must not make the `cd` look reached.
    let task = the_task();
    let exp = run_of_parallel(
        &task,
        &[
            (
                "cd /app && python3 -m unittest",
                "bash: line 1: cd: /app: No such file or directory\n\n[exit code 1]",
            ),
            ("cat README.md", "L1:# project"),
        ],
    );
    let finding = audit_exposure(&task, &exp, &hidden());
    assert_eq!(finding.outcome, Outcome::Abstain, "{}", finding.evidence);

    // A sibling that really reached outside still fails the audit.
    let reached = run_of_parallel(
        &task,
        &[
            ("cd /app && ls", "bash: cd: /app: No such file or directory"),
            ("cat /srv/x/y.py", "SECRET = 1"),
        ],
    );
    assert_eq!(
        audit_exposure(&task, &reached, &hidden()).outcome,
        Outcome::Fail
    );
}

#[test]
fn a_failed_read_of_a_hidden_path_saw_nothing_and_is_only_a_probe() {
    let task = the_task();
    let hidden_name = &hidden().paths[0];
    let command = format!("cat {hidden_name}");
    let failed = run_of(
        &task,
        &command,
        "cat: tests/test_x.py: No such file or directory\n\n[exit code 1]",
    );
    let finding = audit_exposure(&task, &failed, &hidden());
    assert_eq!(finding.outcome, Outcome::Abstain, "{}", finding.evidence);
    assert_eq!(
        finding.evidence["detail"]["outside_probes"][0]["kind"],
        "hidden_path_absent"
    );

    // The same call that returned text is exposure, and so is a hidden path
    // named in what a call returned.
    let read = run_of(&task, &command, "L1:import unittest");
    assert_eq!(
        audit_exposure(&task, &read, &hidden()).outcome,
        Outcome::Fail
    );
}
