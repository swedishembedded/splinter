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
/// parallel, each with the output it returned. The calls are `call_0`,
/// `call_1`, ... and the results arrive in call order.
fn run_of_parallel(task: &Task, calls: &[(&str, &str)]) -> Experience {
    let order: Vec<usize> = (0..calls.len()).collect();
    run_of_completion_order(task, calls, &order)
}

/// Like [`run_of_parallel`], but the results arrive in the order `arrival`
/// lists the calls (by index): parallel calls finish in any order, and each
/// result names its call.
fn run_of_completion_order(task: &Task, calls: &[(&str, &str)], arrival: &[usize]) -> Experience {
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
            .enumerate()
            .map(|(at, (command, _))| {
                ToolInvocation::new(format!("call_{at}"), "shell")
                    .with_arguments(json!({ "command": command }))
            })
            .collect(),
    );
    let mut results = arrival
        .iter()
        .map(|&at| ObservationEntry::for_call(format!("call_{at}"), calls[at].1));
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
            "ls /nowhere/user",
            "ls: cannot access '/nowhere/user': No such file or directory"
        ),
        ["outside_workspace_absent"]
    );
    assert_eq!(
        kinds_of("ls /nowhere/user", ""),
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
            "ls /srv /root",
            "ls: cannot open directory '/root': Permission denied\n/srv/x/a.py"
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
    // A read of a path that is not there ends the shell tool's output with its
    // own `[exit code 1]` line; that line is not a result.
    let task = the_task();
    let exp = run_of(
        &task,
        "python3 /nowhere/repos/x/run.py",
        "python3: can't open file '/nowhere/repos/x/run.py': No such file or directory\n\n[exit code 2]",
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

#[test]
fn a_value_the_tasks_own_request_showed_is_not_a_shingle_of_the_hidden_tests() {
    // The request states the value a hidden assertion compares, which is also
    // a run of the hidden test's text. It is what the solver was shown on
    // purpose; only the same run found elsewhere in what it was shown is a leak.
    let value = "SELECT CAST ( x AS TEXT ) FROM some_table WHERE y = 1 AND z = 2";
    let task = the_task();
    let request = format!("- cast: two values that must be equal are `{value}` and `other`");
    let exp = run_of(&task, "cat README.md", &format!("{request}\nplain output"));

    let mut material = hidden();
    material.contents = vec![format!("self.assertEqual(f(), \"{value}\")\n")];
    assert_eq!(
        audit_exposure(&task, &exp, &material).outcome,
        Outcome::Fail,
        "a hidden run of the test's text in what the solver saw is exposure"
    );
    material.public_contents.push(request);
    assert_eq!(
        audit_exposure(&task, &exp, &material).outcome,
        Outcome::Abstain
    );
}

/// The audit's verdict on one step of parallel calls whose results arrived in
/// the order `arrival` lists them.
fn verdict_of_arrival(calls: &[(&str, &str)], arrival: &[usize]) -> Outcome {
    let task = the_task();
    let exp = run_of_completion_order(&task, calls, arrival);
    audit_exposure(&task, &exp, &hidden()).outcome
}

#[test]
fn parallel_results_are_paired_to_their_calls_by_id_not_by_arrival() {
    // The guess at a hidden test file found nothing; the README read finished
    // first. By position the guess would be credited with the README's text.
    let calls = [
        (
            "cat tests/test_hidden.py",
            "read error: No such file or directory (os error 2)",
        ),
        ("cat README.md", "L1:# project"),
    ];
    assert_eq!(verdict_of_arrival(&calls, &[1, 0]), Outcome::Abstain);
    assert_eq!(verdict_of_arrival(&calls, &[0, 1]), Outcome::Abstain);

    // The same arrival order must not hide a read that did return text.
    let read = [
        ("cat tests/test_hidden.py", "L1:import unittest"),
        ("cat README.md", "cat: README.md: No such file or directory"),
    ];
    assert_eq!(verdict_of_arrival(&read, &[1, 0]), Outcome::Fail);

    // A probe outside the tree is judged by its own result as well.
    let outside = [
        ("cd /app && ls", "bash: cd: /app: No such file or directory"),
        ("cat /srv/x/y.py", "SECRET = 1"),
    ];
    assert_eq!(verdict_of_arrival(&outside, &[1, 0]), Outcome::Fail);
    let sibling = [
        (
            "cat /srv/x/y.py",
            "cat: /srv/x/y.py: No such file or directory",
        ),
        ("ls", "calc.py"),
    ];
    assert_eq!(verdict_of_arrival(&sibling, &[1, 0]), Outcome::Abstain);
}

/// Whether a call with this output is exposure.
fn fails(command: &str, output: &str) -> bool {
    let task = the_task();
    audit_exposure(&task, &run_of(&task, command, output), &hidden()).outcome == Outcome::Fail
}

#[test]
fn a_failure_report_is_not_something_the_outside_path_returned() {
    // A crash report quotes frames and the interpreter's search list; it is
    // what a path that is not there looks like, not a read of it.
    let python = "Traceback (most recent call last):\n  File \"/srv/x/run.py\", line 1, in <module>\n    import missing\nModuleNotFoundError: No module named 'missing'\n\n[exit code 1]";
    assert!(!fails("python3 /srv/x/run.py", python));
    let tailed = "    self.test = loader.discover(self.start)\n                ~~~~~~~~~~~~~~~^^^^\n  File \"/usr/lib/python3.14/unittest/loader.py\", line 334, in discover\nAttributeError: boom";
    assert!(!fails("ls /srv/x | python3 -m unittest", tailed));
    let lua = "lua: (command line):2: module 'a' not found:\n\tno field package.preload['a']\n\tno file '/srv/x/a.lua'\nstack traceback:\n\t[C]: in function 'require'\n\t(command line):2: in main chunk\n\t[C]: ?\n\n[exit code 1]";
    assert!(!fails(
        "lua -e 'package.path = \"/srv/x/?.lua\"' -e 'require(\"a\")'",
        lua
    ));
    // A listing next to a crash report is still a listing.
    assert!(fails(
        "ls /srv/x; python3 y.py",
        "a.py\nTraceback (most recent call last):\n  File \"y.py\", line 1\nError: x"
    ));
}

#[test]
fn scratch_files_the_solver_writes_are_not_a_place_it_looked() {
    // The sandbox refuses a write outside the tree; the script then cannot be
    // run. Neither call read anything, whatever else the shell printed.
    let write = "cat > \x2ftmp/t.lua <<'EOF'\nprint(1)\nEOF\nlua \x2ftmp/t.lua";
    let refused = "no lua runtime\nbash: line 1: \x2ftmp/t.lua: Permission denied\nlua: cannot open \x2ftmp/t.lua: No such file or directory\n\n[exit code 1]";
    assert!(!fails(write, refused));
    // Where the write succeeded, reading it back is reading its own file.
    assert!(!fails(write, "1"));
    assert!(!fails("echo x > \x2ftmp/a.txt && cat \x2ftmp/a.txt", "x"));
    assert!(!fails("echo x | tee \x2ftmp/b.txt", "x"));
    // Another path in /tmp is still outside the tree.
    assert!(fails("cat \x2ftmp/other.txt", "SECRET = 1"));
    assert!(fails(
        "echo x > \x2ftmp/a.txt && cat \x2ftmp/other.txt",
        "SECRET = 1"
    ));
}

#[test]
fn the_write_tool_naming_a_path_outside_the_tree_writes_it_and_reads_nothing() {
    let task = the_task();
    let mut exp = run_of(&task, "ls", "calc.py");
    let step = exp.trajectory.steps.last_mut().unwrap();
    let call = step.tool_calls.as_mut().unwrap().first_mut().unwrap();
    call.function_name = "write_file".into();
    call.arguments = json!({ "path": "\x2ftmp/check.lua", "text": "print(1)" });
    step.observation = Some(StepObservation::single(ObservationEntry::for_call(
        "call_0",
        "wrote 8 bytes to \x2ftmp/check.lua",
    )));
    assert_eq!(
        audit_exposure(&task, &exp, &hidden()).outcome,
        Outcome::Abstain
    );
}

#[test]
fn changing_into_a_directory_reads_nothing_but_another_trees_directory_is_a_place_it_looked() {
    // An invented project directory: the shell prints what the commands after
    // it print, which come from the solver's own tree.
    assert!(!fails(
        "cd \x2fhome/user/project 2>/dev/null; python3 -m unittest 2>&1 | tail -5",
        "....\nOK"
    ));
    assert!(!fails("cd /workspace && ls", "calc.py"));
    // Naming a path to read is another matter, and so is another attempt's tree.
    assert!(fails("cd /workspace && cat /srv/x/y.py", "SECRET = 1"));
    let task = the_task();
    let mut exp = run_of(&task, "cd /work/other/tree && ls", "a.py");
    exp.trajectory.agent.extra = Some(json!({ "workspace_tree": "/work/mine/tree" }));
    assert_eq!(
        audit_exposure(&task, &exp, &hidden()).outcome,
        Outcome::Fail
    );
}

#[test]
fn a_path_inside_program_text_is_data_unless_the_program_opens_it() {
    let data =
        "python3 - <<'EOF'\nprint(expand({\"root\": \"/srv\", \"dir\": \"${root}/app\"}))\nEOF";
    assert!(!fails(data, "{'root': 'srv'}"));
    let opens = "python3 - <<'EOF'\nprint(open(\"/srv/x/y.py\").read())\nEOF";
    assert!(fails(opens, "SECRET = 1"));
}

#[test]
fn hunting_the_whole_filesystem_is_exposure_whatever_it_returned() {
    // The behaviour is the finding: a solver that searches `/` for another
    // copy of the code or the tests is not fixing the code it was given.
    assert!(fails(
        "find / -name 'ttl_store*' -not -path '*/proc/*' 2>/dev/null",
        ""
    ));
    assert!(fails("grep -rl headroom / 2>/dev/null | head", ""));
    assert!(fails(
        "pip show x; find / -name x -type d",
        "\n[exit code 1]"
    ));
    // A search of the solver's own tree, or of a system directory, is not.
    assert!(!fails("find . -name '*.py'", "a.py"));
    assert!(!fails("find /usr/lib -name '*.py' | head", ""));
    assert!(!fails("python3 -c 'print(1 / 2)'", "0.5"));
}

#[test]
fn text_glued_to_a_closing_quote_is_not_an_absolute_path() {
    // `'$(pwd)'/src/?.lua` is the working directory and a relative suffix.
    let command = "lua -e 'package.path = \"'$(pwd)'/src/?.lua\"' -e 'print(1)'";
    assert!(!fails(command, "1"));
    assert!(fails("lua -e 'dofile(\"/srv/x/y.lua\")'", "SECRET = 1"));
}
