// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchange of agent benchmark tasks between
// task formats for its clients. If your team needs expertise in importing
// containerised benchmark tasks into reproducible, verifiable task sets, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: Harbor tasks and workspace tasks convert into each other.
//!
//! * a workspace task exported as a Harbor task and imported again is the
//!   same task (the same content address), so nothing the format can express
//!   is lost;
//! * a Harbor task written by someone else is imported when its environment
//!   is a repository checkout, held to the bar of every workspace task (its
//!   check must fail on the tree the solver gets), and graded by its own
//!   `tests/test.sh`: only a reward of exactly 1 passes, partial credit does
//!   not, and no reward is a failure;
//! * a task the host cannot honour is refused with the reason, and
//!   discovery lists importable and refused tasks without writing anything.
//!
//! The reward rules the parity checks encode are those of Harbor's own
//! verifier (`harbor/verifier/verifier.py` of harbor 0.24.0: `reward.json`
//! first, else `reward.txt`, the value read as a float).

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::{git, write, Repo};
use splinter_core::annotation::Outcome;
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Experience, Provenance};
use splinter_eval::verifiers::workspace::WorkspaceVerifier;
use splinter_eval::verifiers::Verifier;
use splinter_knowledge::tasks::workspace::{
    discover, Fixture, HarborTask, Held, Oracle, WorkspaceGenerator, WorkspaceTask,
};
use splinter_sandbox::{CommandCall, Limits, ProcessSandbox};

const AT: &str = "2026-10-19T09:00:00.000Z";

struct Rig {
    repo: Repo,
    generator: WorkspaceGenerator,
}

fn rig(name: &str) -> Rig {
    let repo = Repo::of_files(name, &[("README.md", "x\n")]);
    let calls = repo.scratch.0.join("calls");
    fs::create_dir_all(&calls).unwrap();
    let sandbox = ProcessSandbox::new(&calls, BTreeMap::new(), &[], Limits::default());
    let generator = WorkspaceGenerator::new(
        Arc::new(sandbox),
        repo.host.clone(),
        repo.store.clone(),
        repo.scratch.0.join("work"),
    );
    Rig { repo, generator }
}

fn sh(script: &str) -> CommandCall {
    CommandCall {
        argv: vec!["sh".into(), "-c".into(), script.into()],
        stdin: None,
    }
}

/// The change from `before` to `after` of `calc.sh`, as `git diff` writes it.
fn calc_diff(root: &Path, before: &str, after: &str) -> String {
    let dir = root.join("diff");
    let _ = fs::remove_dir_all(&dir);
    write(&dir.join("calc.sh"), before);
    git(&dir, &["init", "-q", "-b", "main"]);
    git(&dir, &["add", "-A"]);
    git(&dir, &["commit", "-q", "-m", "base"]);
    write(&dir.join("calc.sh"), after);
    git(&dir, &["diff"])
}

/// An authored workspace task: `calc.sh` must print 3, checked by a hidden
/// script, with the repository's README guarded and protected.
fn authored(rig: &Rig) -> WorkspaceTask {
    let root = &rig.repo.scratch.0;
    let tree = root.join("authored-tree");
    write(&tree.join("calc.sh"), "echo 1\n");
    write(&tree.join("README.md"), "a calculator\n");
    let fixture = Fixture {
        family: "calc".into(),
        instruction: "Make calc.sh print the sum of one and two.".into(),
        tree,
        hidden: BTreeMap::from([(
            "check.sh".to_string(),
            b"[ \"$(sh calc.sh)\" = 3 ]\n".to_vec(),
        )]),
        fail_to_pass: vec![sh("sh check.sh")],
        pass_to_pass: vec![sh("test -f README.md")],
        protected: vec!["README.md".into()],
        reference: Some(calc_diff(root, "echo 1\n", "echo 3\n")),
        expectations: Vec::new(),
        pinned: Vec::new(),
        hidden_strength: None,
        process: None,
    };
    let mut admitted = rig
        .generator
        .import_fixture(&fixture, &Held::none(), &FixedClock::new(AT))
        .unwrap();
    assert_eq!(admitted.admitted.len(), 1, "{:?}", admitted.refused);
    admitted.admitted.remove(0)
}

#[test]
fn a_workspace_task_exported_as_a_harbor_task_and_imported_again_is_the_same_task() {
    let rig = rig("harbor-roundtrip");
    let original = authored(&rig);
    let out = rig.repo.scratch.0.join("exported");

    rig.generator
        .export_harbor(&original.task, &out, "ubuntu:24.04")
        .unwrap();

    // Harbor's layout.
    for file in [
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "environment/workspace/calc.sh",
        "tests/test.sh",
        "tests/overlay/check.sh",
        "solution/solve.sh",
        "solution/reference.patch",
    ] {
        assert!(out.join(file).is_file(), "{file} is missing");
    }
    let dockerfile = fs::read_to_string(out.join("environment/Dockerfile")).unwrap();
    assert!(dockerfile.starts_with("FROM ubuntu:24.04"), "{dockerfile}");
    let verifier = fs::read_to_string(out.join("tests/test.sh")).unwrap();
    assert!(
        verifier.contains("'sh' '-c' 'sh check.sh' || status=1"),
        "{verifier}"
    );

    // Imported again, it is the task it was.
    let harbor = HarborTask::read(&out).unwrap();
    let work = rig.repo.scratch.0.join("reimport");
    fs::create_dir_all(&work).unwrap();
    let fixture = Fixture::harbor(&harbor, &work).unwrap();
    let again = rig
        .generator
        .import_fixture(&fixture, &Held::none(), &FixedClock::new(AT))
        .unwrap();
    assert_eq!(again.admitted.len(), 1, "{:?}", again.refused);
    assert_eq!(
        again.admitted[0].task, original.task,
        "the same content address"
    );
    assert_eq!(again.admitted[0].key, original.key);
}

/// A Harbor task written the way a benchmark author would: a repository
/// copied to /app, a test.sh that writes a reward, a solve.sh that fixes it.
fn foreign(root: &Path, name: &str, test_sh: &str) -> PathBuf {
    let dir = root.join(name);
    write(
        &dir.join("environment/Dockerfile"),
        "FROM ubuntu:24.04\nWORKDIR /app\nCOPY . /app/\n",
    );
    write(&dir.join("environment/calc.sh"), "echo 1\n");
    write(
        &dir.join("instruction.md"),
        "Make calc.sh print the sum of one and two.\n",
    );
    write(
        &dir.join("task.toml"),
        "schema_version = \"1.4\"\n[metadata]\ncategory = \"arithmetic\"\n[task]\nname = \"org/calc\"\n",
    );
    write(&dir.join("tests/test.sh"), test_sh);
    write(
        &dir.join("solution/solve.sh"),
        "#!/bin/bash\ncd /app\necho 'echo 3' > calc.sh\n",
    );
    dir
}

const REWARD_TXT: &str = "#!/bin/bash\ncd /app\nmkdir -p /logs/verifier\nif [ \"$(sh calc.sh)\" = 3 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n";

/// Grades `patch` on `task` as the loop does: the workspace verifier over an
/// experience that answered with exactly that change.
fn verdict(rig: &Rig, admitted: &WorkspaceTask, patch: &str) -> Outcome {
    let env = rig.generator.environment(&admitted.snapshot);
    let verifier = WorkspaceVerifier::new(
        vec![env],
        Arc::new(rig.repo.store.clone()),
        rig.repo.scratch.0.join("verify"),
    );
    let provenance = Provenance::new("test/candidate", &FixedClock::new(AT));
    let experience =
        Experience::answered_without_a_run(admitted.task.clone(), patch, provenance).unwrap();
    verifier
        .verify(&admitted.task, &experience)
        .unwrap()
        .outcome
}

fn import(rig: &Rig, dir: &Path, oracle: Oracle) -> Result<WorkspaceTask, String> {
    let task = HarborTask::read(dir).map_err(|e| e.to_string())?;
    let generation = rig
        .generator
        .import_harbor(&task, oracle, &Held::none(), &FixedClock::new(AT))
        .map_err(|e| e.to_string())?;
    generation
        .admitted
        .into_iter()
        .next()
        .ok_or_else(|| format!("refused: {:?}", generation.refused))
}

#[test]
fn a_foreign_task_is_graded_by_its_own_test_script_and_only_a_reward_of_one_passes() {
    let rig = rig("harbor-foreign");
    let dir = foreign(&rig.repo.scratch.0, "calc", REWARD_TXT);

    let admitted = import(&rig, &dir, Oracle::Require).unwrap();
    assert_eq!(
        admitted.family, "arithmetic",
        "the metadata category groups the task"
    );

    let root = &rig.repo.scratch.0;
    let right = calc_diff(root, "echo 1\n", "echo 3\n");
    let wrong = calc_diff(root, "echo 1\n", "echo 2\n");
    assert_eq!(verdict(&rig, &admitted, &right), Outcome::Pass);
    assert_eq!(verdict(&rig, &admitted, &wrong), Outcome::Fail);
    assert_eq!(
        verdict(&rig, &admitted, ""),
        Outcome::Fail,
        "the empty change fails"
    );
}

#[test]
fn partial_credit_and_a_missing_reward_are_not_a_pass() {
    let rig = rig("harbor-partial");
    // The verifier gives 0.5 for a calc.sh that prints something other than
    // the starting value, and 1.0 only for the right one.
    let json = "#!/bin/bash\ncd /app\nout=$(sh calc.sh)\nif [ \"$out\" = 3 ]; then echo '{\"reward\": 1.0}' > /logs/verifier/reward.json; elif [ \"$out\" != 1 ]; then echo '{\"reward\": 0.5}' > /logs/verifier/reward.json; else echo '{\"reward\": 0}' > /logs/verifier/reward.json; fi\n";
    let admitted = import(
        &rig,
        &foreign(&rig.repo.scratch.0, "partial", json),
        Oracle::Skip,
    )
    .unwrap();
    let root = &rig.repo.scratch.0;
    assert_eq!(
        verdict(&rig, &admitted, &calc_diff(root, "echo 1\n", "echo 3\n")),
        Outcome::Pass
    );
    assert_eq!(
        verdict(&rig, &admitted, &calc_diff(root, "echo 1\n", "echo 2\n")),
        Outcome::Fail,
        "a reward of 0.5 is partial credit, never success"
    );

    // A verifier that writes no reward cannot pass anything; a task whose
    // check passes on the unmodified tree is refused at import.
    let silent = foreign(&rig.repo.scratch.0, "silent", "#!/bin/bash\ntrue\n");
    // Without an oracle that is all the bar can see; with one, the solution
    // is shown not to pass and the task is refused.
    assert!(import(&rig, &silent, Oracle::Skip).is_ok());
    let why = import(&rig, &silent, Oracle::Require).unwrap_err();
    assert!(
        why.contains("NotRestorable"),
        "no reward means the oracle never passes: {why}"
    );
    let vacuous = foreign(
        &rig.repo.scratch.0,
        "vacuous",
        "#!/bin/bash\nmkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n",
    );
    let why = import(&rig, &vacuous, Oracle::Skip).unwrap_err();
    assert!(
        why.contains("NotFailingAsDescribed") || why.contains("refused"),
        "{why}"
    );
}

#[test]
fn a_task_the_host_cannot_honour_is_refused_with_the_reason_and_discovery_writes_nothing() {
    let rig = rig("harbor-refused");
    let root = rig.repo.scratch.0.join("tasks");
    let good = foreign(&root, "good", REWARD_TXT);
    let run_step = foreign(&root, "builds", REWARD_TXT);
    write(
        &run_step.join("environment/Dockerfile"),
        "FROM ubuntu\nWORKDIR /app\nRUN apt-get install -y gcc\nCOPY . /app/\n",
    );
    let image = foreign(&root, "image", REWARD_TXT);
    write(
        &image.join("task.toml"),
        "schema_version = \"1.4\"\n[environment]\ndocker_image = \"x/y:1\"\n",
    );
    let windows = foreign(&root, "windows", REWARD_TXT);
    write(
        &windows.join("task.toml"),
        "schema_version = \"1.4\"\n[environment]\nos = \"windows\"\n",
    );
    let no_test = foreign(&root, "no-test", REWARD_TXT);
    fs::remove_file(no_test.join("tests/test.sh")).unwrap();

    let before: Vec<_> = walk(&root);
    let found = discover(&root).unwrap();
    assert_eq!(walk(&root), before, "discovery writes nothing");

    let reason = |name: &str| {
        found
            .iter()
            .find(|d| d.dir.ends_with(name))
            .unwrap_or_else(|| panic!("{name} not discovered"))
            .refusal
            .clone()
    };
    assert_eq!(reason("good"), None);
    assert!(reason("builds").unwrap().contains("RUN"));
    assert!(reason("image").unwrap().contains("prebuilt image"));
    assert!(reason("windows").unwrap().contains("Windows"));
    assert!(reason("no-test").unwrap().contains("test.sh"));
    assert_eq!(found.iter().filter(|d| d.refusal.is_none()).count(), 1);
    let _ = good;
}

fn walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(root).unwrap().flatten() {
        out.push(entry.path());
        if entry.path().is_dir() {
            out.extend(walk(&entry.path()));
        }
    }
    out.sort();
    out
}
