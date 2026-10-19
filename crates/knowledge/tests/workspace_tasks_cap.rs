// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a test report longer than the sandbox's output cap keeps its
//! summary, and the verdict is decided from it rather than refused.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::sync::Arc;

use common::Repo;
use splinter_eval::verifiers::test_report::Verdict;
use splinter_knowledge::tasks::workspace::{Harness, WorkspaceGenerator, WorkspacePlan};
use splinter_sandbox::{CommandCall, Limits, ProcessSandbox};

/// A runner that prints 200 KiB of noise, then the unittest summary the
/// suite would end with: `FAILED` with one failure listed, or `OK`.
fn fake_runner(failing: bool) -> CommandCall {
    let tail = if failing {
        "======\\nFAIL: test_b (m.T.test_b)\\n------\\nAssertionError\\n\\nRan 3 tests in 0.1s\\n\\nFAILED (failures=1)\\n"
    } else {
        "\\nRan 3 tests in 0.1s\\n\\nOK\\n"
    };
    let script = format!(
        "import sys\nsys.stderr.write('test_a (m.T.test_a) ... ok\\n')\n\
         for i in range(4000):\n    sys.stderr.write('x' * 49 + '\\n')\n\
         sys.stderr.write('{tail}')\nsys.exit({})\n",
        i32::from(failing)
    );
    CommandCall {
        argv: vec!["python3".into(), "-c".into(), script],
        stdin: None,
    }
}

#[test]
fn a_test_report_past_the_output_cap_is_decided_from_its_summary_not_refused() {
    let repo = Repo::new("cap", &[]);
    let calls = repo.scratch.0.join("capped-calls");
    fs::create_dir_all(&calls).unwrap();
    let limits = Limits {
        output_cap_bytes: 4096,
        ..Limits::default()
    };
    let generator = WorkspaceGenerator::new(
        Arc::new(ProcessSandbox::new(
            &calls,
            BTreeMap::new(),
            &["PATH"],
            limits,
        )),
        repo.host.clone(),
        repo.store.clone(),
        repo.scratch.0.join("capped-work"),
    )
    .with_plan(WorkspacePlan::default())
    .unwrap();
    let tree = repo.capture("HEAD");
    let laid = repo.scratch.0.join("capped-laid");
    let env = generator.environment(&tree.snapshot);
    env.materialize(&tree.archive_path, &laid).unwrap();

    let failed = env.run(&laid, &fake_runner(true)).unwrap();
    assert!(failed.stderr_truncated, "the report must pass the cap");
    let report = Harness::Unittest.read(&failed);
    assert!(!report.unreadable(), "a summary was kept: {report:?}");
    assert_eq!(report.verdict, Some(Verdict::Fail));
    assert!(report.failed.contains("m.T.test_b"), "{report:?}");

    let passed = env.run(&laid, &fake_runner(false)).unwrap();
    assert!(passed.stderr_truncated);
    let report = Harness::Unittest.read(&passed);
    assert!(!report.unreadable(), "{report:?}");
    assert_eq!(report.verdict, Some(Verdict::Pass));
    assert!(report.failed.is_empty());
}
