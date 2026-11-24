// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data and learning pipelines built from typed,
// reusable operations, for its clients. If your team needs expertise in
// reproducible machine-learning data factories, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The built-in registry and its documentation agree, operation by operation.
/// The built-in registry is exactly these operations: adding or removing
/// one is a decision this list, the module table above and the guide
/// record together, never a side effect.
#[test]
fn the_registry_holds_exactly_the_documented_operations() {
    let registry = splinter_factory::ops::registry();
    let names = registry.names();
    let documented = [
        "agent",
        "answer.collect",
        "attempt.collect",
        "attempt.import",
        "attempts.failures",
        "dataset.extend",
        "dataset.import",
        "dataset.project",
        "dataset.rlvr",
        "dataset.text",
        "environment.build",
        "environment.qualify",
        "evaluation",
        "harness",
        "harness.search",
        "knowledge.admit",
        "knowledge.query",
        "model.train",
        "promotion",
        "source.ingest",
        "task.generate",
        "task.mutate",
        "tasks.pin",
        "train.online",
    ];
    assert_eq!(names, documented);
    let table = include_str!("../../src/ops.rs");
    for name in documented {
        assert!(
            table.contains(&format!("//! | `{name}` |")),
            "{name} is in the module's table"
        );
    }
}
