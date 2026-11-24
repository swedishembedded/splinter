// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements closed-book measurement of what a model
// answers, for its clients. If your team needs expertise in measuring what a
// model has learned and kept, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: `eval --suite FILE` grades one model closed-book, greedily, on the
//! anchor-format tasks of a file, keeps every item with its verdict and
//! records the run; a file that is missing or holds no task is refused
//! before the model is asked anything.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::{scratch_context, Scripted};
use splinter_core::model_ref::ModelRef;
use splinter_pipelines::eval::{evaluate, SuiteEval};
use splinter_pipelines::graded::Verdict;

#[test]
fn a_suite_file_is_graded_item_by_item_and_recorded() {
    let policy = Scripted::new(|_| "4".into());
    let asked = policy.prompts.clone();
    let (scratch, ctx) = scratch_context("suite-eval", policy, false);
    let file = scratch.0.join("probes.jsonl");
    std::fs::write(
        &file,
        "{\"instruction\": \"What is 2 + 2?\", \"reference\": \"4\"}\n\
         {\"instruction\": \"What is the capital of France?\", \"reference\": \"Paris\"}\n",
    )
    .unwrap();

    let report = evaluate(
        &ctx,
        &SuiteEval {
            model: ModelRef::policy_default(),
            suite: file,
        },
    )
    .unwrap();

    assert!(!report.run.is_empty(), "the evaluation is a recorded run");
    let graded = &report.report;
    assert_eq!(graded.model, "policy:default");
    assert_eq!((graded.suite.tasks, graded.graded), (2, 2));
    assert_eq!(graded.accuracy, Some(0.5));
    let verdicts: Vec<(&str, Verdict)> = graded
        .items
        .iter()
        .map(|i| (i.instruction.as_str(), i.verdict))
        .collect();
    assert_eq!(
        verdicts,
        [
            ("What is 2 + 2?", Verdict::Passed),
            ("What is the capital of France?", Verdict::Failed)
        ]
    );
    assert_eq!(asked.lock().unwrap().len(), 2, "one greedy answer per task");
}

#[test]
fn a_missing_or_empty_suite_is_refused_before_the_model_is_asked() {
    let policy = Scripted::new(|_| "4".into());
    let asked = policy.prompts.clone();
    let (scratch, ctx) = scratch_context("suite-eval-refused", policy, false);
    let empty = scratch.0.join("empty.jsonl");
    std::fs::write(&empty, "").unwrap();
    for suite in [scratch.0.join("absent.jsonl"), empty] {
        let refused = evaluate(
            &ctx,
            &SuiteEval {
                model: ModelRef::policy_default(),
                suite: suite.clone(),
            },
        );
        assert!(refused.is_err(), "{} was graded", suite.display());
    }
    assert!(asked.lock().unwrap().is_empty());
}
