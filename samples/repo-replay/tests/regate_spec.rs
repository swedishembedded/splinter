// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the current gate can be applied again to attempts that already
//! passed, without a model. What it still passes stays as it was; what it
//! fails is recorded as failed (the report, the splits and the dataset see the
//! new verdict) and can be redone by class. A report-only run changes nothing.

#![allow(clippy::unwrap_used)]

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::dataset::{attempts_set_filtered, build_dataset, DatasetSettings};
use splinter_repo_replay::fixture;
use splinter_repo_replay::gate::GateConfig;
use splinter_repo_replay::ledger::{self, Outcome};
use splinter_repo_replay::regate::{regate, RegateConfig};
use splinter_repo_replay::scripted::{Behaviour, Specificity};

fn passed(state: &std::path::Path) -> usize {
    ledger::latest(&ledger::read(state).unwrap())
        .iter()
        .filter(|e| e.outcome == Outcome::Passed)
        .count()
}

#[test]
fn passes_are_judged_again_by_the_current_gate_without_a_model() {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let ids = fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.from = Some(ids[1].clone());
    config.to = Some(ids[2].clone());
    run_blocking(
        &ctx,
        &scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths),
        &config,
    )
    .unwrap();
    let before = passed(&state);
    assert!(before > 0);

    let mut regating = RegateConfig {
        source: source.clone(),
        state: state.clone(),
        select: config.select,
        gate: config.gate,
        budget: config.budget,
        write: true,
    };

    // The gate that judged them: nothing changes and nothing is written.
    let lines = ledger::read(&state).unwrap().len();
    let same = regate(&ctx, &regating).unwrap();
    assert_eq!((same.examined, same.failed.len()), (before, 0));
    assert_eq!(ledger::read(&state).unwrap().len(), lines);

    // A stricter gate, report only: found, not recorded.
    regating.gate = GateConfig {
        min_tool_calls: 1000,
        ..config.gate
    };
    regating.write = false;
    let dry = regate(&ctx, &regating).unwrap();
    assert_eq!(dry.failed.len(), before);
    assert_eq!(dry.by_class()["tool_calls_out_of_range"], before);
    assert_eq!(ledger::read(&state).unwrap().len(), lines);
    assert_eq!(passed(&state), before);

    // Recorded: the new verdict stands, the dataset leaves them out.
    regating.write = true;
    regate(&ctx, &regating).unwrap();
    assert_eq!(passed(&state), 0);
    let latest = ledger::read(&state).unwrap();
    let last = latest.last().unwrap();
    assert_eq!(last.outcome, Outcome::Failed);
    assert_eq!(last.classes, vec!["tool_calls_out_of_range"]);
    assert!(last.detail.as_deref().unwrap().starts_with("regated: "));
    let set =
        attempts_set_filtered(&ctx, &state, "all", &|e| e.outcome == Outcome::Failed).unwrap();
    let built = build_dataset(
        &ctx,
        &[set],
        &DatasetSettings {
            serving_prompt: "serve".into(),
            max_block: None,
        },
    );
    assert!(built.is_err() || built.unwrap().records == 0);

    // Already failed: not judged again.
    assert_eq!(regate(&ctx, &regating).unwrap().examined, 0);
}
