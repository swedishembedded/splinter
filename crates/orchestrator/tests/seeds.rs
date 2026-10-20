// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training and evaluation
// pipelines for its clients. If your team needs expertise in seeding,
// reproducibility and experiment provenance for machine learning then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a run owns the seeds of its random draws. A seed the caller gives is
//! used and recorded; one left out is drawn, recorded and reported to the
//! progress receiver once, so passing the reported value reproduces the run.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde_json::json;
use splinter_core::clock::FixedClock;
use splinter_core::rng::Seed;
use splinter_orchestrator::runs::record;
use splinter_orchestrator::{Config, Context};
use splinter_store::runs::read_run;
use splinter_store::StateRoot;

/// The stages a context reported, with the detail each carried.
type Reported = Arc<Mutex<Vec<(String, serde_json::Value)>>>;

/// A context over a scratch state root, with every reported stage collected.
fn context(test: &str) -> (Context, Reported) {
    let root = std::env::temp_dir().join(format!("splinter-seeds-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let config = Config {
        state_root: StateRoot::new(root.join("state")),
        model_store: root.join("models"),
        policy_base: root.join("models/base"),
        policy_context_tokens: None,
        openrouter_api_key: None,
        brain_api_key: None,
        allow_remote: false,
        command_env: BTreeMap::new(),
        working_dir: root,
        brain_binary: None,
        front_door_model: None,
        assistant_model: None,
        judge_model: None,
        bf16_base: false,
        default_budget: None,
        remote_concurrency: 4,
        solve_deadline: splinter_orchestrator::config::DEFAULT_SOLVE_DEADLINE,
        min_calibration_controls: 2,
        workspace: Default::default(),
        thinking: false,
        served_temperature: None,
        served_top_p: None,
        served_top_k: None,
        served_seed: None,
        served_thinking_budget: None,
        build: Default::default(),
    };
    let reported = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&reported);
    let ctx = Context::new(config, false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-09-30T08:00:00.000Z")))
        .with_progress(Box::new(move |stage, summary| {
            sink.lock()
                .unwrap()
                .push((stage.to_string(), summary.clone()));
        }));
    (ctx, reported)
}

/// Draws the `train` and `replay` seeds as a command would; the seeds and the
/// run's record of them.
fn draw_both(
    test: &str,
    train: Option<Seed>,
    replay: Option<Seed>,
) -> ([Seed; 2], BTreeMap<String, u64>, Vec<serde_json::Value>) {
    let (ctx, reported) = context(test);
    let mut used = None;
    let recorded = record(&ctx, "train", &json!({}), |run| {
        used = Some([run.seed("train", train)?, run.seed("replay", replay)?]);
        Ok(json!({}))
    })
    .unwrap();
    let seeds = read_run(ctx.workspace(), &recorded.run).unwrap().seeds;
    let seed_reports = reported
        .lock()
        .unwrap()
        .iter()
        .filter(|(stage, _)| stage == "seed")
        .map(|(_, summary)| summary.clone())
        .collect();
    (used.unwrap(), seeds, seed_reports)
}

#[test]
fn seeds_given_explicitly_are_recorded_and_reproduce_the_run() {
    let given = [splinter_model::seed::draw(), splinter_model::seed::draw()];
    let (used, seeds, reports) = draw_both("given", Some(given[0]), Some(given[1]));
    assert_eq!(used, given);
    assert_eq!(
        seeds,
        BTreeMap::from([
            ("train".into(), given[0].value()),
            ("replay".into(), given[1].value())
        ])
    );
    assert_eq!(
        reports,
        [
            json!({ "name": "train", "seed": given[0].value(), "drawn": false }),
            json!({ "name": "replay", "seed": given[1].value(), "drawn": false }),
        ]
    );
}

#[test]
fn seeds_left_out_are_drawn_recorded_and_reported_once_each() {
    let (used, seeds, reports) = draw_both("drawn", None, None);
    assert_eq!(seeds.get("train"), Some(&used[0].value()));
    assert_eq!(seeds.get("replay"), Some(&used[1].value()));
    assert_eq!(reports.len(), 2);
    assert_eq!(
        reports[0],
        json!({ "name": "train", "seed": used[0].value(), "drawn": true })
    );
    assert_eq!(
        reports[1],
        json!({ "name": "replay", "seed": used[1].value(), "drawn": true })
    );

    // Passing what was reported draws nothing and records the same seeds.
    let (again, again_seeds, again_reports) = draw_both("again", Some(used[0]), Some(used[1]));
    assert_eq!(again, used);
    assert_eq!(again_seeds, seeds);
    assert!(again_reports.iter().all(|r| r["drawn"] == false));
}
