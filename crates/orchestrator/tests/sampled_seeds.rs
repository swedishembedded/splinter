// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training and evaluation
// pipelines for its clients. If your team needs expertise in seeding,
// reproducibility and experiment provenance for sampled decoding then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a run owns the seeds its local model drew for sampled requests. The
//! seeds are taken from the model into the run that made the requests, in
//! request order, at every stage boundary and when the work ends; the run
//! record stays bounded however many requests were made, and the whole list
//! can be read back to replay them. Greedy requests draw nothing.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde_json::json;
use splinter_agent::sven::model::{
    CompletionRequest, Message, MessageContent, ResponseEvent, Role,
};
use splinter_core::clock::FixedClock;
use splinter_core::model_ref::ModelRef;
use splinter_model::local::{AGENT_SAMPLING, GREEDY_SAMPLING};
use splinter_model::residency::scripted::ScriptedLoader;
use splinter_model::{Residency, Sampling};
use splinter_orchestrator::runs::{record, Recorder};
use splinter_orchestrator::{Config, Context};
use splinter_store::runs::{read_run, read_sampled_seeds, SAMPLED_FIRST};
use splinter_store::StateRoot;

type Reported = Arc<Mutex<Vec<(String, serde_json::Value)>>>;

/// A context over a scratch state root whose local models run on a scripted
/// base, with every reported stage collected.
fn context(test: &str) -> (Context, Reported, ModelRef) {
    let root = std::env::temp_dir().join(format!("splinter-sampled-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let base = root.join("models/base");
    std::fs::create_dir_all(&base).unwrap();
    std::fs::write(base.join("model.safetensors"), b"scripted").unwrap();
    let config = Config {
        state_root: StateRoot::new(root.join("state")),
        model_store: root.join("models"),
        policy_base: base.clone(),
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
        min_calibration_controls: 2,
        workspace: Default::default(),
        thinking: false,
        served_temperature: None,
        build: Default::default(),
    };
    let reported: Reported = Arc::default();
    let sink = Arc::clone(&reported);
    let ctx = Context::new(config, false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-10-15T08:00:00.000Z")))
        .with_residency(Residency::new(Arc::new(ScriptedLoader::new(|_, _| {
            "answer".into()
        }))))
        .with_progress(Box::new(move |stage, summary| {
            sink.lock()
                .unwrap()
                .push((stage.to_string(), summary.clone()));
        }));
    let reference = ModelRef::Local {
        checkpoint: base.display().to_string(),
        adapter: None,
        context_tokens: Some(4096),
    };
    (ctx, reported, reference)
}

/// `requests` completions asked of the local model `reference` sampling as
/// `sampling` says.
fn ask(ctx: &Context, reference: &ModelRef, sampling: Sampling, requests: usize) {
    let model = ctx.resampled(reference, sampling).unwrap().unwrap();
    for _ in 0..requests {
        let request = CompletionRequest {
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("hello".into()),
            }],
            ..CompletionRequest::default()
        };
        ctx.block_on(async {
            let mut stream = model.provider.complete(request).await.unwrap();
            while let Some(event) = stream.next().await {
                if let ResponseEvent::Error(e) = event.unwrap() {
                    panic!("generation failed: {e}");
                }
            }
        });
    }
}

/// A recorded run of `work`: its id.
fn run(ctx: &Context, work: impl FnOnce(&mut Recorder<'_>)) -> String {
    record(ctx, "sample", &json!({}), |run| {
        work(run);
        Ok(json!({}))
    })
    .unwrap()
    .run
}

#[test]
fn a_run_records_the_seeds_its_local_model_drew_for_the_requests_it_made() {
    let (ctx, reported, reference) = context("drawn");
    let id = run(&ctx, |run| {
        ask(&ctx, &reference, AGENT_SAMPLING, 2);
        run.stage("first", &json!({})).unwrap();
        ask(&ctx, &reference, AGENT_SAMPLING, 1);
    });

    let seeds = read_sampled_seeds(ctx.workspace(), &id).unwrap();
    assert_eq!(seeds.len(), 3, "one per sampled request: {seeds:?}");
    assert_ne!(seeds[0], seeds[1]);

    let record = read_run(ctx.workspace(), &id).unwrap().sampled;
    assert_eq!(record.count, 3);
    assert_eq!(
        record.first,
        seeds.iter().map(|s| s.value()).collect::<Vec<_>>()
    );
    assert!(record.digest.is_some());

    let sampled: Vec<_> = reported
        .lock()
        .unwrap()
        .iter()
        .filter(|(stage, _)| stage == "sampled")
        .map(|(_, summary)| summary["count"].as_u64().unwrap())
        .collect();
    assert_eq!(
        sampled,
        [2, 3],
        "reported at the stage boundary and at the end, as the record stood"
    );
}

#[test]
fn the_run_record_stays_bounded_and_the_whole_list_stays_readable() {
    let (ctx, _, reference) = context("bounded");
    let requests = SAMPLED_FIRST + 5;
    let id = run(&ctx, |_| ask(&ctx, &reference, AGENT_SAMPLING, requests));
    let record = read_run(ctx.workspace(), &id).unwrap().sampled;
    assert_eq!(record.count, requests as u64);
    assert_eq!(record.first.len(), SAMPLED_FIRST);
    let all = read_sampled_seeds(ctx.workspace(), &id).unwrap();
    assert_eq!(all.len(), requests);
    assert_eq!(
        record.first,
        all[..SAMPLED_FIRST]
            .iter()
            .map(|s| s.value())
            .collect::<Vec<_>>()
    );
}

#[test]
fn seeds_drawn_outside_a_run_or_in_another_are_not_this_runs() {
    let (ctx, _, reference) = context("attribution");
    ask(&ctx, &reference, AGENT_SAMPLING, 4);
    let first = run(&ctx, |_| ask(&ctx, &reference, AGENT_SAMPLING, 1));
    let second = run(&ctx, |_| ask(&ctx, &reference, AGENT_SAMPLING, 2));
    assert_eq!(
        read_sampled_seeds(ctx.workspace(), &first).unwrap().len(),
        1
    );
    assert_eq!(
        read_sampled_seeds(ctx.workspace(), &second).unwrap().len(),
        2
    );
}

#[test]
fn greedy_decoding_and_named_seeds_leave_nothing_to_record() {
    let (ctx, _, reference) = context("greedy");
    let id = run(&ctx, |_| {
        ask(&ctx, &reference, GREEDY_SAMPLING, 2);
        ask(
            &ctx,
            &reference,
            Sampling {
                seed: Some(splinter_model::seed::draw().value()),
                ..AGENT_SAMPLING
            },
            1,
        );
    });
    assert!(read_sampled_seeds(ctx.workspace(), &id).unwrap().is_empty());
    assert!(read_run(ctx.workspace(), &id).unwrap().sampled.is_empty());
}
