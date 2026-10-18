// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that keep a model server's
// decode slots full while they measure a candidate, for its clients. If your
// team needs expertise in batched evaluation of served language models, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a model that decodes several requests together is asked as many of a
//! suite's tasks at once as it has slots - the release gate's probing and the
//! serve check against a `brain serve` alike - and the outcomes keep the
//! order of the suite's tasks however the answers came in.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::time::Duration;

use common::fake_brain::slots_log;
use common::gate::{answer, fact, gate_context, Brain};
use common::{scratch_context, Counted, Scripted};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_pipelines::brain_server::DEFAULT_SERVE_SLOTS;
use splinter_pipelines::release::probe::{grade, Probe, Suite};
use splinter_pipelines::release::serve::{check, ServeCheck};

const TASKS: usize = 10;

/// A model that knows every fact of the `alpha` topic, answering the earlier
/// questions later than the later ones.
fn knowing_alpha() -> Scripted {
    Scripted::new(|prompt| {
        let number = prompt
            .split("code number ")
            .nth(1)
            .map(|rest| {
                rest.chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>()
            })
            .and_then(|digits| digits.parse::<usize>().ok())
            .unwrap_or(0);
        answer("alpha", number)
    })
}

fn pause_by_task(prompt: &str) -> Duration {
    let number = prompt
        .split("code number ")
        .nth(1)
        .map(|rest| {
            rest.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
        })
        .and_then(|digits| digits.parse::<u64>().ok())
        .unwrap_or(0);
    Duration::from_millis((TASKS as u64 - number.min(TASKS as u64)) * 10 + 5)
}

#[test]
fn the_gate_asks_a_batching_model_as_many_tasks_at_once_as_it_has_slots_and_keeps_their_order() {
    let (_scratch, ctx) = scratch_context("probe-slots", Scripted::new(|_| String::new()), false);
    let suite = Suite::of_tasks("held-out", (0..TASKS).map(|i| fact("alpha", i)).collect());
    for slots in [1, 4] {
        let busy = Counted::new(knowing_alpha(), pause_by_task);
        let model = Model::new(busy.clone(), "scripted/batching").with_slots(slots);
        let probes = grade(&ctx, &model, &suite, &CancelToken::new()).unwrap();
        assert_eq!(busy.most(), slots, "{slots} slots kept full");
        let answers: Vec<_> = probes.iter().map(|p| p.answer.clone().unwrap()).collect();
        let expected: Vec<_> = (0..TASKS).map(|i| answer("alpha", i)).collect();
        assert_eq!(answers, expected, "outcomes in the order of the tasks");
        assert!(probes.iter().all(|p| p.verdict == Some(true)));
    }
}

#[test]
fn the_serve_check_asks_a_brain_server_as_many_tasks_at_once_as_it_was_told_to_decode() {
    let (scratch, ctx) = gate_context("serve-slots", Brain::Honest);
    let adapter = scratch.0.join("candidate.json");
    let card = br#"{"candidate": "slots", "knows": ["alpha"]}"#;
    std::fs::write(&adapter, card).unwrap();
    // More tasks than the server has slots, so the slots can all be full.
    let tasks = DEFAULT_SERVE_SLOTS as usize + 4;
    std::fs::write(scratch.0.join("pause"), b"").unwrap();
    let sample = Suite::of_tasks("sample", (0..tasks).map(|i| fact("alpha", i)).collect());
    let in_process: Vec<Probe> = (0..tasks)
        .map(|i| Probe::graded(Some(answer("alpha", i)), Some(true)))
        .collect();
    let digest = Digest::sha256_of(card).to_string();
    let checked = check(
        &ctx,
        &ServeCheck {
            binary: ctx.config().brain_binary.as_deref(),
            base: &ctx.config().policy_base,
            adapter: Some(&adapter),
            adapter_digest: &digest,
            system: None,
            sample: &sample,
            in_process: &in_process,
            startup: Duration::from_secs(30),
        },
        &CancelToken::new(),
    )
    .unwrap();
    assert!(checked.passed, "{checked:?}");
    assert_eq!(checked.measured.unwrap().agreed, tasks);
    let log = std::fs::read_to_string(slots_log(&scratch)).unwrap();
    let seen: Vec<(&str, usize)> = log
        .lines()
        .map(|line| {
            let (slots, in_flight) = line.split_once(' ').unwrap();
            (slots, in_flight.parse().unwrap())
        })
        .collect();
    assert!(
        seen.iter()
            .all(|(slots, _)| *slots == DEFAULT_SERVE_SLOTS.to_string()),
        "the server is told its slots: {seen:?}"
    );
    let most = seen.iter().map(|(_, in_flight)| *in_flight).max().unwrap();
    assert_eq!(most, DEFAULT_SERVE_SLOTS as usize, "{seen:?}");
}
