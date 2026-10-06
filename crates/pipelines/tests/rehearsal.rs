// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning that rehearses a model's own
// answers so new training does not cost it what it could already do, for
// its clients. If your team needs expertise in continual learning without
// catastrophic forgetting, you can procure our services by sending an email
// to info@swedishembedded.com.

//! Spec: the rehearsal set is the base's own answers to general tasks, holds
//! nothing the anchor suite holds, and a training run mixes it in at its
//! share and monitors on it.
//!
//! * The anchor suite is the gate's frozen check: no anchor task, nor a near
//!   copy of one - the same text in other case and spacing, a close
//!   rewording, the task's words inside a longer prompt - is ever in the
//!   set, whether the base model proposed it or the form generator built
//!   it; the exclusions are counted.
//! * A form task the base answered wrong is left out; an answer no verifier
//!   grades is kept as the base gave it.
//! * `train --rehearsal` hands the trainer the rehearsed records at their
//!   share, never held out, and a monitoring share of them beside the
//!   training families' monitoring records; the candidate and its release
//!   record what was rehearsed, and the leakage check reads it. A share
//!   outside (0, 1), or one that with the replay leaves the new records
//!   nothing, is refused before anything trains.
//! * A `learn` with a persona rehearses by default; `--rehearsal 0` and a
//!   run without a persona do not.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::gate::{anchor_file, dataset, gate_context, put_base, Brain, FakeTrainer, ANCHOR};
use common::{Scratch, Scripted};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_knowledge::rehearsal::{form_tasks, GENERAL_BRIEF};
use splinter_knowledge::tasks::dedup::Seen;
use splinter_knowledge::tasks::{DEFAULT_MAX_OVERLAP, DEFAULT_SHINGLE_WORDS};
use splinter_orchestrator::Context;
use splinter_pipelines::learn::{learn, LearnRequest, Learned, DEFAULT_REHEARSAL_SHARE, STAGES};
use splinter_pipelines::rehearsal::{rehearse, RehearseRequest, FORM_SHARE, REHEARSAL_SEED};
use splinter_pipelines::release::{anchor, arm};
use splinter_pipelines::train::{train, Rehearse, TrainRequest, Tuning, DEFAULT_REPLAY_FRACTION};

/// A general request the base proposes that is new.
const FRESH: [&str; 6] = [
    "Explain how a bicycle pump raises the pressure in a tyre.",
    "Compare renting and buying a home for someone who moves every three years.",
    "Write a short note thanking a neighbour for watering the plants.",
    "What are the steps to change a flat tyre on a car?",
    "How does compound interest work on a savings account?",
    "Give three tips for keeping basil alive on a kitchen windowsill.",
];

/// The base as the specs script it: it writes general requests when asked
/// for them - the fresh ones, and copies and near copies of anchor tasks -
/// answers a format request with as many lines as it asks (right), a sum
/// with no number (wrong), and anything else with a sentence.
fn scripted_base(anchor_copies: Vec<String>) -> Scripted {
    // The prompt reaches the script with its quotes escaped, so the brief
    // is known by a phrase of it that has none.
    let brief_phrase = GENERAL_BRIEF
        .split(',')
        .next()
        .unwrap()
        .replace("You write ", "");
    Scripted::new(move |prompt| {
        if prompt.contains(&brief_phrase) {
            let mut prompts: Vec<String> = anchor_copies.clone();
            prompts.extend(FRESH.iter().map(|p| p.to_string()));
            return serde_json::json!({ "prompts": prompts }).to_string();
        }
        if prompt.contains("on a line of its own")
            || prompt.contains("one per line")
            || prompt.contains("on its own line")
        {
            let asked = ["two", "three", "four", "five", "six", "seven"]
                .iter()
                .position(|word| prompt.contains(&format!(" {word} ")))
                .map_or(3, |at| at + 2);
            return vec!["Alpha"; asked].join("\n");
        }
        if prompt.contains("Answer with the number") || prompt.contains("Work it out") {
            return "I would rather not say.".into();
        }
        "Here is my answer, in a sentence.".into()
    })
}

/// A gate context whose base answers as [`scripted_base`] says, with the
/// anchor suite frozen from the fixtures' facts, `extra_anchor` lines and
/// the first form task the generator itself builds.
fn rehearsal_context(test: &str, extra_anchor: &[String]) -> (Scratch, Context, Vec<String>) {
    let (scratch, ctx) = gate_context(test, Brain::Honest);
    let base = arm(ctx.config(), None);
    // The anchor holds an instance of the generator's own: it must be kept
    // out although the generator would build it again.
    let own = form_tasks(REHEARSAL_SEED, 1)[0].clone();
    let own_reference = own.privileged[0].content.clone();
    let path = anchor_file(&scratch.0, 4);
    let mut text = std::fs::read_to_string(&path).unwrap();
    for line in extra_anchor {
        text.push_str(&serde_json::json!({ "instruction": line, "reference": "x" }).to_string());
        text.push('\n');
    }
    text.push_str(
        &serde_json::json!({ "instruction": own.instruction, "reference": own_reference, "kind": own.task.kind })
            .to_string(),
    );
    text.push('\n');
    std::fs::write(&path, text).unwrap();
    anchor::freeze(&ctx, &[path]).unwrap();
    let anchor_instructions: Vec<String> = anchor::current(&ctx)
        .unwrap()
        .unwrap()
        .suite
        .tasks
        .iter()
        .map(|t| t.instruction.clone())
        .collect();
    let copies = vec![
        // The anchor fact, verbatim.
        anchor_instructions[0].clone(),
        // In other case and spacing.
        format!("  {}  ", anchor_instructions[1].to_uppercase()),
        // Reworded closely.
        format!("{} Please.", anchor_instructions[2]),
        // Inside a longer prompt.
        format!("Quick one before we start: {}", anchor_instructions[3]),
    ];
    ctx.add_model(
        base,
        Model::new(Arc::new(scripted_base(copies)), "scripted/base"),
    );
    (scratch, ctx, anchor_instructions)
}

fn leaks_anchor(anchor_instructions: &[String], prompt: &str) -> bool {
    let mut seen = Seen::new(DEFAULT_SHINGLE_WORDS, DEFAULT_MAX_OVERLAP);
    for instruction in anchor_instructions {
        seen.admit(instruction);
    }
    seen.leaks(prompt).is_some() || seen.encloses(prompt)
}

#[test]
fn the_rehearsal_set_never_holds_an_anchor_task_or_a_near_copy_of_one() {
    // One of the requests the base will propose is an anchor task itself.
    let (_scratch, ctx, anchor_instructions) =
        rehearsal_context("rehearsal-anchor", &[FRESH[4].to_string()]);
    let rehearsed = rehearse(
        &ctx,
        &RehearseRequest {
            records: 12,
            seed: REHEARSAL_SEED,
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap();
    assert_eq!(rehearsed.base, "scripted/base");
    assert!(rehearsed.anchor.is_some(), "the anchor in force is named");
    // Half the set is form tasks, the rest general requests; the generator's
    // own instance that the anchor holds was passed over, as were the four
    // copies the base proposed and the request that is an anchor task -
    // every time the base proposed them again - and the base's five other
    // requests were admitted once each, so the set stopped short.
    assert_eq!(rehearsed.form_tasks, (12.0 * FORM_SHARE).ceil() as usize);
    assert_eq!(
        rehearsed.tasks,
        rehearsed.form_tasks + rehearsed.general_tasks
    );
    assert!(rehearsed.excluded_as_anchor >= 6, "{rehearsed:#?}");
    assert_eq!(rehearsed.general_tasks, FRESH.len() - 1);
    assert!(
        rehearsed.repeated > 0 && rehearsed.stopped.is_some(),
        "{rehearsed:#?}"
    );
    let tasks = ctx.tasks().get_set(&rehearsed.task_set).unwrap();
    for entry in &tasks.members {
        let task = ctx.tasks().get(&entry.task).unwrap();
        assert!(
            !leaks_anchor(&anchor_instructions, &task.instruction),
            "an anchor task reached the set: {}",
            task.instruction
        );
        assert!(
            !anchor_instructions.contains(&task.instruction),
            "{}",
            task.instruction
        );
    }
    // The dataset holds the base's answers under the default prompt: every
    // format request answered with three lines and every general request,
    // and none of the sums, which the base answered with no number.
    assert!(rehearsed.verified.failed > 0, "{:#?}", rehearsed.verified);
    let text = std::fs::read_to_string(&rehearsed.dataset.path).unwrap();
    let records: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(records.len(), rehearsed.dataset.records);
    assert_eq!(rehearsed.dataset.view, "rehearsal");
    for record in &records {
        let messages = record["messages"].as_array().unwrap();
        assert_eq!(
            messages[0]["content"],
            splinter_core::prompt::SYSTEM_PROMPT,
            "the default prompt, not a persona's"
        );
        let prompt = messages[1]["content"].as_str().unwrap();
        assert!(!leaks_anchor(&anchor_instructions, prompt), "{prompt}");
        assert!(
            !prompt.contains("Work it out") && !prompt.contains("Answer with the number"),
            "a sum the base got wrong is not rehearsed: {prompt}"
        );
        assert_eq!(messages[2]["train"], true);
        assert!(
            !record["metadata"]["experiences"]
                .as_array()
                .unwrap()
                .is_empty(),
            "the base's answer is the experience behind it"
        );
    }
    assert!(
        records.iter().any(|r| r["messages"][2]["content"]
            .as_str()
            .unwrap()
            .starts_with("Alpha")),
        "a format request the base answered right is rehearsed"
    );
    assert!(
        records
            .iter()
            .any(|r| r["messages"][2]["content"] == "Here is my answer, in a sentence."),
        "an answer no verifier grades is rehearsed as given"
    );
}

#[test]
fn a_train_with_a_rehearsal_mixes_it_in_at_its_share_and_monitors_on_it() {
    let (_scratch, ctx, _) = rehearsal_context("rehearsal-train", &[]);
    let rehearsed = rehearse(
        &ctx,
        &RehearseRequest {
            records: 24,
            seed: REHEARSAL_SEED,
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap();
    let data = dataset(&ctx, "alpha", 40);
    let request = |rehearsal: Option<Rehearse>| TrainRequest {
        datasets: vec![data.to_string()],
        rehearsal,
        from: common::gate::policy(),
        replay_fraction: DEFAULT_REPLAY_FRACTION,
        steps: None,
        rank: 4,
        beta: None,
        tuning: Tuning::default(),
    };
    let trainer = FakeTrainer::knowing(&[ANCHOR, "alpha"]);
    let candidate = train(
        &ctx,
        &request(Some(Rehearse {
            dataset: rehearsed.dataset.dataset.to_string(),
            share: 0.3,
        })),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    let plan = trainer.plans.lock().unwrap()[0].clone();
    let rehearsal = plan
        .rehearsal
        .as_ref()
        .expect("the plan carries the rehearsal");
    assert_eq!(rehearsal.share, 0.3);
    assert_eq!(rehearsal.dataset.id, rehearsed.dataset.dataset);
    let count = |path: &std::path::Path| {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count()
    };
    // Split as the training families are: a monitoring share set aside,
    // the rest mixed in; together, every record. (The working directory is
    // gone once the candidate is kept; the records mixed in are an
    // artifact.)
    assert!(rehearsal.monitor.is_some(), "{rehearsal:?}");
    assert!(rehearsal.monitored >= 1, "{rehearsal:?}");
    assert_eq!(
        rehearsal.trained + rehearsal.monitored,
        rehearsed.dataset.records
    );
    let sample = candidate
        .rehearsal
        .as_ref()
        .expect("the candidate records what it rehearsed");
    assert_eq!(
        (sample.trained, sample.monitored, sample.share),
        (rehearsal.trained, rehearsal.monitored, 0.3)
    );
    assert_eq!(sample.dataset, rehearsed.dataset.dataset);
    let kept = ctx
        .artifacts()
        .path(sample.digest.as_ref().unwrap())
        .unwrap();
    assert_eq!(
        count(&kept),
        rehearsal.trained,
        "the rehearsed records are kept"
    );
    // The gate's leakage check reads them: an anchor task that were among
    // them would be left out of the suite, which is what makes the
    // admission's exclusion load-bearing.
    let released = common::gate::decide(&ctx, &candidate);
    let anchor_check = released.gate.anchor.measured.as_ref().unwrap();
    assert_eq!(
        anchor_check.suite.excluded.get("leaked"),
        None,
        "{anchor_check:#?}"
    );
    let manifest = ctx
        .releases()
        .get(released.release.as_ref().unwrap())
        .unwrap()
        .manifest;
    assert_eq!(manifest.rehearsal.as_ref(), Some(sample));

    // Nothing rehearsed: no plan entry, no record.
    let trainer = FakeTrainer::knowing(&[ANCHOR, "alpha"]);
    let plain = train(&ctx, &request(None), &trainer, &CancelToken::new()).unwrap();
    assert!(trainer.plans.lock().unwrap()[0].rehearsal.is_none());
    assert!(plain.rehearsal.is_none());

    // A share that is not a share, or that leaves the new records nothing,
    // is refused before anything trains.
    for share in [0.0, 1.0, 1.5] {
        let trainer = FakeTrainer::knowing(&[ANCHOR, "alpha"]);
        let refused = train(
            &ctx,
            &request(Some(Rehearse {
                dataset: rehearsed.dataset.dataset.to_string(),
                share,
            })),
            &trainer,
            &CancelToken::new(),
        )
        .unwrap_err();
        assert!(
            refused.is_refusal() && refused.to_string().contains("rehearsal share"),
            "{refused}"
        );
        assert!(trainer.plans.lock().unwrap().is_empty());
    }
}

#[test]
fn a_learn_with_a_persona_rehearses_by_default_and_not_when_told_not_to() {
    let (scratch, ctx) = gate_context("rehearsal-learn-plan", Brain::Honest);
    put_base(&ctx);
    let manual = scratch.0.join("manual.md");
    std::fs::write(&manual, common::manual::MANUAL).unwrap();
    let stages = |persona: Option<&str>, rehearsal: Option<f64>| {
        let learned = learn(
            &ctx,
            &LearnRequest {
                sources: vec![manual.display().to_string()],
                persona: persona.map(str::to_string),
                rehearsal,
                dry_run: true,
                ..LearnRequest::default()
            },
            &FakeTrainer::knowing(&[]),
        )
        .unwrap();
        let Learned::Planned(plan) = learned else {
            panic!("a dry run plans");
        };
        plan.stages
    };
    assert!(STAGES.contains(&"rehearse"));
    assert!(stages(Some("The Writer"), None).contains(&"rehearse"));
    assert!(!stages(Some("The Writer"), Some(0.0)).contains(&"rehearse"));
    assert!(!stages(None, None).contains(&"rehearse"));
    assert!(stages(None, Some(DEFAULT_REHEARSAL_SHARE)).contains(&"rehearse"));
    let refused = learn(
        &ctx,
        &LearnRequest {
            sources: vec![manual.display().to_string()],
            rehearsal: Some(1.0),
            dry_run: true,
            ..LearnRequest::default()
        },
        &FakeTrainer::knowing(&[]),
    )
    .unwrap_err();
    assert!(refused.is_refusal() && refused.to_string().contains("rehearsal share"));
}
