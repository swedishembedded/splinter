// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training sequences that take a model from
// random weights through pretraining, fine-tuning and learning from its own
// experience, for its clients. If your team needs expertise in staged
// small-model training, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The sequence one model goes through over the shared data stages: random
//! init, pretraining, fine-tuning, exploration, improvement and the
//! examination, each probed.

use std::collections::BTreeMap;

use anyhow::Context as _;
use serde_json::Value;
use splinter_sdk::model::scratch::Checkpoint;
use splinter_sdk::store::stage_ledger::Commit;

use crate::stages::{Run, Stage};
use crate::template::Template;
use crate::world;

/// Chat records as text documents in the template, one per conversation:
/// of the records of one episode (each holds the conversation up to a call
/// and its reply) the one with the most messages, which holds them all.
pub(crate) fn conversations_as_text(template: &Template, records: &str) -> anyhow::Result<String> {
    let mut last: BTreeMap<String, Value> = BTreeMap::new();
    for line in records.lines() {
        let record: Value = serde_json::from_str(line)?;
        let episode = record["metadata"]["episode"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let len = |r: &Value| r["messages"].as_array().map_or(0, Vec::len);
        if last
            .get(&episode)
            .is_none_or(|kept| len(&record) > len(kept))
        {
            last.insert(episode, record);
        }
    }
    let docs: Vec<world::Document> = last
        .values()
        .map(|r| world::Document {
            kind: "conversation",
            text: template.render_messages(
                r["messages"].as_array().map_or(&[][..], Vec::as_slice),
                r["tools"].as_array().map_or(&[][..], Vec::as_slice),
            ),
        })
        .collect();
    Ok(world::to_jsonl(&docs))
}

/// The model stages of the model labelled `label`.
pub(crate) fn model_stages(
    run: &mut Run<'_>,
    label: &str,
    world: &Commit,
    demonstrations: &Commit,
) -> anyhow::Result<()> {
    let corpus = world
        .output("train")
        .context("the world has no training corpus")?
        .clone();
    let init = run.checkpoint(label, world)?;
    run.probe(&format!("{label}/init"), (&init.0, &init.1))?;
    run.retrieval(&format!("{label}/init"), (&init.0, &init.1))?;
    // Pretraining in segments, each continuing the last, with the retrieval
    // probe after each; the last is the pretrained checkpoint.
    let segments = run.config.pretrain_segments.max(1);
    let full = run.config.pretrain.clone();
    let mut pretrained = init.clone();
    for i in 1..=segments {
        run.config.pretrain.steps = full.steps / segments + u32::from(i <= full.steps % segments);
        let name = if i == segments {
            "pretrain".to_string()
        } else {
            format!("pretrain-{i}")
        };
        let trained = run.train(
            Stage::Pretrain,
            label,
            &name,
            (&pretrained.0, &pretrained.1),
            (&corpus, "world/train.jsonl"),
        );
        run.config.pretrain = full.clone();
        pretrained = trained?;
        run.retrieval(&format!("{label}/{name}"), (&pretrained.0, &pretrained.1))?;
    }
    run.probe(
        &format!("{label}/pretrained"),
        (&pretrained.0, &pretrained.1),
    )?;
    let sat_pretrained = run.sit(label, "pretrained", (&pretrained.0, &pretrained.1), false)?;
    let demos = demonstrations
        .output("records")
        .context("the demonstrations have no records")?
        .clone();
    let sft = run.train(
        Stage::Sft,
        label,
        "sft",
        (&pretrained.0, &pretrained.1),
        (&demos, "data/demonstrate.jsonl"),
    )?;
    run.probe(&format!("{label}/sft"), (&sft.0, &sft.1))?;
    let sat_sft = run.sit(label, "sft", (&sft.0, &sft.1), false)?;
    run.judge(
        label,
        "scratch",
        sat_pretrained.as_ref(),
        sat_sft.as_ref(),
        None,
    )?;
    let (explored, episodes) = run.explore(label, (&sft.0, &sft.1))?;
    let accepted = explored.summary["accepted"].as_u64().unwrap_or(0);
    if accepted == 0 {
        writeln!(
            run.out,
            "[ skip ] {label}/improve: no verified episode of its own to learn from"
        )?;
        return Ok(());
    }
    let mixed = run.experience(label, "improve", &episodes, demonstrations, false)?;
    let improved = train_on(run, label, "improve", (&sft.0, &sft.1), &mixed)?;
    run.probe(&format!("{label}/improved"), (&improved.0, &improved.1))?;
    let sat_improved = run.sit(label, "improved", (&improved.0, &improved.1), false)?;
    // The spurious-reward control arm exists for the examination alone.
    let sat_control = if run.examiner.is_some() {
        let mixed = run.experience(label, "control", &episodes, demonstrations, true)?;
        let control = train_on(run, label, "control", (&sft.0, &sft.1), &mixed)?;
        run.sit(label, "control", (&control.0, &control.1), true)?
    } else {
        None
    };
    run.judge(
        label,
        "experience",
        sat_sft.as_ref(),
        sat_improved.as_ref(),
        sat_control.as_ref(),
    )?;
    Ok(())
}

/// An improvement run named `name` from `base` on the mix `data` made.
fn train_on(
    run: &mut Run<'_>,
    label: &str,
    name: &str,
    base: (&Commit, &Checkpoint),
    data: &Commit,
) -> anyhow::Result<(Commit, Checkpoint)> {
    let digest = data
        .output("records")
        .context("the mix has no records")?
        .clone();
    let rel = format!("data/{label}/{name}.jsonl");
    run.train(Stage::Improve, label, name, base, (&digest, &rel))
}
