// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from what agents
// actually did, every record traceable to the attempt it came from, for its
// clients. If your team needs expertise in preference and reinforcement
// learning from agent experience, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The objectives other than SFT: attempts projected by splinter-data's own
//! views over the trials' corpus ([`super::super::corpus`]), so a dataset
//! here is the one the same view makes from the experience database.
//!
//! * `preference` ([`Preference`]): a verified pass over a verified fail of
//!   one task by one policy, from where their conversations part: the shared
//!   prompt never trained on, each continuation to its last action a
//!   candidate (`generic-preference-v1`). `preference.pairs_per_task`
//!   bounds the pairs of one task, drawn by the stage's seed.
//! * `feedback` ([`Feedback`]): each verified attempt's continuation of its
//!   prompt, desirable when it passed and undesirable when it failed, for
//!   KTO (`generic-feedback-v1`); no counterpart is needed.
//! * `policy-gradient` ([`PolicyGradient`]): every action of every graded
//!   attempt, each record carrying its attempt's reward and the group of
//!   attempts at the same task it is compared with (`generic-messages-v2`
//!   with `reward` and `group`); a group whose attempts all earned the same
//!   says nothing and is left out.
//!
//! Every trial that did not become an experience is listed in
//! `excluded.jsonl` with why; what a view left out is counted by its reason
//! in `dataset.json`. Validation holds out whole groups as SFT does; a record
//! of several trials (a pair) goes with its first trial's group, and a split
//! that puts one task's trials on both sides is refused.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::json;
use splinter_core::annotation::Strength;
use splinter_core::factory::product::{DatasetProduct, WaterfallStep, DATASET_SCHEMA};
use splinter_data::{Feedback, Objective, PolicyGradient, Preference, Record, RecordBody, View};
use splinter_store::factory::products::Staging;

use super::super::corpus::corpus_of;
use super::super::rollout::trials_of;
use super::super::{io, stage_error, write_json, write_lines, Factory, RunError};
use super::{file_ref, group_of, held_out, Accepted, Dataset, Excluded, DATASET_FILE};
use crate::plan::PlannedStage;

/// The view `dataset`'s objective is projected by.
fn view_of(dataset: &Dataset) -> Result<(Box<dyn View>, Objective), String> {
    // A trial's grade is the executable verifier's verdict.
    let strength = Strength::Executable;
    match dataset.objective.as_str() {
        "preference" => {
            let mut view = Preference::new(strength);
            if let Some(cap) = dataset.preference.as_ref().and_then(|p| p.pairs_per_task) {
                view = view.pairs_per_group(cap, dataset.seed);
            }
            Ok((Box::new(view), Objective::Preference))
        }
        "feedback" => Ok((Box::new(Feedback::new(strength)), Objective::Feedback)),
        "policy-gradient" => Ok((
            Box::new(PolicyGradient::new(strength)),
            Objective::PolicyGradient,
        )),
        other => Err(format!("objective {other} has no view here")),
    }
}

/// Takes every turn's reasoning out of `record`: a target trained for
/// no-think mode is never shown any. How many turns held some.
fn without_reasoning(record: &mut Record) -> u64 {
    let turns: Vec<&mut splinter_core::chat::WireMessage> = match &mut record.body {
        RecordBody::Chat { messages, .. } => messages.iter_mut().collect(),
        RecordBody::Preference {
            prompt,
            chosen,
            rejected,
            ..
        } => prompt
            .iter_mut()
            .chain(chosen.iter_mut())
            .chain(rejected.iter_mut())
            .collect(),
        RecordBody::Feedback {
            prompt, completion, ..
        } => prompt.iter_mut().chain(completion.iter_mut()).collect(),
        _ => Vec::new(),
    };
    let mut removed = 0;
    for turn in turns {
        if turn.reasoning_content.take().is_some() {
            removed += 1;
        }
    }
    removed
}

/// The assistant turns of `record` it trains on.
fn supervised(record: &Record) -> usize {
    match &record.body {
        RecordBody::Chat { messages, .. } => messages.iter().filter(|m| m.train).count(),
        RecordBody::Preference { chosen, .. } => {
            chosen.iter().filter(|m| m.role == "assistant").count()
        }
        RecordBody::Feedback { completion, .. } => {
            completion.iter().filter(|m| m.role == "assistant").count()
        }
        _ => 0,
    }
}

pub(super) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
    dataset: &Dataset,
) -> Result<(), RunError> {
    let fail = |why: String| stage_error(&stage.key, why);
    let (view, objective) = view_of(dataset).map_err(fail)?;
    let mut held = Vec::new();
    let mut sources = Vec::new();
    for source in &dataset.sources {
        sources.push(factory.product(source)?.1.digest());
        let at = factory.view(source);
        let trials = trials_of(&at)?;
        held.push((source.clone(), at, trials));
    }
    let raw: usize = held.iter().map(|(_, _, t)| t.len()).sum();
    let trials = corpus_of(&held).map_err(fail)?;
    let mut projection = view
        .project(&trials.corpus)
        .map_err(|e| fail(e.to_string()))?;
    let mut excluded: Vec<Excluded> = trials
        .skipped
        .iter()
        .map(|(source, trial, why)| Excluded {
            pass: source.clone(),
            slot: trial.slot.to_string(),
            task: trial.task.clone(),
            reason: "no_experience".into(),
            detail: why.clone(),
        })
        .collect();
    excluded.sort_by(|a, b| (&a.pass, &a.slot).cmp(&(&b.pass, &b.slot)));
    let mut reasoning_removed = 0;
    if !dataset.thinking {
        for record in &mut projection.records {
            reasoning_removed += without_reasoning(record);
        }
    }
    if projection.records.is_empty() {
        return Err(fail(format!(
            "no record of {raw} trials; the {} view left out {:?}",
            view.name(),
            projection.excluded
        )));
    }
    // Each record's trials, in the order the record names their experiences.
    let of = |record: &Record| -> Vec<&(String, splinter_core::factory::product::TrialResult)> {
        record
            .metadata
            .experiences
            .iter()
            .filter_map(|id| trials.trials.get(id))
            .collect()
    };
    let group = |record: &Record| {
        of(record)
            .first()
            .map(|(_, trial)| group_of(factory, dataset, &trial.task))
            .unwrap_or_default()
    };
    let groups: BTreeSet<String> = projection.records.iter().map(group).collect();
    let validation_groups = dataset
        .validation
        .as_ref()
        .map(|v| held_out(&groups, v.percent, dataset.seed))
        .unwrap_or_default();
    let (mut train, mut validation) = (Vec::new(), Vec::new());
    let (mut accepted, mut supervision) = (Vec::new(), Vec::new());
    let mut sides: BTreeMap<String, BTreeSet<&str>> = BTreeMap::new();
    for record in &projection.records {
        let split = if validation_groups.contains(&group(record)) {
            "validation"
        } else {
            "train"
        };
        let line =
            splinter_data::dataset_line(objective, record).map_err(|e| fail(e.to_string()))?;
        let lines = if split == "train" {
            &mut train
        } else {
            &mut validation
        };
        let index = lines.len();
        lines.push(line);
        let made_of = of(record);
        for (source, trial) in &made_of {
            sides
                .entry(group_of(factory, dataset, &trial.task))
                .or_default()
                .insert(split);
            accepted.push(Accepted {
                pass: source.clone(),
                slot: trial.slot.to_string(),
                task: trial.task.clone(),
                agent: format!(
                    "{}{} {}",
                    if trial.guided { "guided " } else { "" },
                    trial.agent.name,
                    trial.agent.version.as_deref().unwrap_or("?")
                ),
                model: trial.model.clone(),
                actions: supervised(record),
                split: split.into(),
                guided: trial.guided,
            });
        }
        let named: Vec<_> = made_of
            .iter()
            .map(|(source, trial)| json!({ "pass": source, "slot": trial.slot.to_string(), "task": trial.task }))
            .collect();
        supervision.push(json!({ "split": split, "line": index, "trials": named }));
    }
    let leaked: Vec<&String> = sides
        .iter()
        .filter(|(_, s)| s.len() > 1)
        .map(|(g, _)| g)
        .collect();
    if !leaked.is_empty() {
        return Err(fail(format!("groups in both splits: {leaked:?}")));
    }
    let write_text = |name: &str, lines: &[String]| -> Result<(), RunError> {
        let path = staging.path().join(name);
        let mut text = lines.join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        std::fs::write(&path, text).map_err(io(&path))
    };
    write_text("train.jsonl", &train)?;
    write_text("validation.jsonl", &validation)?;
    write_lines(&staging.path().join("accepted.jsonl"), &accepted)?;
    write_lines(&staging.path().join("excluded.jsonl"), &excluded)?;
    write_lines(&staging.path().join("supervision.jsonl"), &supervision)?;
    let mut exclusions: BTreeMap<String, u64> = BTreeMap::new();
    for e in &excluded {
        *exclusions.entry(e.reason.clone()).or_default() += 1;
    }
    for (reason, count) in &projection.excluded {
        let name = serde_json::to_value(reason)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| format!("{reason:?}"));
        *exclusions.entry(name).or_default() += *count as u64;
    }
    let waterfall = vec![
        WaterfallStep {
            step: "raw_trials".into(),
            remaining: raw as u64,
        },
        WaterfallStep {
            step: "experiences".into(),
            remaining: trials.trials.len() as u64,
        },
        WaterfallStep {
            step: "records".into(),
            remaining: projection.records.len() as u64,
        },
    ];
    let product = DatasetProduct {
        schema: DATASET_SCHEMA.into(),
        name: dataset.name.clone(),
        objective: dataset.objective.clone(),
        projection: view.name().to_owned(),
        admission: "verified".into(),
        target: dataset.target.to_string(),
        sources,
        train: file_ref(staging, "train.jsonl")?,
        validation: file_ref(staging, "validation.jsonl")?,
        accepted: file_ref(staging, "accepted.jsonl")?,
        excluded: file_ref(staging, "excluded.jsonl")?,
        supervision: file_ref(staging, "supervision.jsonl")?,
        waterfall: waterfall.clone(),
        exclusions: exclusions.clone(),
        thinking: dataset.thinking,
    };
    write_json(&staging.path().join(DATASET_FILE), &product)?;
    let mut report = format!(
        "# {}\n\n{} records by the {} view ({} train, {} validation) from {raw} trials.\n\n",
        dataset.name,
        dataset.objective,
        view.name(),
        train.len(),
        validation.len()
    );
    if !dataset.thinking && reasoning_removed > 0 {
        report.push_str(&format!(
            "No-think mode: the reasoning of {reasoning_removed} turns was taken out.\n\n"
        ));
    }
    report.push_str("| Left out | Count |\n|---|---|\n");
    for (reason, count) in &exclusions {
        report.push_str(&format!("| {reason} | {count} |\n"));
    }
    std::fs::write(staging.path().join("report.md"), report).map_err(io(staging.path()))
}
