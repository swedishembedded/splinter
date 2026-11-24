// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from what agents
// actually did, every record traceable to the attempt it came from, for its
// clients. If your team needs expertise in preference and reinforcement
// learning from agent experience, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The objectives other than SFT of attempts: attempts and answers
//! projected by splinter-data's own views over one corpus ([`super::super::corpus`]), so a dataset
//! here is the one the same view makes from the experience database.
//!
//! * `sft` of `answers` ([`SftFinal`]): each passed answer's final reply,
//!   the only supervised turn, under its question (an attempt's SFT record,
//!   every action of it supervised, is [`super`]'s).
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
//! * `distill` ([`PolicyGradient`] distilling every attempt): each
//!   attempt's conversation as the student saw it, with a teacher context
//!   beside it (the task's reference solution, the checks that failed),
//!   which only the frozen teacher of a self-distillation run reads.
//! * `retrieval` ([`Retrieval`]): each passed experience's question and
//!   the source passage its evidence names, with the passages other
//!   experiences rest on as negatives (contrastive triples).
//!
//! The corpus is the trials of the `attempts` sources and the experiences
//! `answer.collect` made in process (`answers`: each item's
//! `experience-N.json` with its `verdicts-N.json`), whose items' source
//! stores (`sources/`) are merged into one for the views that read
//! evidence. A view admits what code decided, a formal comparison or an
//! executed check, never a judge's opinion.
//!
//! Every trial that did not become an experience is listed in
//! `excluded.jsonl` with why; what a view left out is counted by its reason
//! in `dataset.json`. Validation holds out whole groups as SFT does; a record
//! of several trials (a pair) goes with its first trial's group, and a split
//! that puts one task's trials on both sides is refused.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::json;
use splinter_core::annotation::Annotation;
use splinter_core::annotation::Strength;
use splinter_core::experience::{Experience, ExperienceId};
use splinter_core::factory::items::{ItemIndex, ITEMS_FILE};
use splinter_core::factory::product::{DatasetProduct, WaterfallStep, DATASET_SCHEMA};
use splinter_data::{
    Corpus, Distill, Feedback, Objective, PolicyGradient, Preference, Record, RecordBody,
    Retrieval, SftFinal, View,
};
use splinter_store::factory::products::Staging;
use splinter_store::sources::SourceStore;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

use super::super::corpus::{corpus_of, Source};
use super::super::rollout::trials_of;
use super::super::{io, stage_error, write_json, write_lines, Factory, RunError};
use super::{file_ref, group_of, held_out, Accepted, Dataset, Excluded, DATASET_FILE};
use crate::plan::PlannedStage;

/// The view `dataset`'s objective is projected by, reading evidence in
/// `sources`.
fn view_of<'s>(
    dataset: &Dataset,
    sources: &'s SourceStore,
) -> Result<(Box<dyn View + 's>, Objective), String> {
    // What code decided: a trial's executed check, an answer's formal
    // comparison; never a judge's opinion.
    let strength = Strength::Formal;
    match dataset.objective.as_str() {
        // Answers made in process: each passed one's final reply.
        "sft" => Ok((Box::new(SftFinal::new(strength)), Objective::Sft)),
        "preference" => {
            let mut view = Preference::new(strength);
            if let Some(cap) = dataset.preference.as_ref().and_then(|p| p.pairs_per_task) {
                view = view.pairs_per_group(cap, dataset.seed);
            }
            Ok((Box::new(view), Objective::Preference))
        }
        "feedback" => Ok((Box::new(Feedback::new(strength)), Objective::Feedback)),
        "policy-gradient" => {
            let penalties = dataset
                .reward
                .iter()
                .flat_map(|r| &r.penalties_percent)
                .map(|(name, percent)| (name.clone(), f64::from(*percent) / 100.0))
                .collect();
            let shape = splinter_data::RewardShape::unshaped()
                .penalising(penalties)
                .map_err(|e| e.to_string())?;
            Ok((
                Box::new(PolicyGradient::new(strength).rewards(shape)),
                Objective::PolicyGradient,
            ))
        }
        // Every attempt, with what only a teacher is shown beside it: the
        // task's reference solution, and the checks that failed.
        "distill" => Ok((
            Box::new(PolicyGradient::new(strength).distilling(Distill::Every)),
            Objective::PolicyGradient,
        )),
        "retrieval" => Ok((
            Box::new(Retrieval::new(strength, sources)),
            Objective::Contrastive,
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

/// The conversations a record of a pair or judged-completion objective is trained as: its prompt followed by
/// each continuation, as chat records the trainer can count.
fn rows_of(record: &Record) -> Vec<Record> {
    let chat =
        |messages: Vec<splinter_core::chat::WireMessage>, tools: &[serde_json::Value]| Record {
            body: RecordBody::Chat {
                messages,
                tools: tools.to_vec(),
                outcome: None,
                teacher_context: Vec::new(),
            },
            metadata: record.metadata.clone(),
        };
    match &record.body {
        RecordBody::Preference {
            prompt,
            chosen,
            rejected,
            tools,
            ..
        } => [chosen, rejected]
            .into_iter()
            .map(|c| chat(prompt.iter().chain(c).cloned().collect(), tools))
            .collect(),
        RecordBody::Feedback {
            prompt,
            completion,
            tools,
            ..
        } => vec![chat(
            prompt.iter().chain(completion).cloned().collect(),
            tools,
        )],
        // A rollout's reward and group are the trainer's to read; the length is that of the conversation.
        RecordBody::Chat {
            messages, tools, ..
        } => vec![chat(messages.clone(), tools)],
        _ => vec![record.clone()],
    }
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

/// Where an experience of the corpus came from.
struct Member {
    /// The stage that holds it.
    source: String,
    /// The trial's slot, or the answer's item and attempt.
    name: String,
    /// The task it attempted: what it is held out with.
    task: String,
    agent: String,
    model: Option<String>,
    guided: bool,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Adds the experiences of the `answer.collect` product `source` at `view`
/// to `corpus`, each with its verdicts, and copies its items' sources into
/// `store`. How many experiences it held.
fn add_answers(
    source: &str,
    view: &std::path::Path,
    corpus: &mut Corpus,
    members: &mut BTreeMap<ExperienceId, Member>,
    store: &SourceStore,
) -> Result<usize, String> {
    let index: ItemIndex = read_json(&view.join(ITEMS_FILE))?;
    let mut held = 0;
    for item in &index.items {
        let dir = view.join(&item.path);
        let sources = dir.join("sources");
        if sources.is_dir() {
            let theirs = SourceStore::new(&Workspace::at(&StateRoot::new(sources)));
            for id in theirs.list().map_err(|e| e.to_string())? {
                store.copy_from(&theirs, &id).map_err(|e| e.to_string())?;
                corpus.add_source(id);
            }
        }
        for attempt in 0.. {
            let path = dir.join(format!("experience-{attempt}.json"));
            if !path.is_file() {
                break;
            }
            let experience: Experience = read_json(&path)?;
            let notes: Vec<Annotation> = read_json(&dir.join(format!("verdicts-{attempt}.json")))?;
            let id = experience.id().map_err(|e| e.to_string())?;
            held += 1;
            if members.contains_key(&id) {
                continue;
            }
            let model = Some(experience.provenance.solver.clone());
            corpus
                .insert(experience, notes)
                .map_err(|e| e.to_string())?;
            members.insert(
                id,
                Member {
                    source: source.to_owned(),
                    name: format!("{}#{attempt}", item.id),
                    task: item.id.clone(),
                    agent: crate::task_ops::ANSWER.into(),
                    model,
                    guided: false,
                },
            );
        }
    }
    Ok(held)
}

pub(super) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
    dataset: &Dataset,
) -> Result<(), RunError> {
    let fail = |why: String| stage_error(&stage.key, why);
    let state = tempfile::tempdir().map_err(io(staging.path()))?;
    let store = SourceStore::new(&Workspace::at(&StateRoot::new(state.path().to_path_buf())));
    let (view, objective) = view_of(dataset, &store).map_err(fail)?;
    let mut held = Vec::new();
    let mut sources = Vec::new();
    for source in &dataset.sources {
        sources.push(factory.product(source)?.1.digest());
        let at = factory.view(source);
        let trials = trials_of(&at)?;
        let tasks = factory
            .resolved
            .plan
            .stage(source)
            .and_then(|s| s.source("tasks"))
            .map(|pin| factory.view(pin));
        held.push(Source {
            name: source.clone(),
            view: at,
            trials,
            tasks,
        });
    }
    let mut raw: usize = held.iter().map(|s| s.trials.len()).sum();
    let mut trials = corpus_of(&held).map_err(fail)?;
    let mut members: BTreeMap<ExperienceId, Member> = trials
        .trials
        .iter()
        .map(|(id, (source, trial))| {
            let agent = format!(
                "{}{} {}",
                if trial.guided { "guided " } else { "" },
                trial.agent.name,
                trial.agent.version.as_deref().unwrap_or("?")
            );
            let member = Member {
                source: source.clone(),
                name: trial.slot.to_string(),
                task: trial.task.clone(),
                agent,
                model: trial.model.clone(),
                guided: trial.guided,
            };
            (id.clone(), member)
        })
        .collect();
    for source in &dataset.answers {
        sources.push(factory.product(source)?.1.digest());
        raw += add_answers(
            source,
            &factory.view(source),
            &mut trials.corpus,
            &mut members,
            &store,
        )
        .map_err(fail)?;
    }
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
    // A record is as long as its longest continuation under the shared prompt: that is the row the
    // trainer must hold (a pair is trained as two rows, a judged completion as one).
    if let (Some(limit), Some(trainer)) = (dataset.max_block, factory.services.trainer) {
        // One count for every row of every record: a request per record would take hours on a large rollout set.
        let mut lines = Vec::new();
        let mut rows_of_record = Vec::with_capacity(projection.records.len());
        for record in &projection.records {
            let rows = rows_of(record);
            rows_of_record.push(rows.len());
            for row in &rows {
                lines.push(
                    splinter_data::dataset_line(Objective::Sft, row)
                        .map_err(|e| fail(e.to_string()))?,
                );
            }
        }
        let lengths = trainer
            .measure(&dataset.target, &lines)
            .map_err(|e| fail(e.to_string()))?;
        let mut at = 0;
        let mut kept = Vec::with_capacity(projection.records.len());
        for (record, rows) in std::mem::take(&mut projection.records)
            .into_iter()
            .zip(rows_of_record)
        {
            let tokens = lengths[at..at + rows].iter().copied().max().unwrap_or(0);
            at += rows;
            if tokens <= limit {
                kept.push(record);
                continue;
            }
            let member = record
                .metadata
                .experiences
                .iter()
                .find_map(|id| members.get(id));
            excluded.push(Excluded {
                pass: member.map(|m| m.source.clone()).unwrap_or_default(),
                slot: member.map(|m| m.name.clone()).unwrap_or_default(),
                task: member.map(|m| m.task.clone()).unwrap_or_default(),
                reason: "longer_than_max_block".into(),
                detail: format!(
                    "{tokens} tokens of {} where max_block is {limit}",
                    dataset.target
                ),
            });
        }
        projection.records = kept;
    }
    if projection.records.is_empty() {
        return Err(fail(format!(
            "no record of {raw} attempts; the {} view left out {:?}",
            view.name(),
            projection.excluded
        )));
    }
    // Each record's attempts, in the order the record names their
    // experiences.
    let of = |record: &Record| -> Vec<&Member> {
        record
            .metadata
            .experiences
            .iter()
            .filter_map(|id| members.get(id))
            .collect()
    };
    let group = |record: &Record| {
        of(record)
            .first()
            .map(|member| group_of(factory, dataset, &member.task))
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
        for member in &made_of {
            sides
                .entry(group_of(factory, dataset, &member.task))
                .or_default()
                .insert(split);
            accepted.push(Accepted {
                pass: member.source.clone(),
                slot: member.name.clone(),
                task: member.task.clone(),
                agent: member.agent.clone(),
                model: member.model.clone(),
                actions: supervised(record),
                split: split.into(),
                guided: member.guided,
            });
        }
        let named: Vec<_> = made_of
            .iter()
            .map(|m| json!({ "pass": m.source, "slot": m.name, "task": m.task }))
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
            step: "attempts".into(),
            remaining: raw as u64,
        },
        WaterfallStep {
            step: "experiences".into(),
            remaining: members.len() as u64,
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
