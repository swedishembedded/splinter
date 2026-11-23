// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training datasets that grow with verified
// new solutions without repeating themselves, for its clients. If your team
// needs expertise in keeping an agent's training data diverse and its
// held-out tasks held out as the data grows, you can procure our services by
// sending an email to info@swedishembedded.com.

//! `dataset.extend`: new records admitted into an existing dataset only when
//! they are a new solution of their task.
//!
//! The stage takes a `base` dataset and a `candidates` dataset (new verified
//! trajectories projected as records, `dataset.project`) and makes a third:
//! every record of the base, byte for byte and in its order, then the
//! candidates that qualify. A candidate qualifies when
//!
//! - it was made under the base's system prompt and tool schema (a record
//!   rendered under another prompt trains the model on a context it is never
//!   served);
//! - the sequence of its tool calls (name and arguments, keys sorted and
//!   whitespace normalised, `ignore_call_arguments` left out) is not one the
//!   task already has, in the base or among the candidates admitted before
//!   it: a repeated solution adds weight, not information;
//! - its task has fewer than `per_task` records yet.
//!
//! Candidates are weighed in the order of their task and their content
//! digest, so the result does not depend on the order they were produced in.
//!
//! A task's side never moves. A task the base holds out stays held out: its
//! new records go to validation. A task the base trains on stays in training.
//! A task the base has never seen is held out as a whole or trained on as a
//! whole, by a hash of `holdout_seed` and the task, at `holdout_percent`. A
//! task on both sides is refused.
//!
//! The product is the dataset product of [`super::dataset`], its projection
//! `extended`, and `lineage.json`: the parents, the counts, and for every
//! candidate left out the reason.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::json;
use splinter_core::digest::Digest;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::{DatasetProduct, WaterfallStep, DATASET_SCHEMA};
use splinter_data::load::{read_records, ChatRecord, Record};
use splinter_store::factory::products::Staging;

use super::dataset::{dataset_of, file_ref, Accepted, Excluded, DATASET_FILE};
use super::{io, stage_error, write_json, write_lines, Factory, RunError};
use crate::ops::ExtendDatasetParams;
use crate::plan::PlannedStage;
use crate::recipe::ModelRef;

/// The projection an extended dataset records.
pub const PROJECTION: &str = "extended";
/// The admission an extended dataset records.
pub const ADMISSION: &str = "novel-solutions";
/// The lineage file of the product.
pub const LINEAGE_FILE: &str = "lineage.json";

/// One record line, read.
#[derive(Clone, Debug)]
struct Line {
    text: String,
    task: String,
    prompt: String,
    tools: String,
    calls: Vec<(String, String)>,
    supervised: usize,
}

/// Why a candidate was left out.
const DUPLICATE_SOLUTION: &str = "duplicate_solution";
const OVER_TASK_CAP: &str = "over_task_cap";
const PROMPT_MISMATCH: &str = "prompt_mismatch";
const TOOLS_MISMATCH: &str = "tools_mismatch";

/// A record line, read: its task, the context it was made under and the
/// calls it makes.
fn line_of(record: &ChatRecord, ignore: &[String]) -> Result<Line, String> {
    let supervised = record.supervised().len();
    if supervised == 0 {
        return Err("supervises no message (`train: true`)".into());
    }
    Ok(Line {
        text: record.line().to_owned(),
        task: record
            .task()
            .ok_or("names no task (`metadata.task`)")?
            .to_owned(),
        prompt: record
            .system_prompt()
            .ok_or("does not open with a system message")?
            .to_owned(),
        tools: record.tools_canonical()?,
        calls: record.calls(ignore)?,
        supervised,
    })
}

fn read_lines(path: &Path, ignore: &[String]) -> Result<Vec<Line>, String> {
    read_records::<ChatRecord>(path)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|(number, record)| {
            line_of(&record, ignore).map_err(|e| format!("{} line {number}: {e}", path.display()))
        })
        .collect()
}

/// The side a task new to the base goes to: held out as a whole at
/// `percent`, by a hash of the seed and the task.
fn held_out(seed: u64, task: &str, percent: u8) -> bool {
    let digest = Digest::of(format!("{seed}\0{task}").as_bytes());
    let draw = u64::from_str_radix(&digest.hex()[..16], 16).unwrap_or(0);
    draw % 100 < u64::from(percent)
}

/// The content digest candidates are ordered by.
fn content_digest(text: &str) -> String {
    Digest::of(text.as_bytes()).hex().to_owned()
}

/// One candidate's fate.
#[derive(Debug, PartialEq, Eq)]
pub enum Fate {
    /// Added to the dataset: to validation when `validation`, else to train.
    Admitted {
        /// Whether the record's task is held out.
        validation: bool,
    },
    /// Left out, for this reason.
    Left(&'static str),
}

/// What the base holds, and what is admitted on top of it.
pub struct Admission {
    prompt: String,
    tools: String,
    seen: BTreeMap<String, BTreeSet<Vec<(String, String)>>>,
    count: BTreeMap<String, usize>,
    held: BTreeSet<String>,
    per_task: usize,
    seed: u64,
    percent: u8,
}

impl Admission {
    fn new(
        base_train: &[Line],
        base_validation: &[Line],
        params: &ExtendDatasetParams,
    ) -> Option<Self> {
        let first = base_train.first().or(base_validation.first())?;
        let mut admission = Self {
            prompt: first.prompt.clone(),
            tools: first.tools.clone(),
            seen: BTreeMap::new(),
            count: BTreeMap::new(),
            held: base_validation.iter().map(|l| l.task.clone()).collect(),
            per_task: params.per_task,
            seed: params.holdout_seed,
            percent: params.holdout_percent,
        };
        for line in base_train.iter().chain(base_validation) {
            admission
                .seen
                .entry(line.task.clone())
                .or_default()
                .insert(line.calls.clone());
            *admission.count.entry(line.task.clone()).or_default() += 1;
        }
        Some(admission)
    }

    /// Weighs one candidate, remembering it when it is admitted.
    fn weigh(&mut self, line: &Line) -> Fate {
        if line.prompt != self.prompt {
            return Fate::Left(PROMPT_MISMATCH);
        }
        if line.tools != self.tools {
            return Fate::Left(TOOLS_MISMATCH);
        }
        if self
            .seen
            .get(&line.task)
            .is_some_and(|calls| calls.contains(&line.calls))
        {
            return Fate::Left(DUPLICATE_SOLUTION);
        }
        if self.count.get(&line.task).copied().unwrap_or(0) >= self.per_task {
            return Fate::Left(OVER_TASK_CAP);
        }
        let validation = if self.count.contains_key(&line.task) {
            self.held.contains(&line.task)
        } else {
            held_out(self.seed, &line.task, self.percent)
        };
        if validation {
            self.held.insert(line.task.clone());
        }
        self.seen
            .entry(line.task.clone())
            .or_default()
            .insert(line.calls.clone());
        *self.count.entry(line.task.clone()).or_default() += 1;
        Fate::Admitted { validation }
    }
}

/// The candidates in the order they are weighed in: by task, then content.
fn weighing_order(lines: &mut [Line]) {
    lines.sort_by_cached_key(|l| (l.task.clone(), content_digest(&l.text)));
}

struct Resolved {
    params: ExtendDatasetParams,
    target: ModelRef,
    base: String,
    candidates: String,
}

fn resolved(factory: &Factory<'_>, stage: &PlannedStage) -> Result<Resolved, RunError> {
    let params: ExtendDatasetParams = stage.params().map_err(|e| stage_error(&stage.key, e))?;
    let target = factory
        .resolved
        .model(&params.target)
        .cloned()
        .ok_or_else(|| {
            stage_error(
                &stage.key,
                format!("model role {} is not bound", params.target),
            )
        })?;
    let source = |port: &str| {
        stage
            .source(port)
            .map(str::to_owned)
            .ok_or_else(|| stage_error(&stage.key, format!("extends from no `{port}` dataset")))
    };
    Ok(Resolved {
        params,
        target,
        base: source("base")?,
        candidates: source("candidates")?,
    })
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let extend = resolved(factory, stage)?;
    let inputs = vec![
        (
            extend.base.clone(),
            factory.product(&extend.base)?.1.digest(),
        ),
        (
            extend.candidates.clone(),
            factory.product(&extend.candidates)?.1.digest(),
        ),
    ];
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &stage.key,
        json!({
            "target": extend.target, "per_task": extend.params.per_task,
            "ignore_call_arguments": extend.params.ignore_call_arguments,
            "holdout_percent": extend.params.holdout_percent,
            "holdout_seed": extend.params.holdout_seed,
        }),
        inputs,
        "",
        None,
    ))
}

fn lines_of(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    key: &str,
    ignore: &[String],
) -> Result<(DatasetProduct, Vec<Line>, Vec<Line>), RunError> {
    let view = factory.view(key);
    let product = dataset_of(&view)?;
    let read = |file: &splinter_core::factory::execution::FileRef| {
        read_lines(&view.join(&file.path), ignore).map_err(|e| stage_error(&stage.key, e))
    };
    let train = read(&product.train)?;
    let validation = read(&product.validation)?;
    Ok((product, train, validation))
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
) -> Result<(), RunError> {
    let extend = resolved(factory, stage)?;
    if extend.params.per_task == 0 || extend.params.holdout_percent > 100 {
        return Err(stage_error(
            &stage.key,
            "per_task is at least 1 and holdout_percent at most 100",
        ));
    }
    let ignore = &extend.params.ignore_call_arguments;
    let (base, base_train, base_validation) = lines_of(factory, stage, &extend.base, ignore)?;
    let (candidates, cand_train, cand_validation) =
        lines_of(factory, stage, &extend.candidates, ignore)?;
    if base.thinking != candidates.thinking {
        return Err(stage_error(
            &stage.key,
            "the base and the candidates were made for different reasoning modes",
        ));
    }
    let shared: Vec<&String> = {
        let train: BTreeSet<&String> = base_train.iter().map(|l| &l.task).collect();
        let held: BTreeSet<&String> = base_validation.iter().map(|l| &l.task).collect();
        train.intersection(&held).copied().collect()
    };
    if !shared.is_empty() {
        return Err(stage_error(
            &stage.key,
            format!("the base has tasks on both sides: {shared:?}"),
        ));
    }
    let mut admission = Admission::new(&base_train, &base_validation, &extend.params)
        .ok_or_else(|| stage_error(&stage.key, "the base holds no record"))?;
    let mut pool: Vec<Line> = cand_train.into_iter().chain(cand_validation).collect();
    let supplied = pool.len();
    weighing_order(&mut pool);
    let (mut added_train, mut added_validation): (Vec<Line>, Vec<Line>) = (Vec::new(), Vec::new());
    let mut excluded: Vec<Excluded> = Vec::new();
    for line in pool {
        match admission.weigh(&line) {
            Fate::Admitted { validation: true } => added_validation.push(line),
            Fate::Admitted { validation: false } => added_train.push(line),
            Fate::Left(reason) => excluded.push(Excluded {
                pass: extend.candidates.clone(),
                slot: content_digest(&line.text)[..16].to_owned(),
                task: line.task.clone(),
                reason: reason.into(),
                detail: format!("{} tool calls", line.calls.len()),
            }),
        }
    }
    // A task never moves between sides, whatever the order records arrived in.
    let side = |lines: &[Line]| {
        lines
            .iter()
            .map(|l| l.task.clone())
            .collect::<BTreeSet<_>>()
    };
    let train_tasks: BTreeSet<String> = side(&base_train)
        .union(&side(&added_train))
        .cloned()
        .collect();
    let held_tasks: BTreeSet<String> = side(&base_validation)
        .union(&side(&added_validation))
        .cloned()
        .collect();
    if let Some(task) = train_tasks.intersection(&held_tasks).next() {
        return Err(stage_error(
            &stage.key,
            format!("task {task} would be on both sides"),
        ));
    }
    let splits: [(&str, &str, Vec<&Line>, usize); 2] = [
        (
            "train",
            "train.jsonl",
            base_train.iter().chain(&added_train).collect(),
            base_train.len(),
        ),
        (
            "validation",
            "validation.jsonl",
            base_validation.iter().chain(&added_validation).collect(),
            base_validation.len(),
        ),
    ];
    let (mut accepted, mut supervision) = (Vec::new(), Vec::new());
    for (split, name, lines, from_base) in &splits {
        let mut text = lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        let path = staging.path().join(name);
        std::fs::write(&path, text).map_err(io(&path))?;
        for (index, line) in lines.iter().enumerate() {
            accepted.push(Accepted {
                pass: if index < *from_base {
                    extend.base.clone()
                } else {
                    extend.candidates.clone()
                },
                slot: format!("{split}/{index}"),
                task: line.task.clone(),
                agent: "extended".into(),
                model: None,
                actions: line.supervised,
                split: (*split).to_owned(),
                guided: false,
            });
            supervision.push(json!({
                "split": split, "line": index, "task": line.task, "supervised_messages": line.supervised,
            }));
        }
    }
    write_lines(&staging.path().join("accepted.jsonl"), &accepted)?;
    write_lines(&staging.path().join("excluded.jsonl"), &excluded)?;
    write_lines(&staging.path().join("supervision.jsonl"), &supervision)?;
    let mut exclusions = BTreeMap::new();
    for e in &excluded {
        *exclusions.entry(e.reason.clone()).or_insert(0u64) += 1;
    }
    let parents = vec![
        factory.product(&extend.base)?.1.digest(),
        factory.product(&extend.candidates)?.1.digest(),
    ];
    let base_total = (base_train.len() + base_validation.len()) as u64;
    let waterfall = vec![
        WaterfallStep {
            step: "base_records".into(),
            remaining: base_total,
        },
        WaterfallStep {
            step: "candidate_records".into(),
            remaining: base_total + supplied as u64,
        },
        WaterfallStep {
            step: "admitted".into(),
            remaining: accepted.len() as u64,
        },
    ];
    let product = DatasetProduct {
        schema: DATASET_SCHEMA.into(),
        name: stage.key.clone(),
        objective: base.objective.clone(),
        projection: PROJECTION.into(),
        admission: ADMISSION.into(),
        target: extend.target.to_string(),
        sources: parents.clone(),
        train: file_ref(staging, "train.jsonl")?,
        validation: file_ref(staging, "validation.jsonl")?,
        accepted: file_ref(staging, "accepted.jsonl")?,
        excluded: file_ref(staging, "excluded.jsonl")?,
        supervision: file_ref(staging, "supervision.jsonl")?,
        waterfall,
        exclusions: exclusions.clone(),
        thinking: base.thinking,
    };
    write_json(&staging.path().join(DATASET_FILE), &product)?;
    let new_tasks: Vec<&String> = train_tasks
        .iter()
        .chain(&held_tasks)
        .filter(|t| !side(&base_train).contains(*t) && !side(&base_validation).contains(*t))
        .collect();
    write_json(
        &staging.path().join(LINEAGE_FILE),
        &json!({
            "parents": { "base": parents[0], "candidates": parents[1] },
            "per_task": extend.params.per_task,
            "ignore_call_arguments": extend.params.ignore_call_arguments,
            "holdout": { "seed": extend.params.holdout_seed, "percent": extend.params.holdout_percent },
            "base_records": base_total, "candidate_records": supplied,
            "admitted": { "train": added_train.len(), "validation": added_validation.len() },
            "left_out": exclusions, "new_tasks": new_tasks,
            "held_out_tasks": held_tasks,
        }),
    )?;
    let report = format!(
        "# Dataset {}\n\nExtended {} with {}: {} candidate records, {} admitted ({} train, {} \
         validation), {} left out {:?}. {} train and {} validation records in all.\n",
        stage.key,
        extend.base,
        extend.candidates,
        supplied,
        added_train.len() + added_validation.len(),
        added_train.len(),
        added_validation.len(),
        excluded.len(),
        exclusions,
        splits[0].2.len(),
        splits[1].2.len(),
    );
    std::fs::write(staging.path().join("report.md"), report).map_err(io(staging.path()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(task: &str, prompt: &str, calls: &[&str]) -> String {
        let mut messages = vec![
            json!({"role": "system", "content": prompt}),
            json!({"role": "user", "content": "do it"}),
        ];
        for (i, command) in calls.iter().enumerate() {
            messages.push(json!({
                "role": "assistant", "content": "ok", "train": true,
                "tool_calls": [{
                    "id": format!("c{i}"), "type": "function",
                    "function": {"name": "vim", "arguments": json!({"commands": [command], "intent": format!("why {i}")}).to_string()}
                }]
            }));
            messages
                .push(json!({"role": "tool", "tool_call_id": format!("c{i}"), "content": "done"}));
        }
        json!({"messages": messages, "tools": [{"name": "vim"}], "metadata": {"task": task}})
            .to_string()
    }

    fn lines(records: &[String], ignore: &[String]) -> Vec<Line> {
        records
            .iter()
            .map(|r| line_of(&ChatRecord::parse(r).unwrap(), ignore).unwrap())
            .collect()
    }

    fn params(per_task: usize) -> ExtendDatasetParams {
        ExtendDatasetParams {
            target: "student".into(),
            per_task,
            ignore_call_arguments: vec!["intent".into()],
            holdout_percent: 10,
            // seed: the held-out draw of a new task is a function of this seed and the task.
            holdout_seed: 7,
        }
    }

    fn base() -> Admission {
        let ignore = vec!["intent".to_string()];
        let train = lines(
            &[record("a", "P", &["x"]), record("b", "P", &["y"])],
            &ignore,
        );
        let validation = lines(&[record("h", "P", &["z"])], &ignore);
        Admission::new(&train, &validation, &params(3)).unwrap()
    }

    fn weigh(admission: &mut Admission, record: &str) -> Fate {
        admission
            .weigh(&line_of(&ChatRecord::parse(record).unwrap(), &["intent".to_string()]).unwrap())
    }

    #[test]
    fn a_solution_the_task_already_has_is_a_duplicate_whatever_its_intent_says() {
        let mut admission = base();
        assert_eq!(
            weigh(&mut admission, &record("a", "P", &["x"])),
            Fate::Left(DUPLICATE_SOLUTION)
        );
        assert_eq!(
            weigh(&mut admission, &record("a", "P", &["w"])),
            Fate::Admitted { validation: false }
        );
        assert_eq!(
            weigh(&mut admission, &record("a", "P", &["w"])),
            Fate::Left(DUPLICATE_SOLUTION),
            "an admitted candidate counts as seen"
        );
    }

    #[test]
    fn a_task_has_at_most_per_task_records() {
        let mut admission = base();
        assert_eq!(
            weigh(&mut admission, &record("a", "P", &["1"])),
            Fate::Admitted { validation: false }
        );
        assert_eq!(
            weigh(&mut admission, &record("a", "P", &["2"])),
            Fate::Admitted { validation: false }
        );
        assert_eq!(
            weigh(&mut admission, &record("a", "P", &["3"])),
            Fate::Left(OVER_TASK_CAP)
        );
    }

    #[test]
    fn a_held_out_task_stays_held_out_and_a_trained_task_stays_trained() {
        let mut admission = base();
        assert_eq!(
            weigh(&mut admission, &record("h", "P", &["q"])),
            Fate::Admitted { validation: true }
        );
        assert_eq!(
            weigh(&mut admission, &record("b", "P", &["q"])),
            Fate::Admitted { validation: false }
        );
    }

    #[test]
    fn a_new_task_is_held_out_as_a_whole_by_its_hash() {
        let mut admission = base();
        let first = weigh(&mut admission, &record("fresh", "P", &["1"]));
        let second = weigh(&mut admission, &record("fresh", "P", &["2"]));
        assert_eq!(first, second, "both records of a task go to one side");
        let again = held_out(7, "fresh", 10);
        assert_eq!(first, Fate::Admitted { validation: again });
    }

    #[test]
    fn a_record_under_another_prompt_or_tool_schema_is_refused() {
        let mut admission = base();
        assert_eq!(
            weigh(&mut admission, &record("a", "OTHER", &["n"])),
            Fate::Left(PROMPT_MISMATCH)
        );
        let other_tools = record("a", "P", &["n"]).replace(r#""name":"vim""#, r#""name":"vi""#);
        assert_eq!(
            weigh(&mut admission, &other_tools),
            Fate::Left(TOOLS_MISMATCH)
        );
    }

    #[test]
    fn the_order_candidates_arrive_in_does_not_change_what_is_admitted() {
        let ignore = vec!["intent".to_string()];
        let make = |order: &[usize]| {
            let pool = [
                record("a", "P", &["1"]),
                record("a", "P", &["2"]),
                record("a", "P", &["3"]),
                record("a", "P", &["4"]),
            ];
            let mut chosen: Vec<Line> = order
                .iter()
                .map(|&i| line_of(&ChatRecord::parse(&pool[i]).unwrap(), &ignore).unwrap())
                .collect();
            weighing_order(&mut chosen);
            let mut admission = base();
            chosen
                .iter()
                .filter(|l| matches!(admission.weigh(l), Fate::Admitted { .. }))
                .map(|l| l.text.clone())
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(make(&[0, 1, 2, 3]), make(&[3, 2, 1, 0]));
        assert_eq!(
            make(&[0, 1, 2, 3]).len(),
            2,
            "per_task of 3 with one base record"
        );
    }
}
