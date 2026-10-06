// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `exam-set create` and the powered `exam`, as recorded runs.

use serde_json::json;
use splinter_sdk::checkpoints::{select as select_checkpoint, SelectRequest, Selected};
use splinter_sdk::exam_set::{build, ExamBuilt, ExamSet, NewExam};
use splinter_sdk::powered::plan::{simulate, Plan, Planned};
use splinter_sdk::powered::{audit, memorisation, reports};
use splinter_sdk::powered::{run, ArmChoice, PoweredExam, PoweredRequest, DEFAULT_RESAMPLES};
use splinter_sdk::runs::{record, Recorded};
use splinter_sdk::sources::SourceTarget;
use splinter_sdk::train::{adopt_checkpoint, load_candidate};
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::{Context, Error};

use crate::cli::{ExamArgs, ExamReportCommand, ExamSetArgs, ExamSetCommand, PowerArgs, SelectArgs};
use crate::output::emit;

/// Captures the sources, reserves the exam's families and freezes the exam.
pub fn create(ctx: &Context, args: &ExamSetArgs) -> Result<Recorded<ExamBuilt>, Error> {
    let targets = args
        .sources
        .iter()
        .map(|s| SourceTarget::from_learn_arg(s))
        .collect::<Result<Vec<_>, _>>()?;
    let generator = args
        .generator
        .clone()
        .unwrap_or_else(ModelRef::policy_default);
    let arguments = json!({
        "sources": args.sources,
        "families": args.families,
        "tasks_per_family": args.tasks_per_family,
        "dev_families": args.dev_families,
        "dev_tasks_per_family": args.dev_tasks_per_family,
        "kinds": args.kinds,
        "seed": args.seed,
        "not_trained_by": args.not_trained_by,
    });
    record(ctx, "exam-set", &arguments, |run| {
        build(
            ctx,
            &NewExam {
                sources: &targets,
                families: args.families,
                seed: args.seed,
                tasks_per_family: args.tasks_per_family,
                dev_families: args.dev_families,
                dev_tasks_per_family: args.dev_tasks_per_family,
                kinds: &args.kinds,
                generator: &generator,
                goal: args.goal.as_deref(),
                author: args.persona.as_deref(),
                not_trained_by: &args.not_trained_by,
                cancel: run.cancel_token(),
            },
        )
    })
}

/// Puts the candidate to the frozen exam `args` names.
pub fn powered(ctx: &Context, args: &ExamArgs) -> Result<Recorded<PoweredExam>, Error> {
    let named = args.exam_set.clone().unwrap_or_default();
    let exam = ExamSet::load(ctx, &named)?;
    let arguments = json!({
        "candidate": args.candidate,
        "exam_set": exam.id,
        "judge": args.judge,
        "prompt": args.prompt,
        "resamples": args.resamples,
        "pilot_families": args.pilot_families,
        "voice": !args.no_voice,
    });
    record(ctx, "exam", &arguments, |cancelled| {
        let cancel = cancelled.cancel_token();
        match run(
            ctx,
            &PoweredRequest {
                exam: &exam,
                candidate: &args.candidate,
                base: None,
                judge: args.judge.as_ref(),
                goal: args.prompt.as_deref(),
                resamples: args.resamples.map_or(DEFAULT_RESAMPLES, |n| n as usize),
                pilot_families: args.pilot_families,
                voice: !args.no_voice,
                arms: ArmChoice::All,
                adapter: None,
                cancel: &cancel,
            },
        ) {
            Ok(powered) => Ok(PoweredExam::Ran(Box::new(powered))),
            Err(e) if e.is_refusal() => Ok(PoweredExam::NotRun(e.to_string())),
            Err(e) => Err(e),
        }
    })
}

/// What `select` found, and the candidate made of the evaluation chosen when
/// that was asked.
#[derive(serde::Serialize)]
pub struct Picked {
    /// Every kept evaluation as the dev suite scored it, and the choice.
    #[serde(flatten)]
    pub selected: Selected,
    /// The candidate that carries the chosen evaluation's adapter, with
    /// `--adopt`.
    pub adopted: Option<String>,
}

/// Puts each evaluation the candidate kept to the dev suite and chooses one.
pub fn select(ctx: &Context, args: &SelectArgs) -> Result<Recorded<Picked>, Error> {
    let exam = ExamSet::load(ctx, &args.exam_set)?;
    let arguments = json!({
        "candidate": args.candidate,
        "exam_set": exam.id,
        "judge": args.judge,
        "adopt": args.adopt,
    });
    record(ctx, "select", &arguments, |running| {
        let selected = select_checkpoint(
            ctx,
            &SelectRequest {
                candidate: &args.candidate,
                exam: &exam,
                judge: args.judge.as_ref(),
                cancel: &running.cancel_token(),
            },
        )?;
        let adopted = if args.adopt {
            let trained = load_candidate(ctx, &selected.candidate)?;
            Some(adopt_checkpoint(ctx, &trained, selected.chosen.step)?.candidate)
        } else {
            None
        };
        Ok(Picked { selected, adopted })
    })
}

/// What `exam-set power` found: the plan and what it came to.
#[derive(serde::Serialize)]
pub struct PlannedExam {
    /// The assumptions, the discordance and clustering taken from a pilot's
    /// report when one was named.
    pub plan: Plan,
    /// What simulating the planned test gave.
    pub planned: Planned,
}

/// Simulates the planned paired test.
pub fn power(args: &PowerArgs) -> Result<PlannedExam, Error> {
    let (discordance, icc) = match &args.from_report {
        None => (args.discordance, args.icc),
        Some(file) => {
            let text = std::fs::read_to_string(file)
                .map_err(|e| Error::Refused(format!("{}: {e}", file.display())))?;
            let report: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| Error::Refused(format!("{}: {e}", file.display())))?;
            // The report is the run's record or the bare report.
            let powered = report["outputs"]["ran"]
                .as_object()
                .map(|_| &report["outputs"]["ran"])
                .or_else(|| report["ran"].as_object().map(|_| &report["ran"]))
                .unwrap_or(&report);
            let power = &powered["comparisons"][0]["power"];
            let read = |name: &str| {
                power[name].as_f64().ok_or_else(|| {
                    Error::Refused(format!(
                        "{} holds no primary comparison's {name}",
                        file.display()
                    ))
                })
            };
            (read("discordance")?, read("icc")?)
        }
    };
    let plan = Plan {
        families: args.families,
        tasks_per_family: args.tasks_per_family,
        effect: args.effect,
        discordance,
        icc,
        replicates: args.replicates,
        resamples: 500,
        seed: 0,
    };
    Ok(PlannedExam {
        plan,
        planned: simulate(&plan),
    })
}

/// What `exam-report labels-export` wrote.
#[derive(serde::Serialize)]
pub struct LabelsExported {
    /// Items to label.
    pub items: usize,
    /// The file of items.
    pub out: std::path::PathBuf,
    /// The key.
    pub key: std::path::PathBuf,
}

/// Writes the sample to label and its key.
pub fn labels_export(
    ctx: &Context,
    report: &std::path::Path,
    out: &std::path::Path,
    key: &std::path::Path,
    n: usize,
    seed: u64,
) -> Result<LabelsExported, Error> {
    let records = reports::read_records(report)?;
    let (items, keyed) = audit::export(&records, &reports::StoredWording(ctx), n, seed);
    let lines: Vec<String> = items
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<_, _>>()
        .map_err(|e| Error::Refused(e.to_string()))?;
    std::fs::write(out, lines.join("\n") + "\n")
        .map_err(|e| Error::Refused(format!("{}: {e}", out.display())))?;
    let key_text =
        serde_json::to_string_pretty(&keyed).map_err(|e| Error::Refused(e.to_string()))?;
    std::fs::write(key, key_text).map_err(|e| Error::Refused(format!("{}: {e}", key.display())))?;
    Ok(LabelsExported {
        items: items.len(),
        out: out.to_path_buf(),
        key: key.to_path_buf(),
    })
}

/// What the labels in `labels` say of the judge, by the key in `key`.
pub fn labels_import(
    labels: &std::path::Path,
    key: &std::path::Path,
) -> Result<audit::Agreement, Error> {
    let read = |path: &std::path::Path| {
        std::fs::read_to_string(path)
            .map_err(|e| Error::Refused(format!("{}: {e}", path.display())))
    };
    let keyed: std::collections::BTreeMap<String, audit::Keyed> = serde_json::from_str(&read(key)?)
        .map_err(|e| Error::Refused(format!("{}: {e}", key.display())))?;
    let mut given = std::collections::BTreeMap::new();
    for line in read(labels)?.lines().filter(|l| !l.trim().is_empty()) {
        let item: audit::Item = serde_json::from_str(line)
            .map_err(|e| Error::Refused(format!("{}: {e}", labels.display())))?;
        if let Some(label) = item.label {
            given.insert(item.id, label);
        }
    }
    Ok(audit::agreement(&given, &keyed))
}

/// Each arm's memorisation against what the candidate was trained to produce.
#[derive(serde::Serialize)]
pub struct Memorisation {
    /// The arms.
    pub arms: Vec<memorisation::ArmMemorisation>,
}

/// Measures the answers of the report against the candidate's training text.
pub fn memorise(
    ctx: &Context,
    report: &std::path::Path,
    candidate: &str,
) -> Result<Memorisation, Error> {
    let records = reports::read_records(report)?;
    let corpus = reports::training_corpus(ctx, candidate)?;
    let arms: std::collections::BTreeSet<String> = records
        .iter()
        .flat_map(|r| r.arms.keys().cloned())
        .collect();
    let arms: Vec<String> = arms.into_iter().collect();
    Ok(Memorisation {
        arms: memorisation::summarise(&corpus, &records, &arms),
    })
}

/// Runs an `exam-report` command.
pub fn report(ctx: &Context, json: bool, command: ExamReportCommand) -> Result<(), Error> {
    match command {
        ExamReportCommand::LabelsExport {
            report,
            out,
            key,
            n,
            seed,
        } => emit(json, &labels_export(ctx, &report, &out, &key, n, seed)?),
        ExamReportCommand::LabelsImport { labels, key } => {
            emit(json, &labels_import(&labels, &key)?);
        }
        ExamReportCommand::Memorisation { report, candidate } => {
            emit(json, &memorise(ctx, &report, &candidate)?);
        }
    }
    Ok(())
}

/// Runs an `exam-set` command.
pub fn set(ctx: &Context, json: bool, command: ExamSetCommand) -> Result<(), Error> {
    match command {
        ExamSetCommand::Create(args) => emit(json, &create(ctx, &args)?),
        ExamSetCommand::Power(args) => emit(json, &power(&args)?),
        ExamSetCommand::Show { id } => emit(json, &ExamSet::load(ctx, &id)?),
    }
    Ok(())
}
