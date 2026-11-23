// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements honest reporting of what a training run
// taught a model, separating program success from learning claims, for its
// clients. If your team needs expertise in evaluating small-model learning,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The report: what each stage produced and what each model can do, read
//! from the ledger alone.
//!
//! Two kinds of evidence are kept apart. Learning from scratch is what the
//! probes show across random init, pretraining and fine-tuning. Learning
//! from experience is what training on the model's own verified episodes
//! changed. A finished program is not a learning claim, and a measurement
//! that was not made is said to be absent.

use std::fmt::Write as _;

use serde_json::Value;
use splinter_sdk::store::stage_ledger::Commit;

fn find<'a>(commits: &'a [Commit], unit: &str) -> Option<&'a Commit> {
    commits.iter().rev().find(|c| c.unit == unit)
}

fn probe_line(commits: &[Commit], label: &str, stage: &str) -> String {
    match find(commits, &format!("probe/{label}/{stage}")) {
        Some(c) => format!(
            "| {stage} | {}/{} | {} |",
            c.summary["passed"],
            c.summary["total"],
            c.summary["text"]["continuation"].as_str().map_or_else(
                || format!(
                    "not measured: {}",
                    c.summary["text"]["unavailable"].as_str().unwrap_or("")
                ),
                |t| format!("{t:?}")
            )
        ),
        None => format!("| {stage} | not run | |"),
    }
}

fn losses(commits: &[Commit], unit: &str) -> String {
    find(commits, unit).map_or("not run".into(), |c| {
        format!(
            "{} steps, loss {} -> {}",
            c.summary["steps"], c.summary["first_loss"], c.summary["last_loss"]
        )
    })
}

/// The markdown report of every model labelled in `labels`.
#[must_use]
pub fn markdown(commits: &[Commit], labels: &[String]) -> String {
    let mut s = String::from("# minimind run report\n\n");
    if let Some(init) = find(commits, "init") {
        let _ = writeln!(
            s,
            "Profile {}, seed {}.\n",
            init.summary["profile"], init.summary["seed"]
        );
    }
    let count = |unit: &str, field: &str| {
        find(commits, unit).map_or(Value::Null, |c| c.summary[field].clone())
    };
    let _ = writeln!(
        s,
        "Data: {} world documents; demonstrations {} accepted of {} expert episodes.\n",
        count("world", "documents"),
        count("demonstrate", "accepted"),
        count("demonstrate", "episodes"),
    );
    for label in labels {
        let _ = writeln!(s, "## {label}\n");
        let _ = writeln!(
            s,
            "Training: pretrain {}; sft {}; improve {}.\n",
            losses(commits, &format!("{label}/pretrain")),
            losses(commits, &format!("{label}/sft")),
            losses(commits, &format!("{label}/improve"))
        );
        let _ = writeln!(s, "### Learning from scratch\n\n| checkpoint | development tasks | text continuation |\n|---|---|---|");
        for stage in ["init", "pretrained", "sft"] {
            let _ = writeln!(s, "{}", probe_line(commits, label, stage));
        }
        let _ = writeln!(s, "### Learning from experience\n");
        let _ = writeln!(
            s,
            "Own episodes: {} verified of {}.\n",
            count(&format!("{label}/explore"), "accepted"),
            count(&format!("{label}/explore"), "episodes")
        );
        let _ = writeln!(
            s,
            "| checkpoint | development tasks | text continuation |\n|---|---|---|"
        );
        for stage in ["sft", "improved"] {
            let _ = writeln!(s, "{}", probe_line(commits, label, stage));
        }
    }
    let _ = writeln!(
        s,
        "## Across stages\n\n```text{}```\n",
        crate::panel::summary(commits, labels)
    );
    s.push_str(
        "A probe is a handful of development tasks of the training families: it shows what a checkpoint does, \
         it is not evidence of learning.\n",
    );
    s
}

/// True for an episode's unit: its last segment is a task id.
fn is_episode(unit: &str) -> bool {
    unit.rsplit('/').next().is_some_and(|last| {
        ["train-", "explore-", "dev-", "heldout-", "midtrain-"]
            .iter()
            .any(|p| last.starts_with(p))
    })
}

/// Every committed unit on one line each, with the probes shown in full.
#[must_use]
pub fn status(commits: &[Commit]) -> String {
    let mut s = String::new();
    for c in commits {
        if let Some(name) = c
            .unit
            .strip_prefix("probe/")
            .filter(|n| n.matches('/').count() == 1)
        {
            s.push_str(&crate::panel::probe(name, &c.summary));
        } else if !is_episode(&c.unit) {
            let _ = writeln!(s, "[{:>6}] {} (attempt {})", "done", c.unit, c.attempt);
        }
    }
    s
}

/// Writes `report.md` into the run directory from the ledger, and says where.
///
/// # Errors
/// The ledger cannot be read or the report written.
pub fn write(run: &mut crate::stages::Run<'_>, labels: &[String]) -> anyhow::Result<()> {
    let commits = run.this_run.clone();
    let text = markdown(&commits, labels);
    let path = run.path("report.md");
    crate::stages::write(&path, &text)?;
    let table = crate::panel::summary(&commits, labels);
    std::io::Write::write_fmt(
        &mut run.out,
        format_args!("{table}\nreport: {}\n", path.display()),
    )?;
    Ok(())
}
