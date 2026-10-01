// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the commands print: every report as text for a person, or - with
//! `--json` - as the report itself, one JSON document on stdout whose field
//! names are the report's documented fields.

use std::fmt::Write as _;

use serde::Serialize;
use splinter_campaign::ask::Answer;
use splinter_campaign::critique::Critiqued;
use splinter_campaign::datasets::{Built, Exported};
use splinter_campaign::experiences::{
    CallReplay, ExperienceLine, ExperienceShow, Replayed, SetList,
};
use splinter_campaign::front_door::Routed;
use splinter_campaign::judge::Calibrated;
use splinter_campaign::runs::{CancelRequested, Recorded, RunList};
use splinter_campaign::solving::Solved;
use splinter_campaign::sources::{SourceAdded, SourceList, SourceSummary};
use splinter_campaign::status::Status;
use splinter_campaign::tasks::{TaskSetList, TaskShow, TasksGenerated};
use splinter_campaign::train::{Candidate, Regime};
use splinter_campaign::verify::Verified;
use splinter_campaign::CampaignError;
use splinter_store::runs::Run;
use splinter_store::source::{Origin, Source};

/// A report a command prints.
pub trait Report: Serialize {
    /// The report as text for a person.
    fn human(&self) -> String;
}

/// Prints `report` as JSON or as text.
pub fn emit(json: bool, report: &impl Report) {
    if json {
        match serde_json::to_string_pretty(report) {
            Ok(text) => println!("{text}"),
            // Every report is plain data; one that cannot be serialized is
            // a bug worth seeing rather than hiding.
            Err(e) => eprintln!("splinter: the report could not be written as JSON: {e}"),
        }
    } else {
        print!("{}", report.human());
    }
}

/// Prints why a command failed: on stderr always, and as
/// `{"error": ..., "refused": ...}` on stdout with `--json`.
pub fn error(json: bool, error: &CampaignError) {
    eprintln!("splinter: {error}");
    if json {
        let value =
            serde_json::json!({ "error": error.to_string(), "refused": error.is_refusal() });
        println!("{value:#}");
    }
}

/// Prints the question the front door asks back, or its refusal: as text,
/// or with `--json` as the routing itself (`{"routed": "clarify", "reason",
/// "candidates": [{"intent", "confidence", "command"}]}` or `{"routed":
/// "refuse", ...}`).
pub fn routed(json: bool, routed: &Routed) {
    if json {
        match serde_json::to_string_pretty(routed) {
            Ok(text) => println!("{text}"),
            Err(e) => eprintln!("splinter: the reply could not be written as JSON: {e}"),
        }
    }
    match routed {
        Routed::Refuse(why) => eprintln!("splinter: {why}"),
        Routed::Clarify(question) if !json => {
            println!("{}", question.reason);
            for (n, candidate) in question.candidates.iter().enumerate() {
                println!(
                    "  {}) splinter {}  ({:.2})",
                    n + 1,
                    shell_words(&candidate.command),
                    candidate.confidence
                );
            }
        }
        _ => {}
    }
}

/// Arguments as a person would type them, each quoted when it has to be.
pub fn shell_words(words: &[String]) -> String {
    words
        .iter()
        .map(|w| {
            if !w.is_empty()
                && w.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_./:,+=@".contains(c))
            {
                w.clone()
            } else {
                format!("'{}'", w.replace('\'', r"'\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A serializable value as a short word: a string as itself, anything else
/// as compact JSON.
pub(crate) fn word(value: &impl Serialize) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(s)) => s,
        Ok(other) => other.to_string(),
        Err(_) => "?".into(),
    }
}

/// Counts by name, as `name n, name n`.
pub(crate) fn tally<K: Serialize, V: std::fmt::Display>(
    counts: impl IntoIterator<Item = (K, V)>,
) -> String {
    let parts: Vec<String> = counts
        .into_iter()
        .map(|(k, v)| format!("{} {v}", word(&k)))
        .collect();
    if parts.is_empty() {
        "none".into()
    } else {
        parts.join(", ")
    }
}

impl<R: Report> Report for Recorded<R> {
    fn human(&self) -> String {
        format!("{}run: {}\n", self.report.human(), self.run)
    }
}

fn origin(origin: &Origin) -> String {
    match origin {
        Origin::Document { path } | Origin::Repository { path, .. } => path.clone(),
        Origin::Command {
            argv,
            cwd,
            exit_code,
            ..
        } => format!(
            "{} (in {cwd}, exit {})",
            shell_words(argv),
            exit_code.map_or("by signal".into(), |c| c.to_string())
        ),
    }
}

pub(crate) fn source_line(source: &SourceSummary) -> String {
    format!(
        "{}  {:<10}  {} part(s), {} bytes  {}",
        source.id,
        source.kind,
        source.parts,
        source.bytes,
        origin(&source.origin)
    )
}

impl Report for SourceAdded {
    fn human(&self) -> String {
        let note = if self.new { "" } else { " (already stored)" };
        format!("source {}{note}\n", source_line(&self.source))
    }
}

impl Report for SourceList {
    fn human(&self) -> String {
        if self.sources.is_empty() {
            return "no sources stored\n".into();
        }
        self.sources.iter().map(|s| source_line(s) + "\n").collect()
    }
}

impl Report for Source {
    fn human(&self) -> String {
        let mut out = format!(
            "source {} ({}), captured {}\n  from: {}\n",
            self.id,
            self.kind(),
            self.captured_at,
            origin(&self.origin)
        );
        for part in &self.parts {
            let _ = writeln!(
                out,
                "  {}  {}  {} bytes",
                part.name, part.media_type, part.bytes
            );
        }
        out
    }
}

impl Report for TasksGenerated {
    fn human(&self) -> String {
        let mut out = format!(
            "task set {}: {} task(s) from {} text part(s)\n",
            self.task_set, self.tasks, self.parts
        );
        for (kind, counts) in &self.per_kind {
            let _ = writeln!(
                out,
                "  {kind}: {} admitted, {} rejected",
                counts.admitted, counts.rejected
            );
        }
        if self.sections > 0 {
            let _ = writeln!(
                out,
                "  {} queued concept section(s) generated from",
                self.sections
            );
        }
        if !self.rejected.is_empty() {
            let _ = writeln!(out, "  rejected: {}", tally(&self.rejected));
        }
        // The first few say why; `--json` carries every one, whole.
        for r in self.rejections.iter().take(REJECTIONS_SHOWN) {
            let detail: String = r.detail.chars().take(REJECTION_DETAIL_CHARS).collect();
            let _ = writeln!(out, "    {} {}: {}", r.kind, r.reason, detail.trim());
        }
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "  stopped: {why}");
        }
        out
    }
}

/// Rejections a task report shows with their detail.
const REJECTIONS_SHOWN: usize = 3;
/// Characters of a rejection's detail shown.
const REJECTION_DETAIL_CHARS: usize = 300;

impl Report for TaskSetList {
    fn human(&self) -> String {
        if self.task_sets.is_empty() {
            return "no task sets stored\n".into();
        }
        self.task_sets
            .iter()
            .map(|s| format!("{}  {} task(s)  {}\n", s.id, s.tasks, s.name))
            .collect()
    }
}

impl Report for TaskShow {
    fn human(&self) -> String {
        match self {
            Self::Set { id, name, tasks } => {
                let mut out = format!("task set {id}: {name}\n");
                for task in tasks {
                    let _ = writeln!(
                        out,
                        "  {}  {:<14}  {}",
                        task.task,
                        task.kind,
                        one_line(&task.instruction)
                    );
                }
                out
            }
            Self::Task { task } => format!(
                "task {} ({}) in {}\n{}\n{} privileged item(s), {} evidence span(s)\n",
                task.task.id,
                task.task.kind,
                task.environment.kind,
                task.instruction,
                task.privileged.len(),
                task.evidence.len()
            ),
        }
    }
}

/// `text` on one line, cut to a readable width.
fn one_line(text: &str) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > 80 {
        format!("{}...", flat.chars().take(77).collect::<String>())
    } else {
        flat
    }
}

impl Report for Solved {
    fn human(&self) -> String {
        let mut out = format!(
            "experience set {}: {} solved by {}, {} answered\n  runs ended: {}\n",
            self.experience_set,
            self.solved,
            self.solver,
            self.answered,
            tally(&self.conclusions)
        );
        for skipped in &self.skipped {
            let _ = writeln!(out, "  skipped {}: {}", skipped.task, skipped.reason);
        }
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "  stopped: {why}");
        }
        out
    }
}

impl Report for Verified {
    fn human(&self) -> String {
        let mut out = format!(
            "{} experience(s) graded: verdicts {} pass, {} fail, {} abstain; now {} pass, {} fail, {} undecided\n",
            self.experiences,
            self.verdicts.pass,
            self.verdicts.fail,
            self.verdicts.abstain,
            self.passed,
            self.failed,
            self.undecided
        );
        for item in &self.unverified {
            let _ = writeln!(out, "  not graded {}: {}", item.experience, item.reason);
        }
        out
    }
}

impl Report for Critiqued {
    fn human(&self) -> String {
        let mut out = format!(
            "{} failure(s) critiqued, {} repaired ({} not failed, left alone)\n  loops stopped: {}\n  experience set {} (revisions: {})\n",
            self.critiqued,
            self.repaired,
            self.not_failed,
            tally(&self.stops),
            self.experience_set,
            self.revisions
        );
        for item in &self.unrepaired {
            let _ = writeln!(out, "  not repaired {}: {}", item.experience, item.reason);
        }
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "  stopped: {why}");
        }
        out
    }
}

fn share(value: Option<f64>) -> String {
    value.map_or("not measured".into(), |v| format!("{v:.3}"))
}

impl Report for Calibrated {
    fn human(&self) -> String {
        let c = &self.calibration;
        format!(
            "judge {} measured on {} labelled experience(s): pass precision {}, fail precision {}, abstains {}\n  calibration {} kept at {}\n",
            self.judge,
            c.n,
            share(c.precision_pass),
            share(c.precision_fail),
            share(c.abstain_rate),
            c.id,
            self.path.display()
        )
    }
}

impl Report for SetList {
    fn human(&self) -> String {
        if self.experience_sets.is_empty() {
            return "no experience sets stored\n".into();
        }
        self.experience_sets
            .iter()
            .map(|s| format!("{}  {} experience(s)  {}\n", s.id, s.experiences, s.name))
            .collect()
    }
}

fn experience_line(line: &ExperienceLine) -> String {
    let strength = line
        .decision
        .strength
        .map_or(String::new(), |s| format!(" ({})", word(&s)));
    format!(
        "{}  {:<10}  {}{strength}  {}",
        line.id, line.kind, line.decision.decision, line.solver
    )
}

impl Report for ExperienceShow {
    fn human(&self) -> String {
        match self {
            Self::Set {
                id,
                name,
                experiences,
            } => {
                let mut out = format!("experience set {id}: {name}\n");
                for line in experiences {
                    let _ = writeln!(out, "  {}", experience_line(line));
                }
                out
            }
            Self::Experience {
                id,
                decision,
                experience,
                annotations,
            } => format!(
                "experience {id} ({}), {} by {}\n{}\n  answer: {}\n  {} annotation(s)\n",
                experience.task.kind,
                decision.decision,
                experience.provenance.solver,
                experience.instruction,
                experience
                    .final_output
                    .as_deref()
                    .map_or("(none)".into(), one_line),
                annotations.len()
            ),
            Self::Graph {
                nodes,
                edges,
                truncated,
            } => {
                let mut out = String::new();
                for node in nodes {
                    let _ = writeln!(out, "{}", experience_line(node));
                }
                for edge in edges {
                    let _ = writeln!(out, "  {} {} {}", edge.from, word(&edge.relation), edge.to);
                }
                if *truncated {
                    out.push_str("  (cut: the graph is larger than shown)\n");
                }
                out
            }
        }
    }
}

impl Report for Replayed {
    fn human(&self) -> String {
        let mut out = String::new();
        for experience in &self.experiences {
            let verdict = if experience.reproduced {
                "reproduced"
            } else {
                "DIVERGED"
            };
            let _ = writeln!(
                out,
                "experience {} in {}: {verdict}, {} call(s)",
                experience.id,
                experience.environment,
                experience.calls.len()
            );
            for call in &experience.calls {
                let how = match &call.replay {
                    CallReplay::Reproduced => "reproduced".to_string(),
                    CallReplay::Diverged { fields } => format!("diverged in {}", fields.join(", ")),
                    CallReplay::NotReplayable { reason } => format!("not replayable: {reason}"),
                };
                let _ = writeln!(out, "  {}: {how}", call.call_id);
            }
        }
        out
    }
}

impl Report for Built {
    fn human(&self) -> String {
        format!(
            "dataset {}: {} {} record(s), objective {}, format {}\n  file: {}\n  excluded: {}\n",
            self.dataset,
            self.records,
            self.view,
            word(&self.objective),
            word(&self.format),
            self.path.display(),
            tally(&self.excluded)
        )
    }
}

impl Report for Exported {
    fn human(&self) -> String {
        format!(
            "dataset {} exported to {}\n  manifest: {}\n",
            self.dataset,
            self.path.display(),
            self.manifest.display()
        )
    }
}

fn loss(value: Option<f32>) -> String {
    value.map_or("not measured".into(), |v| format!("{v:.4}"))
}

impl Report for Candidate {
    fn human(&self) -> String {
        let parent = self
            .parent
            .as_ref()
            .map_or("no release".to_string(), |id| format!("release {id}"));
        let replay = self.replay.as_ref().map_or("none".to_string(), |r| {
            format!(
                "{} record(s) from {} earlier release(s), fraction {}",
                r.records,
                r.sources.len(),
                r.fraction
            )
        });
        let measured = match &self.preference {
            Some(preference) => {
                let (accuracy, margin) = preference
                    .held_out_score
                    .as_ref()
                    .map_or((None, None), |s| (s.accuracy, s.mean_margin));
                let reference = preference
                    .reference_adapter
                    .as_deref()
                    .map_or("the base".to_string(), |a| format!("adapter {a}"));
                format!(
                    "held-out preference against {reference}: accuracy {}, mean margin {} nats (beta {})",
                    loss(accuracy),
                    loss(margin),
                    preference.beta
                )
            }
            None => format!(
                "held-out loss: base {}, candidate {}",
                loss(self.base_score.and_then(|s| s.loss)),
                loss(self.tuned_score.and_then(|s| s.loss))
            ),
        };
        let regime = match self.regime {
            Regime::Sft => "supervised fine-tuning",
            Regime::Dpo => "preference (DPO) fine-tuning",
        };
        format!(
            "candidate {} trained by {regime} from {} ({parent}) on {} record(s)\n  replayed: {replay}\n  adapter: {} ({})\n  {measured}\n  not released: `splinter release {}` runs the gate\n",
            self.candidate,
            self.from,
            self.records,
            self.adapter.display(),
            self.adapter_digest,
            self.candidate
        )
    }
}

impl Report for RunList {
    fn human(&self) -> String {
        if self.runs.is_empty() {
            return "no runs recorded\n".into();
        }
        self.runs
            .iter()
            .map(|r| {
                format!(
                    "{}  {:<10}  {}  {}\n",
                    r.id,
                    r.status.as_str(),
                    r.started_at,
                    r.command
                )
            })
            .collect()
    }
}

impl Report for Run {
    fn human(&self) -> String {
        let mut out = format!(
            "run {} ({}): {}, started {}, updated {}\n",
            self.id,
            self.command,
            self.status.as_str(),
            self.started_at,
            self.updated_at
        );
        for stage in &self.stages {
            let _ = writeln!(out, "  {} finished {}", stage.stage, stage.finished_at);
        }
        if let Some(error) = &self.error {
            let _ = writeln!(out, "  error: {error}");
        }
        out
    }
}

impl Report for CancelRequested {
    fn human(&self) -> String {
        format!(
            "cancel requested for {}; it stops at its next check (runs show {})\n",
            self.run, self.run
        )
    }
}

impl Report for Status {
    fn human(&self) -> String {
        let c = &self.counts;
        let mut out = format!(
            "state:  {}\npolicy: {} = {} ({}{})\nstores: {} source(s), {} task set(s) of {} task(s), {} experience(s) in {} set(s), {} dataset(s), {} candidate(s)\n",
            self.state.display(),
            self.policy.reference,
            self.policy.model,
            self.policy
                .adapter
                .as_ref()
                .map_or("no adapter: nothing is released".into(), |a| a.display().to_string()),
            self.policy
                .release
                .as_ref()
                .map_or(String::new(), |r| format!(", release {r}")),
            c.sources,
            c.task_sets,
            c.tasks,
            c.experiences,
            c.experience_sets,
            c.datasets,
            c.candidates
        );
        if self.recent_runs.is_empty() {
            out.push_str("runs:   none recorded\n");
        }
        for run in &self.recent_runs {
            let _ = writeln!(
                out,
                "run:    {}  {:<10}  {}",
                run.id,
                run.status.as_str(),
                run.command
            );
        }
        out.push_str(&self.concepts.human());
        out
    }
}

impl Report for Answer {
    fn human(&self) -> String {
        format!("{}\n", self.answer.trim_end())
    }
}
