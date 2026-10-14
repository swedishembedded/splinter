// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What `learn` prints as text: the plan of a dry run, each stage's line
//! as it finishes, and the whole run's report.

use std::fmt::Write as _;

use splinter_sdk::learn::{LearnPlan, LearnReport, Learned};
use splinter_sdk::variants::VariantsGenerated;

use crate::output::{source_line, tally, word, Report};

/// What the plan stage found and chose, on one line.
fn plan_line(summary: &serde_json::Value) -> String {
    let (survey, plan) = (&summary["survey"], &summary["plan"]);
    let kinds: Vec<&str> = plan["kinds"]
        .as_array()
        .map(|k| k.iter().filter_map(|k| k.as_str()).collect())
        .unwrap_or_default();
    let persona = plan["persona"]
        .as_str()
        .map_or(String::new(), |p| format!(" as {p}"));
    let mode = if plan["distill"] == true {
        ", distil from a teacher"
    } else {
        ""
    };
    format!(
        "{} part(s), {} section(s) read as advice, {} as the writer judging -> {}{persona}{mode}: {}",
        survey["parts"],
        survey["advice_sections"],
        survey["judgment_sections"],
        kinds.join(", "),
        plan["rationale"].as_str().unwrap_or("")
    )
}

/// What the exam found, on one line: the judge's standing, each arm's judged
/// and grounded results, and the paired test.
fn exam_line(summary: &serde_json::Value) -> String {
    if let Some(why) = summary["not_run"].as_str() {
        return format!("not run: {why}");
    }
    let ran = &summary["ran"];
    let arm = |a: &serde_json::Value| {
        format!(
            "{}/{} judged right ({} unanswered), {}/{} invented a specific",
            a["judged_right"], a["judged"], a["unanswered"], a["invented"], a["checked"]
        )
    };
    let judge = &ran["judge"];
    let trust = if judge["trusted"] == true {
        "trusted"
    } else {
        "NOT trusted, so no claim"
    };
    let won = |t: &serde_json::Value, against: &str| {
        t.as_object().map_or(String::new(), |t| {
            format!(
                "; the candidate won {} of {} discordant task(s) against {against}, p = {:.4}",
                t["candidate_wins"],
                t["discordant"],
                t["p_value"].as_f64().unwrap_or(1.0)
            )
        })
    };
    let interval = ran["paired_interval"]
        .as_object()
        .map_or(String::new(), |i| {
            format!(
                " (gain per family {:.2}, interval {:.2} to {:.2})",
                i["mean"].as_f64().unwrap_or(0.0),
                i["low"].as_f64().unwrap_or(0.0),
                i["high"].as_f64().unwrap_or(0.0)
            )
        });
    let test = format!("{}{interval}", won(&ran["paired"], "the base"));
    let prompted = ran["prompted"].as_object().map_or(String::new(), |_| {
        format!(
            "; base asked under the goal {}{}",
            arm(&ran["prompted"]),
            won(&ran["paired_vs_prompted"], "it")
        )
    });
    let retrieval = ran["retrieval"].as_object().map_or(String::new(), |r| {
        let wins = ran["paired_retrieval"].as_object().map_or(String::new(), |t| {
            format!(
                "; retrieval won {} of {} discordant task(s), p = {:.4}",
                t["candidate_wins"],
                t["discordant"],
                t["p_value"].as_f64().unwrap_or(1.0)
            )
        });
        let right = |found: bool, key: &str| {
            r["by_task"].as_array().map_or(0, |tasks| {
                tasks
                    .iter()
                    .filter(|t| t["evidence_found"] == found && t[key] == true)
                    .count()
            })
        };
        format!(
            "; candidate with {} retrieved passage(s) {} (retrieval found the task's evidence for {}/{}; of those, right with the passages {} and alone {}; of the rest, with {} and alone {}){wins}",
            r["passages"],
            arm(&ran["retrieval"]["arm"]),
            r["hits"],
            r["tasks"],
            right(true, "with_passages"),
            right(true, "alone"),
            right(false, "with_passages"),
            right(false, "alone"),
        )
    });
    let warnings: String = ran["training_warnings"]
        .as_array()
        .map(|w| {
            w.iter()
                .filter_map(|w| w.as_str())
                .map(|w| format!("\nwarning: {w}"))
                .collect()
        })
        .unwrap_or_default();
    format!(
        "{} task(s) from {} source family(ies), judge {} {trust}: base {}; candidate {}{test}{prompted}{retrieval}{warnings}",
        ran["tasks"],
        ran["families"],
        judge["judge"].as_str().unwrap_or("?"),
        arm(&ran["base"]),
        arm(&ran["candidate"]),
    )
}

/// One line saying what a finished `learn` stage did, from its summary.
pub fn stage_line(stage: &str, summary: &serde_json::Value) -> String {
    if let Some(line) = crate::absorb_output::stage_line(stage, summary) {
        return line;
    }
    let field = |name: &str| match &summary[name] {
        serde_json::Value::Null => "?".to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    match stage {
        "sources" => format!("{} source(s)", summary.as_array().map_or(0, Vec::len)),
        "plan" => plan_line(summary),
        "exam" if !summary["ran"]["comparisons"].is_null() || summary["not_run"].is_string() => {
            crate::exam_output::powered_line(summary)
        }
        "exam" => exam_line(summary),
        "reserve" => format!(
            "{} of {} families reserved ({} could have been)",
            summary["families"].as_array().map_or(0, Vec::len),
            summary["total_families"],
            summary["examinable"]
        ),
        "exam-set" => format!(
            "{} task(s) over {} reserved families, frozen as {}",
            summary["tasks"].as_array().map_or(0, Vec::len),
            summary["families"].as_array().map_or(0, Vec::len),
            summary["id"].as_str().unwrap_or("?")
        ),
        "tasks" => format!("{} task(s) in {}", field("tasks"), field("task_set")),
        "solve" => format!(
            "{} solved, {} answered, in {}",
            field("solved"),
            field("answered"),
            field("experience_set")
        ),
        "verify" => format!(
            "{} passed, {} failed, {} undecided",
            field("passed"),
            field("failed"),
            field("undecided")
        ),
        "critique" => format!(
            "{} critiqued, {} repaired",
            field("critiqued"),
            field("repaired")
        ),
        "author" => summary["skipped"].as_str().map_or_else(
            || {
                format!(
                    "{} of {} passage(s) of the writer's own words fit their messages, {} refused",
                    summary["kept"], summary["tasks"], summary["refused"]
                )
            },
            |why| format!("skipped: {why}"),
        ),
        "teach" => format!(
            "{} task(s) never solved closed-book; the teacher's answer verified for {}",
            summary["solve"]["solved"], summary["verify"]["passed"]
        ),
        "frontier" => {
            let d = &summary["distribution"];
            format!(
                "{} kept: {} on the frontier, {} taught ({} always, {} never, {} unmeasured) in {}",
                d["frontier"].as_u64().unwrap_or(0) + d["taught"].as_u64().unwrap_or(0),
                d["frontier"],
                d["taught"],
                d["always"],
                d["never"],
                d["unmeasured"],
                field("frontier_task_set")
            )
        }
        "variants" => format!(
            "{} variant(s) of {} task(s){}",
            field("variants"),
            field("tasks"),
            match &summary["variant_set"] {
                serde_json::Value::String(set) => format!(" in {set}"),
                _ => String::new(),
            }
        ),
        "select" => {
            let s = &summary["selection"];
            format!(
                "{} of {} kept ({} repeats dropped) in {}",
                s["selected"].as_array().map_or(0, Vec::len),
                s["candidates"],
                s["duplicates"],
                field("experience_set")
            )
        }
        "dataset" => {
            let voice = summary["voice"].as_object().map_or(String::new(), |voice| {
                format!(
                    ", and {} record(s) of the writer's own text in {}",
                    voice["records"],
                    voice["dataset"].as_str().unwrap_or("?")
                )
            });
            format!(
                "{} record(s) in {}{voice}",
                field("records"),
                field("dataset")
            )
        }
        "rehearse" => format!(
            "{} task(s) for the base ({} by code, {} general by the base; {} refused as anchor \
             tasks, {} repeats): {} answered, {} record(s) in {}{}",
            field("tasks"),
            field("form_tasks"),
            field("general_tasks"),
            field("excluded_as_anchor"),
            field("repeated"),
            summary["solved"]["answered"],
            summary["dataset"]["records"],
            summary["dataset"]["dataset"].as_str().unwrap_or("?"),
            summary["stopped"]
                .as_str()
                .map_or(String::new(), |why| format!("; stopped: {why}"))
        ),
        "train" => format!("candidate {}", field("candidate")),
        "checkpoint" => match &summary["chosen"] {
            serde_json::Value::Null => field("why"),
            chosen => format!(
                "step {} of {} kept evaluation(s) chosen on the dev suite{}",
                chosen["step"],
                summary["scored"].as_array().map_or(0, Vec::len),
                if chosen["gate_passed"] == true {
                    ""
                } else {
                    " (NO evaluation passed the gate on invented specifics)"
                }
            ),
        },
        "policy" => match &summary["release"] {
            serde_json::Value::Null => {
                format!("policy:{} is the base: no release yet", field("alias"))
            }
            _ => format!("policy:{} is release {}", field("alias"), field("release")),
        },
        "release" => match &summary["release"] {
            serde_json::Value::Null => "the gate blocked the candidate: not released".into(),
            _ => format!("released {}", field("release")),
        },
        _ => summary.to_string(),
    }
}

impl Report for VariantsGenerated {
    fn human(&self) -> String {
        let mut out = match &self.variant_set {
            Some(set) => format!(
                "task set {set}: {} variant(s) of {} task(s)\n",
                self.variants, self.tasks
            ),
            None => format!("no variant admitted for {} task(s)\n", self.tasks),
        };
        if !self.ineligible.is_empty() {
            let _ = writeln!(out, "  not varied: {}", tally(&self.ineligible));
        }
        if !self.rejected.is_empty() {
            let _ = writeln!(out, "  rejected: {}", tally(&self.rejected));
        }
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "  stopped: {why}");
        }
        out
    }
}

impl Report for LearnPlan {
    fn human(&self) -> String {
        let sources: Vec<String> = self.sources.iter().map(word).collect();
        format!(
            "dry run - nothing written under {}\n  sources: {}\n  kinds:   {}\n  goal:    {}\n  budget:  {}\n  policy:  {}\n  writer:  {}\n  teacher: {}\n  stages:  {}\n",
            self.state.display(),
            sources.join(", "),
            self.planner.as_ref().map_or_else(
                || self.kinds.join(", "),
                |planner| format!("chosen by the planner ({planner}) from what the sources hold")
            ),
            self.goal.as_deref().unwrap_or("-"),
            self.budget_secs.map_or("none".into(), |s| format!("{s}s")),
            self.policy,
            self.generator,
            self.teacher,
            self.stages.join(" -> ")
        )
    }
}

impl Report for splinter_sdk::exam::Exam {
    fn human(&self) -> String {
        exam_line(&serde_json::to_value(self).unwrap_or(serde_json::Value::Null)) + "\n"
    }
}

/// What came of a run, as five separate facts: a run can finish and train a
/// candidate the gate then blocks, and none of that is the policy changing.
/// Each line says only what the report recorded; "improved" is the gate's
/// word, on the checks it names, and never inferred here.
fn outcome(report: &LearnReport) -> String {
    let run = match &report.stopped {
        Some(why) => format!("stopped before it finished: {why}"),
        None => "ran every stage it was asked to".to_string(),
    };
    let training = match &report.candidate {
        Some(candidate) => format!("candidate {} trained", candidate.candidate),
        None => "no candidate was trained".to_string(),
    };
    let (gate, release, policy) = match &report.release {
        None => (
            "not run".to_string(),
            "none".to_string(),
            format!("policy:{} unchanged", report.policy.alias),
        ),
        Some(released) => {
            let failed: Vec<&str> = [
                ("improvement", released.gate.improvement.passed),
                ("retention", released.gate.retention.passed),
                ("anchor", released.gate.anchor.passed),
                ("serve", released.gate.serve.passed),
            ]
            .into_iter()
            .filter_map(|(name, passed)| (!passed).then_some(name))
            .collect();
            let gate = if released.gate.passed {
                "passed improvement, retention, anchor and serve".to_string()
            } else {
                format!("blocked on {}", failed.join(", "))
            };
            match &released.release {
                Some(id) => (
                    gate,
                    format!("released {id}"),
                    format!("{} now points at {id}", released.alias),
                ),
                None => (
                    gate,
                    "not released".to_string(),
                    format!("{} unchanged", released.alias),
                ),
            }
        }
    };
    format!(
        "outcome  run:      {run}\n         training: {training}\n         gate:     {gate}\n         release:  {release}\n         policy:   {policy}\n"
    )
}

impl Report for LearnReport {
    fn human(&self) -> String {
        let mut out = outcome(self);
        let stage = |out: &mut String, name: &str, body: String| {
            for (i, line) in body.lines().enumerate() {
                let label = if i == 0 { name } else { "" };
                let _ = writeln!(out, "{label:<9}{line}");
            }
        };
        let policy = match &self.policy.release {
            Some(release) => format!("policy:{} = release {release}\n", self.policy.alias),
            None => format!("policy:{} = the base: no release yet\n", self.policy.alias),
        };
        stage(&mut out, "policy", policy);
        let sources: String = self.sources.iter().map(|s| source_line(s) + "\n").collect();
        stage(&mut out, "sources", sources);
        if let Some(planned) = &self.plan {
            let summary = serde_json::to_value(planned).unwrap_or(serde_json::Value::Null);
            stage(&mut out, "plan", plan_line(&summary));
        }
        if let Some(r) = &self.tasks {
            stage(&mut out, "tasks", r.human());
        }
        if let Some(r) = &self.solve {
            stage(&mut out, "solve", r.human());
        }
        if let Some(r) = &self.verify {
            stage(&mut out, "verify", r.human());
        }
        if let Some(r) = &self.teach {
            stage(&mut out, "teach", r.human());
        }
        if let Some(r) = &self.frontier {
            stage(&mut out, "frontier", r.human());
        }
        if let Some(r) = &self.variants {
            stage(&mut out, "variants", r.human());
        }
        if let Some(r) = &self.critique {
            stage(&mut out, "critique", r.human());
        }
        if let Some(r) = &self.select {
            stage(&mut out, "select", r.human());
        }
        if let Some(r) = &self.dataset {
            stage(&mut out, "dataset", r.human());
        }
        if let Some(r) = &self.voice {
            stage(&mut out, "voice", r.human());
        }
        if let Some(r) = &self.rehearsal {
            stage(&mut out, "rehearse", r.human());
        }
        if let Some(r) = &self.candidate {
            stage(&mut out, "train", r.human());
        }
        if let Some(exam) = &self.exam {
            let summary = serde_json::to_value(exam).unwrap_or(serde_json::Value::Null);
            stage(&mut out, "exam", exam_line(&summary));
        }
        if let Some(r) = &self.release {
            stage(&mut out, "release", r.human());
        }
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "stopped  {why}");
        }
        out
    }
}

impl Report for Learned {
    fn human(&self) -> String {
        match self {
            Self::Planned(plan) => plan.human(),
            Self::Ran(run) => run.human(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_sdk::measure::gate::{Check, GateConfig, GateReport};
    use splinter_sdk::release::Released;

    fn blocked() -> Released {
        Released {
            candidate: "cand-1".into(),
            alias: "policy:default".into(),
            champion: None,
            gate: GateReport::new(
                GateConfig::default(),
                Check::unmeasured("no held-out tasks"),
                Check::unmeasured("no earlier release"),
                Check::unmeasured("no anchor suite"),
                Check::unmeasured("no brain binary"),
            ),
            requeued: Vec::new(),
            warnings: Vec::new(),
            release: None,
            adapter: None,
        }
    }

    /// A run that stopped before training says so, and moves nothing.
    #[test]
    fn a_run_that_stopped_early_reports_no_candidate_and_an_unchanged_policy() {
        let report = LearnReport {
            stopped: Some("no task was worth training on".into()),
            ..LearnReport::default()
        };
        let text = outcome(&report);
        assert!(text.contains("stopped before it finished: no task was worth training on"));
        assert!(text.contains("no candidate was trained"));
        assert!(text.contains("gate:     not run") && text.contains("release:  none"));
        assert!(text.contains("unchanged"));
    }

    /// A gate that blocked the candidate is not a release, whatever else
    /// finished: the failed checks are named and the policy stays put.
    #[test]
    fn a_blocked_gate_names_what_failed_and_leaves_the_policy_unchanged() {
        let report = LearnReport {
            release: Some(blocked()),
            ..LearnReport::default()
        };
        let text = outcome(&report);
        assert!(
            text.contains("blocked on improvement, retention, anchor, serve"),
            "{text}"
        );
        assert!(text.contains("not released"), "{text}");
        assert!(text.contains("policy:default unchanged"), "{text}");
        assert!(!text.contains("points at"), "{text}");
    }

    /// The dataset line names the writer's own text beside the dialogues.
    #[test]
    fn the_dataset_stage_line_names_the_writers_text_beside_the_dialogues() {
        let line = stage_line(
            "dataset",
            &serde_json::json!({"records": 140, "dataset": "blake3:aa", "voice": {"records": 214, "dataset": "blake3:bb"}}),
        );
        assert!(
            line.contains("140 record(s) in blake3:aa")
                && line.contains("214 record(s) of the writer's own text in blake3:bb"),
            "{line}"
        );
        let alone = stage_line(
            "dataset",
            &serde_json::json!({"records": 140, "dataset": "blake3:aa", "voice": null}),
        );
        assert!(!alone.contains("writer"), "{alone}");
    }

    #[test]
    fn the_plan_stage_says_what_was_chosen_and_why() {
        let line = stage_line(
            "plan",
            &serde_json::json!({
                "survey": {"parts": 300, "advice_sections": 412},
                "plan": {"persona": "Thomas Jefferson", "kinds": ["advise", "recall"], "distill": true, "rationale": "the letters hold advice"}
            }),
        );
        assert!(line.contains("advise, recall"), "{line}");
        assert!(
            line.contains("distil") && line.contains("Thomas Jefferson"),
            "{line}"
        );
        assert!(
            line.contains("300 part(s)") && line.contains("412"),
            "{line}"
        );
        assert!(line.contains("the letters hold advice"), "{line}");
    }
}
