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
    let field = |name: &str| match &summary[name] {
        serde_json::Value::Null => "?".to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    match stage {
        "sources" => format!("{} source(s)", summary.as_array().map_or(0, Vec::len)),
        "plan" => plan_line(summary),
        "exam" => exam_line(summary),
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

impl Report for LearnReport {
    fn human(&self) -> String {
        let mut out = String::new();
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
