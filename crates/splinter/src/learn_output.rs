// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What `learn` prints as text: the plan of a dry run, each stage's line
//! as it finishes, and the whole run's report.

use std::fmt::Write as _;

use splinter_campaign::learn::{LearnPlan, LearnReport, Learned};

use crate::output::{source_line, word, Report};

/// One line saying what a finished `learn` stage did, from its summary.
pub fn stage_line(stage: &str, summary: &serde_json::Value) -> String {
    let field = |name: &str| match &summary[name] {
        serde_json::Value::Null => "?".to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    match stage {
        "sources" => format!("{} source(s)", summary.as_array().map_or(0, Vec::len)),
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
        "dataset" => format!("{} record(s) in {}", field("records"), field("dataset")),
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

impl Report for LearnPlan {
    fn human(&self) -> String {
        let sources: Vec<String> = self.sources.iter().map(word).collect();
        format!(
            "dry run - nothing written under {}\n  sources: {}\n  kinds:   {}\n  goal:    {}\n  budget:  {}\n  policy:  {}\n  writer:  {}\n  teacher: {}\n  stages:  {}\n",
            self.state.display(),
            sources.join(", "),
            self.kinds.join(", "),
            self.goal.as_deref().unwrap_or("-"),
            self.budget_secs.map_or("none".into(), |s| format!("{s}s")),
            self.policy,
            self.generator,
            self.teacher,
            self.stages.join(" -> ")
        )
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
        if let Some(r) = &self.critique {
            stage(&mut out, "critique", r.human());
        }
        if let Some(r) = &self.select {
            stage(&mut out, "select", r.human());
        }
        if let Some(r) = &self.dataset {
            stage(&mut out, "dataset", r.human());
        }
        if let Some(r) = &self.candidate {
            stage(&mut out, "train", r.human());
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
