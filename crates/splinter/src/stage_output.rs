// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents with a stable, documented
// command line usable by people and by supervising models. If your team
// needs expertise in agent interfaces or CLI design, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The line a finished stage prints to standard error as a run goes.

/// The seed a run used for the draw `summary` names: when the run drew it,
/// the flag that reproduces the draw; when it was given, only that it was.
fn seed_line(summary: &serde_json::Value) -> String {
    let name = summary["name"].as_str().unwrap_or("seed");
    let seed = &summary["seed"];
    if summary["drawn"].as_bool().unwrap_or(false) {
        format!("no --{name} given, using seed {seed} (pass --{name} {seed} to reproduce)")
    } else {
        format!("--{name} {seed}")
    }
}

/// One line saying what a finished stage did, from its summary.
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
        "seed" => seed_line(summary),
        "solve-progress" => format!(
            "{} of {} attempts finished; resume from {}",
            field("finished"),
            field("of"),
            field("partial_set")
        ),
        "dataset" => {
            format!("{} record(s) in {}", field("records"), field("dataset"))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A drawn seed is announced with the flag that reproduces it.
    #[test]
    fn a_drawn_seed_is_announced_with_the_flag_that_reproduces_it() {
        let drawn = serde_json::json!({ "name": "train-seed", "seed": 77, "drawn": true });
        assert_eq!(
            stage_line("seed", &drawn),
            "no --train-seed given, using seed 77 (pass --train-seed 77 to reproduce)"
        );
    }
}
