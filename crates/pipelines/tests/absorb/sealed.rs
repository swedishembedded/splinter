// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The guard: a record that contains a sealed probe's question, or shares a
//! run of eight words with a probe beyond what its claim states, is refused
//! with the probe named.

use splinter_core::chat::WireMessage;
use splinter_data::{Objective, Projection, Record, RecordBody, RecordMetadata};
use splinter_pipelines::absorb::sealed::{LeakKind, SealedProbes};

use crate::fixtures::Outcome;

const STATEMENT: &str = "The Tessera dashboard listens on port 9090 behind the corporate proxy.";

fn probes() -> SealedProbes {
    SealedProbes::of([(
        "p-restart",
        "After a restart, which port should I open for the Tessera dashboard?",
        "Open port 9090, the Tessera dashboard listens on port 9090 behind the corporate proxy.",
    )])
}

fn record(user: &str, assistant: &str) -> Record {
    let message = |role: &str, content: &str, train: bool| WireMessage {
        role: role.into(),
        content: content.into(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        train,
    };
    Record {
        body: RecordBody::Chat {
            messages: vec![
                message(
                    "system",
                    "You answer plainly and remember what you are told.",
                    false,
                ),
                message("user", user, false),
                message("assistant", assistant, true),
            ],
        },
        metadata: RecordMetadata {
            experiences: Vec::new(),
            task: None,
            sources: Vec::new(),
            group: None,
            split: None,
            view: "session-claims".into(),
            objective: Objective::Sft,
        },
    }
}

fn projection(records: Vec<Record>) -> Projection {
    Projection {
        view: "session-claims".into(),
        objective: Objective::Sft,
        strip: None,
        min_strength: None,
        records,
        excluded: Default::default(),
        system_prompt: None,
        terms: None,
    }
}

#[test]
fn a_record_containing_a_probe_question_is_refused_naming_the_probe() -> Outcome {
    let leaking = record(
        "After a restart, which port should I open for the Tessera dashboard?",
        "Port 9090.",
    );
    let clean = record(
        "Which port does the Tessera dashboard listen on?",
        "Port 9090.",
    );
    let mut set = projection(vec![leaking, clean.clone()]);
    let refused = probes().remove_leaks(&mut set, |_| vec![STATEMENT.to_string()]);
    assert_eq!(set.records, vec![clean]);
    assert_eq!(refused.len(), 1);
    assert_eq!(refused[0].1.probe, "p-restart");
    assert_eq!(refused[0].1.kind, LeakKind::Question);
    assert!(refused[0].1.to_string().contains("p-restart"));
    Ok(())
}

#[test]
fn eight_shared_words_are_refused_unless_the_claims_own_statement_has_them() -> Outcome {
    // The probe's answer says more than the statement does; the record
    // repeats the part the statement lacks.
    let probes = SealedProbes::of([(
        "p-rota",
        "Who is on call this week?",
        "Ines is on call this week and hands the pager to Ravi on Friday at noon sharp.",
    )]);
    let leaking = record(
        "Who carries the pager?",
        "Ines hands the pager to Ravi on Friday at noon sharp, every week.",
    );
    let statement = "Ines is on call this week.";
    let refused = probes.check(
        &[
            "Who carries the pager?",
            "Ines hands the pager to Ravi on Friday at noon sharp, every week.",
        ],
        &[statement],
    );
    let leak = refused.ok_or_else(|| anyhow::anyhow!("an 8-gram of the probe is in the record"))?;
    assert_eq!(leak.probe, "p-rota");
    assert!(matches!(leak.kind, LeakKind::Overlap { .. }), "{leak:?}");
    let _ = leaking;

    // What the statement itself says is the point of the record and is not
    // held against it, even though the probe's answer says it too.
    let stated = SealedProbes::of([(
        "p-port",
        "Where do I reach it?",
        "The Tessera dashboard listens on port 9090 behind the corporate proxy.",
    )]);
    assert_eq!(
        stated.check(&["Which port?", STATEMENT], &[STATEMENT]),
        None
    );
    assert!(stated.check(&["Which port?", STATEMENT], &[]).is_some());
    Ok(())
}

#[test]
fn probe_files_are_json_lines_or_an_array_and_a_bad_line_names_itself() -> Outcome {
    let dir = tempfile::tempdir()?;
    let lines = dir.path().join("a.jsonl");
    std::fs::write(
        &lines,
        "{\"question\": \"Which port?\", \"reference\": \"9090\", \"name\": \"p1\"}\n\n{\"question\": \"Who?\"}\n",
    )?;
    let array = dir.path().join("b.json");
    std::fs::write(&array, "[{\"question\": \"Where?\", \"name\": \"p3\"}]")?;
    let loaded = SealedProbes::load(&[lines.clone(), array])?;
    assert!(!loaded.is_empty());
    assert_eq!(
        loaded
            .check(&["please tell me, Who?"], &[])
            .map(|l| l.probe),
        Some(format!("{}:2", lines.display()))
    );
    assert_eq!(
        loaded.check(&["Where?"], &[]).map(|l| l.probe),
        Some("p3".into())
    );

    let bad = dir.path().join("c.jsonl");
    std::fs::write(&bad, "{\"question\": \"ok\"}\nnot json\n")?;
    let error = SealedProbes::load(&[bad]).unwrap_err().to_string();
    assert!(
        error.contains("c.jsonl") && error.contains("line 2"),
        "{error}"
    );
    Ok(())
}
