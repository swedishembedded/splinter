// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in training models on the reasons
// behind decisions and in measuring those reasons, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a record teaches a reason the author gave in its own words. No record
//! has an empty reason, none repeats a phrase of the brief (in the voice of
//! the brief or of the author), and the examination reports, per arm, how
//! many reasons name the fact the deciding belief turns on, how many copy the
//! brief, and how many differ.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use serde_json::{json, Value};
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use synthetic_author::arms::Outcome;
use synthetic_author::binding::ModelIdentity;
use synthetic_author::cache::Cache;
use synthetic_author::dataset::records_for;
use synthetic_author::episode::{did_phrase, did_phrase_first_person, episodes_for};
use synthetic_author::exam::{Item, Version};
use synthetic_author::gates::{brief_phrase_copied, check_form, Article, Rejection};
use synthetic_author::lab::{exam_items, run_arm, Arm, ExamRun};
use synthetic_author::mine::Verified;
use synthetic_author::motives::grade;
use synthetic_author::persona::Persona;
use synthetic_author::scenario::Archetype;

fn verified(chose: Archetype, reasons: &[&str]) -> Verified {
    Verified {
        case: "the Dunmere granary".into(),
        briefing:
            "A matter of the winter stores, which the community must settle before the roads close."
                .into(),
        offer_made: false,
        facts: synthetic_author::mine::MinedFacts::default(),
        chose,
        open: vec![
            Archetype::ActAlone,
            Archetype::ConsultThenAct,
            Archetype::HoldAndWait,
        ],
        reasons: reasons.iter().map(|r| (*r).to_string()).collect(),
        dropped: 0,
    }
}

fn motivations(records: &[Value]) -> Vec<String> {
    records
        .iter()
        .map(|r| {
            let args: Value = serde_json::from_str(
                r["messages"][2]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap(),
            )
            .unwrap();
            args.as_object()
                .unwrap()
                .iter()
                .find(|(k, _)| !["choice", "course", "ruling"].contains(&k.as_str()))
                .and_then(|(_, v)| v.as_str())
                .unwrap()
                .to_string()
        })
        .collect()
}

#[test]
fn a_decision_without_a_quoted_reason_is_not_a_record() {
    for reasons in [&[][..], &[""][..], &["   "][..]] {
        let d = verified(Archetype::HoldAndWait, reasons);
        assert!(records_for(7, "g", &d).is_empty(), "{reasons:?}");
    }
}

#[test]
fn a_reason_that_repeats_a_phrase_of_the_brief_is_not_taught_in_any_voice() {
    let own = "We held back because the stores would not have lasted another winter.";
    for copied in [
        "I settled it myself, at once, without waiting.",
        "You settled it yourself, at once, without waiting.",
        "Then I put it before the assembly and acted on its word.",
    ] {
        let d = verified(Archetype::HoldAndWait, &[copied, own]);
        let reasons = motivations(&records_for(7, "g", &d));
        assert!(!reasons.is_empty());
        assert!(reasons.iter().all(|r| r == own), "{copied}: {reasons:?}");
        let only = verified(Archetype::HoldAndWait, &[copied]);
        assert!(records_for(7, "g", &only).is_empty(), "{copied}");
    }
    // Five words of a phrase are ordinary narrative; the sixth makes a quotation.
    let near = "I settled it myself, at once, and never regretted it.";
    assert!(brief_phrase_copied(near).is_none());
    assert!(brief_phrase_copied("I settled it myself, at once, without delay.").is_some());
}

#[test]
fn an_essay_in_the_authors_own_voice_that_repeats_the_brief_is_refused() {
    // seed: the episodes are those of one author with a decision to act alone
    let persona = Persona::generate(5);
    let eps = episodes_for(&persona, 5, 0, 12, &mut Vec::new());
    let episode = eps
        .iter()
        .find(|e| persona.decide(&e.scenario).choice == Archetype::ActAlone)
        .expect("one of twelve episodes is decided by acting alone");
    let chose = Archetype::ActAlone;
    assert_ne!(
        did_phrase(chose, &episode.case),
        did_phrase_first_person(chose, &episode.case)
    );
    let mut body = String::new();
    for e in &eps {
        body.push_str(&format!(
            "Of {} I will say this. I did it on my own authority, at once, and on my own.\n\n",
            e.case.name
        ));
        for _ in 0..40 {
            body.push_str("The weather held and the roads were dry that year. ");
        }
        body.push_str("\n\n");
    }
    let article = Article {
        title: "t".into(),
        body: format!(
            "{body}Of {} {}.\n\n",
            episode.case.name,
            did_phrase_first_person(chose, &episode.case)
        ),
    };
    assert!(matches!(
        check_form(&article, &persona, &eps, 100),
        Err(Rejection::CopiesBrief { .. })
    ));
}

fn items(persona: &Persona) -> Vec<Item> {
    exam_items(persona, persona.seed, 20, 2)
}

fn answering(item: &Item, reason: &str) -> Outcome {
    Outcome {
        id: item.id.clone(),
        correct: true,
        topic_seen: item.topic_seen,
        combo_seen: item.combo_seen,
        combo_trained: item.combo_trained,
        tool_seen: item.tool_seen,
        call: Some(json!({
            "name": item.schema.name,
            "arguments": { item.schema.reason: reason },
        })),
        item: String::new(),
        model: String::new(),
    }
}

#[test]
fn reasons_are_graded_by_code_for_the_deciding_fact_for_copying_and_for_variety() {
    let persona = Persona::generate(splinter_sdk::model::seed::draw().value());
    let all = items(&persona);
    let deciding = |item: &Item| persona.decide(&item.scenario).deciding.unwrap();

    // Every reason names the fact of the belief that decided its item.
    let naming: Vec<Outcome> = all
        .iter()
        .map(|i| {
            let term = deciding(i).fact().terms()[0];
            answering(i, &format!("It came down to the {term} of the matter."))
        })
        .collect();
    let graded = grade(&all, &naming, &persona);
    assert_eq!(graded.reasons, all.len());
    assert_eq!(graded.naming_deciding_fact, all.len());
    assert_eq!(graded.copying_brief, 0);

    // The same sentence for every item names the fact only where it happens to.
    let blank: Vec<Outcome> = all
        .iter()
        .map(|i| answering(i, "I did what seemed right."))
        .collect();
    let graded = grade(&all, &blank, &persona);
    assert_eq!(graded.naming_deciding_fact, 0);
    assert_eq!(graded.distinct, 1);

    // A reason that repeats the brief counts as copying, and a call without a
    // reason is not a reason.
    let mut copied = blank.clone();
    copied[0] = answering(&all[0], "I settled it myself, at once, without waiting.");
    copied[1].call = None;
    copied[2] = answering(&all[2], "");
    let graded = grade(&all, &copied, &persona);
    assert_eq!(graded.reasons, all.len() - 2);
    assert_eq!(graded.copying_brief, 1);
    assert_eq!(graded.distinct, 2);
}

/// Takes the first listed option and gives the same reason for every item.
struct Terse;

#[async_trait::async_trait]
impl ModelProvider for Terse {
    fn name(&self) -> &str {
        "terse"
    }
    fn model_name(&self) -> &str {
        "terse-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let tool = &req.tools[0];
        let required = tool.parameters["required"].as_array().unwrap();
        let (choice, reason) = (required[0].as_str().unwrap(), required[1].as_str().unwrap());
        let label = tool.parameters["properties"][choice]["enum"][0].clone();
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::ToolCall {
                index: 0,
                id: "c".into(),
                name: tool.name.clone(),
                arguments: json!({ choice: label, reason: "It seemed right." }).to_string(),
            }),
            Ok(ResponseEvent::Done),
        ])))
    }
}

#[test]
fn the_report_lists_what_each_arm_s_reasons_come_to() {
    let persona = Persona::generate(splinter_sdk::model::seed::draw().value());
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("synthetic-author-motives");
    let _ = std::fs::remove_dir_all(&dir);
    let cache = Cache::open(dir).unwrap();
    cache.write_persona(&persona).unwrap();
    let all = items(&persona);
    let run = ExamRun {
        articles: 20,
        per_cell: 2,
        max_tokens: 64,
        model: ModelIdentity {
            base_digest: "sha256:base".into(),
            adapter_digest: None,
            decoding: "greedy".into(),
            decoding_seed: None,
        },
        exam: Version::V1,
        adopt: false,
        slots: 1,
    };
    let provider: Arc<dyn ModelProvider> = Arc::new(Terse);
    run_arm(&cache, &persona, Arm::Base, &provider, &all, &run).unwrap();

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_synthetic-author"))
        .args(["report", "--arms", "base", "--cache"])
        .arg(cache.dir())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{out:?}");
    assert!(
        stdout.contains(&format!("| base | {0} | 0/{0} | 0/{0} | 1 |", all.len())),
        "{stdout}"
    );
}
