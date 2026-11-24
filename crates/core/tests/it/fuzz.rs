// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for data models that refuse a
// forged or damaged record by name instead of crashing, for its clients. If
// your team needs expertise in content-addressed data and property-based
// testing then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec, by property: the core types read from outside are answered, a value
//! or an error, however they are damaged, and never panic; and what is
//! accepted keeps the type's invariant:
//!
//! * a task definition that validates is the task its own content
//!   addresses: rebuilding it from its fields gives the same task;
//! * a checkpoint manifest accepted names only relative paths that never
//!   leave the checkpoint;
//! * a digest accepted reads back as the text it was;
//! * a reward accepted is finite and in `[0, 1]`;
//! * a person's words or a model's reply (what is taught, a speech
//!   lesson's directive, confirmation or feedback), any Unicode text, is
//!   answered.
#![allow(clippy::unwrap_used)]

use proptest::prelude::*;
use serde_json::{json, Value};
use splinter_conformance::fuzz::{config, damaged_bytes, damaged_json, damaged_text, json_value};
use splinter_core::checkpoint::{CheckpointManifest, JobPlan};
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, Privileged, PrivilegedKind, Task};
use splinter_core::factory::trial::parse_reward;
use splinter_core::{speech_lesson, teaching};

fn seed_task() -> Task {
    Task::new(
        "workspace-change",
        Vec::new(),
        Environment::new(Environment::WORKSPACE, json!({"snapshot": "tree-1"})),
        "Fix the parser.",
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: "fn parse() {}".into(),
            span: None,
        }],
    )
    .unwrap()
    .with_concepts(["parsing"])
    .unwrap()
}

const MANIFEST: &str = r#"{"job":"run","plan":"{\"job\":\"run\",\"world_size\":2}","step":3,"world_size":2,"files":[{"role":"weights","path":"state/weights","len":1,"sha256":"sha256:1111111111111111111111111111111111111111111111111111111111111111"},{"role":"optimizer","path":"state/optimizer","len":2,"sha256":"sha256:2222222222222222222222222222222222222222222222222222222222222222"}]}"#;
const PLAN: &str = r#"{"job":"run","world_size":2,"tp":1,"pp":1,"dp":2,"coordinator":"127.0.0.1:9000","handshake_timeout_ms":1000,"op_timeout_ms":1000}"#;

/// What a person says to an assistant, or a model replies: an opener the
/// parsers read, then any text (Unicode included, where a byte offset can
/// fall inside a character).
fn spoken() -> impl Strategy<Value = String> {
    let opener = prop::sample::select(vec![
        "",
        "Actually, ",
        "That's wrong, ",
        "Remember that ",
        "From now on ",
        "Here is how to deploy: ",
        "Here is how to deploy: first, ",
        "Talk as follows: ",
        "Pronounce ",
        "say \"",
        "yes",
        "no, ",
        "\u{2019}",
    ]);
    (opener, "\\PC{0,40}").prop_map(|(o, rest)| format!("{o}{rest}"))
}

#[test]
fn every_seed_is_accepted_as_written() {
    seed_task().validate().unwrap();
    CheckpointManifest::parse(MANIFEST.as_bytes()).unwrap();
    JobPlan::parse(PLAN).unwrap();
}

/// The minimal inputs the property found: a letter whose lowercase is
/// longer in UTF-8 (`Ⱥ`, 2 bytes, lowercases to `ⱥ`, 3) moved every offset
/// the parsers took from the lowercased text, and they cut the original
/// outside it.
#[test]
fn a_letter_whose_lowercase_is_longer_is_read_like_any_other() {
    assert!(teaching::parse_teaching("From now on Ⱥ").is_none());
    let rule = teaching::parse_teaching("From now on always greet Ⱥnna first.").unwrap();
    assert!(format!("{rule:?}").contains("greet Ⱥnna first"), "{rule:?}");
    let fact = teaching::parse_teaching("Actually, ȺȺȺ the server is atlas.").unwrap();
    assert!(format!("{fact:?}").contains("ȺȺȺ the server"), "{fact:?}");
    let procedure = teaching::parse_teaching("Here is how to deploy: FIRSTȺ build, then ȺTest it.");
    assert!(
        format!("{procedure:?}").contains("Test it"),
        "{procedure:?}"
    );
    let directive = speech_lesson::parse_directive("Pronounce Ⱥdams as Add-ums").unwrap();
    assert!(format!("{directive:?}").contains("Ⱥdams"), "{directive:?}");
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn a_task_that_validates_is_the_task_its_content_addresses(
        doc in damaged_json(serde_json::to_value(seed_task()).unwrap())
    ) {
        let Ok(task) = serde_json::from_value::<Task>(doc) else { return Ok(()) };
        if task.validate().is_err() {
            return Ok(());
        }
        let rebuilt = Task::new(
            task.task.kind.clone(),
            task.evidence.clone(),
            task.environment.clone(),
            task.instruction.clone(),
            task.privileged.clone(),
        )
        .unwrap()
        .with_concepts(task.concepts.clone())
        .unwrap();
        prop_assert_eq!(rebuilt, task);
    }

    #[test]
    fn a_checkpoint_manifest_accepted_never_leaves_the_checkpoint(
        bytes in damaged_bytes(MANIFEST.as_bytes().to_vec()),
        doc in damaged_json(serde_json::from_str(MANIFEST).unwrap()),
        plan in damaged_text(PLAN),
    ) {
        for candidate in [bytes, serde_json::to_vec(&doc).unwrap()] {
            if let Ok(manifest) = CheckpointManifest::parse(&candidate) {
                for file in &manifest.files {
                    let path = file.path.as_str();
                    prop_assert!(!path.is_empty() && !path.starts_with('/'), "{path:?}");
                    prop_assert!(path.split('/').all(|part| part != ".."), "{path:?}");
                }
            }
        }
        let _ = JobPlan::parse(&plan);
    }

    #[test]
    fn a_digest_accepted_reads_back_as_written(
        text in damaged_text(&Digest::sha256_of(b"x").to_string())
    ) {
        if let Ok(digest) = Digest::parse(&text) {
            prop_assert_eq!(digest.to_string(), text);
        }
    }

    #[test]
    fn a_reward_accepted_is_a_finite_share(value in json_value()) {
        if let Ok(reward) = parse_reward(&value) {
            prop_assert!(reward.is_finite() && (0.0..=1.0).contains(&reward), "{reward}");
        }
        if let Value::String(text) = &value {
            let _ = parse_reward(&Value::String(format!("0{text}")));
        }
    }

    #[test]
    fn what_a_person_or_a_model_says_is_answered(text in spoken()) {
        let _ = teaching::parse_teaching(&text);
        let _ = teaching::teaching_from_json(&text);
        let _ = speech_lesson::parse_directive(&text);
        let _ = speech_lesson::parse_confirmation(&text);
        let _ = speech_lesson::parse_retraction(&text);
        let _ = speech_lesson::parse_feedback(&text);
        let _ = speech_lesson::parse_comparison(&text);
    }
}
