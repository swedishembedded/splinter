// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from a writer's own
// text whose held-out measurement cannot leak, for its clients. If your
// team needs expertise in training a model on a person's voice and proving
// what it was measured on was never trained on, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the `voice` view turns the writer's own text into records of the
//! shape the policy is trained and asked in, with no model in the loop, and
//! such a record is held out with the family of the letter it prints.
//!
//! Two editions print one letter. A conversation was written from one
//! print; the other print reaches the training set only through the voice
//! view. Both records name the same family, so when the conversation's
//! family is held out, the other print's text is held out with it: the exam
//! never asks about a letter the policy was trained on, under either name.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;

use common::{scratch_context, Scratch, Scripted};
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::experience::{Environment, Experience, Provenance, Span, Task};
use splinter_core::prompt::persona_prompt;
use splinter_core::source::PartRef;
use splinter_data::holdout::holdout_split_records;
use splinter_data::Exclusion;
use splinter_orchestrator::Context;
use splinter_pipelines::datasets::{build, BuildRequest, ViewName};
use splinter_pipelines::sources::{self, SourceTarget};
use splinter_store::experiences::ExperienceSet;

/// `i` spelled in letters, so a word carries no digit and the text is prose.
fn letters(mut i: usize) -> String {
    let mut out = String::new();
    loop {
        out.push(char::from(b'a' + u8::try_from(i % 26).unwrap()));
        i /= 26;
        if i == 0 {
            return out;
        }
    }
}

/// A letter of `n` words no other seed shares, under an address line.
fn letter(seed: &str, n: usize) -> String {
    let words: Vec<String> = (0..n).map(|i| format!("{seed}x{}", letters(i))).collect();
    format!("To a friend\nMonticello, 1800\n\n{}.\n", words.join(" "))
}

/// The materials: `count` letters, the first printed twice (the second print
/// with a few words more), captured as one source.
fn materials(scratch: &Scratch, ctx: &Context, count: usize) -> splinter_core::source::SourceId {
    let dir = scratch.0.join("letters");
    std::fs::create_dir_all(&dir).unwrap();
    for n in 0..count {
        std::fs::write(dir.join(format!("letter-{n}.txt")), letter(&letters(n), 80)).unwrap();
    }
    let reprint = format!(
        "{} and a few words more.\n",
        letter("a", 80).trim_end_matches(".\n")
    );
    std::fs::write(dir.join("letter-0-reprint.txt"), reprint).unwrap();
    let target = SourceTarget::from_learn_arg(&dir.display().to_string()).unwrap();
    sources::add(ctx, &target).unwrap().source.id
}

/// A passed conversation written from `part` of `source`, stored with its
/// verdict.
fn conversation(
    ctx: &Context,
    source: &splinter_core::source::SourceId,
    part: &str,
    n: usize,
) -> splinter_core::experience::ExperienceId {
    let found = ctx.sources().get_source(source).unwrap();
    let found = found.part(part).unwrap();
    let span = Span::in_part(
        PartRef {
            source: source.clone(),
            name: part.to_string(),
        },
        found.content.clone(),
        0,
        found.bytes,
    )
    .unwrap();
    let task = Task::new(
        "converse",
        vec![span],
        Environment::closed_book(),
        format!("What do you make of the matter in letter {n}?"),
        vec![],
    )
    .unwrap();
    let experience = Experience::answered_without_a_run(
        task,
        &format!("What I make of it is in letter {n}."),
        Provenance::new("scripted/teacher", ctx.clock()),
    )
    .unwrap();
    let store = ctx.experiences();
    let id = store.put(&experience).unwrap();
    store
        .annotate(&Annotation {
            experience: id.clone(),
            producer: Producer {
                name: "scripted/judge".into(),
                version: "1".into(),
            },
            body: AnnotationBody::Verdict {
                outcome: Outcome::Pass,
                strength: Strength::Judged,
                evidence: serde_json::Value::Null,
            },
        })
        .unwrap();
    id
}

#[test]
fn the_writers_text_is_held_out_with_the_family_of_the_letter_it_prints() {
    let (scratch, ctx) = scratch_context("voice", Scripted::new(|_| String::new()), false);
    let source = materials(&scratch, &ctx, 12);
    // Conversations from the first print of each letter; the reprint of
    // letter 0 is in no conversation's evidence.
    let members: Vec<_> = (0..12)
        .map(|n| conversation(&ctx, &source, &format!("letter-{n}.txt"), n))
        .collect();
    let set = ctx
        .experiences()
        .put_set(&ExperienceSet {
            name: "conversations".into(),
            members,
        })
        .unwrap();
    let prompt = persona_prompt("The Writer");
    let dialogues = build(
        &ctx,
        &BuildRequest {
            sets: vec![set.clone()],
            view: ViewName::SftFinal,
            strip: None,
            min_strength: Some(Strength::Judged),
            system_prompt: Some(prompt.clone()),
            export_only: false,
            limit: None,
        },
    )
    .unwrap();
    let voice = build(
        &ctx,
        &BuildRequest {
            sets: vec![set],
            view: ViewName::Voice,
            strip: None,
            min_strength: None,
            system_prompt: Some(prompt.clone()),
            export_only: false,
            limit: None,
        },
    )
    .unwrap();
    assert_eq!(dialogues.records, 12);
    // Every text part, the reprint among them, became a record of the
    // writer's own words under the persona, with the part it prints named
    // and no model's experience behind it.
    assert_eq!(voice.records, 13, "{:?}", voice.excluded);
    let lines: Vec<serde_json::Value> = std::fs::read_to_string(&voice.path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    for record in &lines {
        assert_eq!(record["messages"][0]["content"], prompt);
        assert_eq!(record["messages"][1]["role"], "user");
        assert_eq!(record["messages"][2]["train"], true);
        assert!(record["metadata"]["experiences"].is_null());
        assert_eq!(record["metadata"]["sources"].as_array().unwrap().len(), 1);
        assert_eq!(record["metadata"]["view"], "voice");
        assert!(record["metadata"]["group"].is_string(), "{record}");
    }
    let reprint = lines
        .iter()
        .find(|r| {
            r["messages"][2]["content"]
                .as_str()
                .unwrap()
                .contains("a few words more")
        })
        .unwrap();
    let first_print = lines
        .iter()
        .find(|r| {
            let answer = r["messages"][2]["content"].as_str().unwrap();
            answer.contains("axa ") && !answer.contains("a few words more")
        })
        .unwrap();
    assert_eq!(
        reprint["metadata"]["group"], first_print["metadata"]["group"],
        "two prints of one letter are one family"
    );

    // The dialogue about letter 0 and both prints of it name one family,
    // whichever dataset the record is in.
    let dialogue_lines: Vec<serde_json::Value> = std::fs::read_to_string(&dialogues.path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let about_zero = dialogue_lines
        .iter()
        .find(|r| {
            r["messages"][1]["content"]
                .as_str()
                .unwrap()
                .contains("letter 0?")
        })
        .unwrap();
    assert_eq!(
        about_zero["metadata"]["group"],
        reprint["metadata"]["group"]
    );

    // Trained together, the split holds the newest families of the
    // dialogues out, and every print of their letters with them: nothing
    // trained on prints a held-out letter.
    let mut records: Vec<String> = Vec::new();
    records.extend(
        std::fs::read_to_string(&dialogues.path)
            .unwrap()
            .lines()
            .map(String::from),
    );
    records.extend(
        std::fs::read_to_string(&voice.path)
            .unwrap()
            .lines()
            .map(String::from),
    );
    let (train, held) = holdout_split_records(&records).unwrap();
    let group_of = |line: &String| -> String {
        serde_json::from_str::<serde_json::Value>(line).unwrap()["metadata"]["group"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let held_families: BTreeSet<String> = held.iter().map(|l| group_of(l)).collect();
    let trained_families: BTreeSet<String> = train.iter().map(|l| group_of(l)).collect();
    assert!(
        held_families.is_disjoint(&trained_families),
        "{held_families:?}"
    );
    assert!(
        held.iter().any(|l| l.contains("\"view\":\"voice\"")),
        "the writer's text of a held-out family is held out"
    );
    assert!(
        held.iter().any(|l| l.contains("\"view\":\"sft-final\"")),
        "the families are the dialogues'"
    );
    assert_eq!(voice.excluded.get(&Exclusion::Duplicate), None);
}

#[test]
fn the_voice_dataset_is_limited_to_a_spread_of_the_corpus() {
    let (scratch, ctx) = scratch_context("voice-limit", Scripted::new(|_| String::new()), false);
    let source = materials(&scratch, &ctx, 12);
    let members = vec![conversation(&ctx, &source, "letter-0.txt", 0)];
    let set = ctx
        .experiences()
        .put_set(&ExperienceSet {
            name: "one conversation".into(),
            members,
        })
        .unwrap();
    let voice = build(
        &ctx,
        &BuildRequest {
            sets: vec![set],
            view: ViewName::Voice,
            strip: None,
            min_strength: None,
            system_prompt: None,
            export_only: false,
            limit: Some(5),
        },
    )
    .unwrap();
    assert_eq!(voice.records, 5);
    assert_eq!(voice.excluded.get(&Exclusion::OverLimit), Some(&8));
}
