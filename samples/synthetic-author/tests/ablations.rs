// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in telling which part of a method
// accounts for a model's result, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: the switches of the mechanism ablations. Records can be made without
//! the fixed-wording briefings or with them only; retrieval is asked at two,
//! six and twelve paragraphs, over whole articles and over the training
//! records themselves; a principles arm also states each belief's verdicts.
//! Each is its own arm with its own name, and the exam manifest records what
//! it was asked with.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use serde_json::json;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use splinter_sdk::store::jsonl;
use synthetic_author::arm::{Arm, Over, Passages, Retrieval};
use synthetic_author::binding::{recorded, ModelIdentity};
use synthetic_author::cache::{Cache, StoredArticle};
use synthetic_author::dataset::{records_with, Briefings};
use synthetic_author::exam::{Item, Version};
use synthetic_author::lab::{
    build_dataset, exam_items, run_arm, system_of, ExamRun, Mined, MinedResult,
};
use synthetic_author::mine::{MinedFacts, Verified};
use synthetic_author::persona::Persona;
use synthetic_author::scenario::Archetype;
use synthetic_author::tools::TRAINING;
use synthetic_author::training::TrainArgs;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("synthetic-author-ablations-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn decision(case: &str, briefing: &str, facts: MinedFacts) -> Verified {
    Verified {
        case: case.into(),
        briefing: briefing.into(),
        offer_made: false,
        facts,
        chose: Archetype::HoldAndWait,
        open: vec![
            Archetype::ActAlone,
            Archetype::ConsultThenAct,
            Archetype::HoldAndWait,
        ],
        reasons: vec![format!(
            "We held back over {case} because the stores would not last."
        )],
        dropped: 0,
    }
}

fn all_facts() -> MinedFacts {
    MinedFacts {
        exceeds_authority: Some(true),
        urgent: Some(false),
        binds_future: Some(true),
        irreversible: Some(false),
        members_harmed: Some(true),
    }
}

fn users(records: &[serde_json::Value]) -> Vec<String> {
    records
        .iter()
        .map(|r| r["messages"][1]["content"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn records_can_be_made_without_the_fixed_wording_briefing_or_with_it_alone() {
    let read = decision(
        "the Dunmere granary",
        "The winter stores were short.",
        all_facts(),
    );
    let both = records_with(7, "g", &read, Briefings::Both);
    let reader = records_with(7, "g", &read, Briefings::ReaderOnly);
    let fixed = records_with(7, "g", &read, Briefings::FixedWordingOnly);
    assert_eq!(both.len(), 2 * TRAINING.len());
    assert_eq!(reader.len(), TRAINING.len());
    assert_eq!(fixed.len(), TRAINING.len());
    assert!(users(&reader)
        .iter()
        .all(|u| u.contains("The winter stores were short.")));
    assert!(users(&reader)
        .iter()
        .all(|u| !u.contains("A matter arose over")));
    assert!(users(&fixed)
        .iter()
        .all(|u| u.contains("A matter arose over")));
    assert!(users(&fixed)
        .iter()
        .all(|u| !u.contains("The winter stores were short.")));

    // Without every fact read there is no fixed-wording briefing to teach from.
    let partial = decision(
        "the Osmere ferry",
        "A crossing.",
        MinedFacts {
            urgent: None,
            ..all_facts()
        },
    );
    assert!(records_with(7, "g", &partial, Briefings::FixedWordingOnly).is_empty());
    assert_eq!(
        records_with(7, "g", &partial, Briefings::ReaderOnly).len(),
        TRAINING.len()
    );
}

#[test]
fn each_setting_of_the_dataset_is_kept_apart_and_recorded_with_the_adapter_it_trains() {
    let cache = Cache::open(scratch("dataset")).unwrap();
    let log = jsonl::Log::open(&cache.dir().join("decisions.jsonl")).unwrap();
    log.append(&Mined {
        article: 0,
        case: "the Dunmere granary".into(),
        result: MinedResult::Kept(decision("the Dunmere granary", "Stores.", all_facts())),
    })
    .unwrap();
    let mut files = BTreeSet::new();
    for briefings in [
        Briefings::Both,
        Briefings::ReaderOnly,
        Briefings::FixedWordingOnly,
    ] {
        let summary = build_dataset(&cache, 7, briefings).unwrap();
        assert_eq!(summary.briefings, briefings);
        assert!(summary.path.exists());
        files.insert(summary.path);
    }
    assert_eq!(files.len(), 3, "no setting overwrites another");

    let train = |args: &[&str]| {
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            train: TrainArgs,
        }
        let mut argv = vec!["x", "--cache", "c", "--base", "b"];
        argv.extend_from_slice(args);
        Cli::parse_from(argv).train
    };
    let records = scratch("train").join("train.jsonl");
    std::fs::write(&records, "{}\n").unwrap();
    let settings = train(&["--briefings", "fixed-wording-only"])
        .settings(1, &records)
        .unwrap();
    assert_eq!(
        serde_json::to_value(&settings).unwrap()["briefings"],
        "fixed-wording-only"
    );
    assert_eq!(
        serde_json::to_value(train(&[]).settings(1, &records).unwrap()).unwrap()["briefings"],
        "both"
    );
}

#[test]
fn every_arm_has_a_name_of_its_own_and_is_read_back_from_it() {
    let names: BTreeSet<&str> = Arm::ALL.iter().map(|a| a.name()).collect();
    assert_eq!(names.len(), Arm::ALL.len());
    for arm in Arm::ALL {
        assert_eq!(Arm::parse(arm.name()), Some(arm));
    }
    assert_eq!(Arm::parse("not-an-arm"), None);
    // The original retrieval arm keeps its name, so its outcomes stay valid.
    assert_eq!(Arm::Retrieval(Retrieval::ORIGINAL).name(), "retrieval");
    for (over, passages) in [
        (Over::Articles, Passages::Two),
        (Over::Articles, Passages::Twelve),
        (Over::Records, Passages::Two),
        (Over::Records, Passages::Six),
        (Over::Records, Passages::Twelve),
    ] {
        assert!(names.contains(Arm::Retrieval(Retrieval { over, passages }).name()));
    }
}

fn fixture(tag: &str) -> (Cache, Persona, Vec<Item>) {
    let persona = Persona::generate(splinter_sdk::model::seed::draw().value());
    let cache = Cache::open(scratch(tag)).unwrap();
    let items = exam_items(&persona, persona.seed, 20, 2);
    let situation = items[0].briefing.clone();
    // Plenty of passages that share the first item's words, so that every item
    // of the examination has more than twelve to be handed.
    for a in 0..2 {
        let body = (0..14)
            .map(|p| {
                format!("{situation} Passage {a}-{p} of the essay goes on to say more about it.")
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        cache
            .write_article(&StoredArticle {
                index: a,
                topic: 0,
                title: format!("essay {a}"),
                body,
                cases: vec![],
                attempts: 1,
            })
            .unwrap();
    }
    let log = jsonl::Log::open(&cache.dir().join("decisions.jsonl")).unwrap();
    for n in 0..14 {
        log.append(&Mined {
            article: 0,
            case: format!("case {n}"),
            result: MinedResult::Kept(decision(
                &format!("case {n}"),
                &format!("{situation} Case {n}."),
                all_facts(),
            )),
        })
        .unwrap();
    }
    (cache, persona, items)
}

#[test]
fn retrieval_hands_over_two_six_or_twelve_passages_of_articles_or_of_records() {
    let (cache, persona, items) = fixture("passages");
    for over in [Over::Articles, Over::Records] {
        for passages in [Passages::Two, Passages::Six, Passages::Twelve] {
            let arm = Arm::Retrieval(Retrieval { over, passages });
            let system = system_of(&cache, &persona, arm).unwrap()(&items[0]);
            // The plain system turn and one line saying what follows, then the passages.
            let handed = system.split("\n\n").count() - 2;
            assert_eq!(handed, passages.count(), "{}", arm.name());
            assert_eq!(
                system.contains("Decision:"),
                over == Over::Records,
                "{}: only records say what was decided",
                arm.name()
            );
        }
    }
}

#[test]
fn retrieval_over_records_needs_the_decisions_read_out_of_the_corpus() {
    let (_, persona, _) = fixture("needs-mine");
    let empty = Cache::open(scratch("no-decisions")).unwrap();
    let arm = Arm::Retrieval(Retrieval {
        over: Over::Records,
        passages: Passages::Six,
    });
    let message = match system_of(&empty, &persona, arm) {
        Ok(_) => panic!("records that were never read were retrieved from"),
        Err(e) => format!("{e:#}"),
    };
    assert!(message.contains("mine"), "{message}");
}

#[test]
fn the_principles_arm_with_verdicts_states_what_each_belief_makes_of_each_way() {
    let persona = Persona::generate(splinter_sdk::model::seed::draw().value());
    let plain = Arm::Principles.system(&persona);
    let with = Arm::PrinciplesWithVerdicts.system(&persona);
    assert!(
        with.starts_with(&plain),
        "it adds to the principles, it does not replace them"
    );
    for belief in &persona.beliefs {
        let rules = belief.rules();
        assert!(!rules.is_empty());
        for rule in rules {
            assert!(with.contains(&rule.describe()), "{}", rule.describe());
        }
    }
    assert!(
        !plain.contains("you rule out"),
        "the plain arm states no verdicts"
    );
}

/// Takes the first listed option.
struct FirstOption;

#[async_trait::async_trait]
impl ModelProvider for FirstOption {
    fn name(&self) -> &str {
        "first"
    }
    fn model_name(&self) -> &str {
        "first-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let tool = &req.tools[0];
        let choice = tool.parameters["required"][0].as_str().unwrap().to_string();
        let label = tool.parameters["properties"][&choice]["enum"][0].clone();
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::ToolCall {
                index: 0,
                id: "c".into(),
                name: tool.name.clone(),
                arguments: json!({ choice: label }).to_string(),
            }),
            Ok(ResponseEvent::Done),
        ])))
    }
}

fn run() -> ExamRun {
    ExamRun {
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
    }
}

#[test]
fn the_exam_manifest_records_what_each_retrieval_arm_was_asked_with() {
    let (cache, persona, items) = fixture("manifest");
    let provider: Arc<dyn ModelProvider> = Arc::new(FirstOption);
    let exam_dir = cache.dir().join("exam");
    let few: Vec<Item> = items.iter().take(3).cloned().collect();
    let mut digests = BTreeSet::new();
    for arm in Arm::ALL {
        if !matches!(arm, Arm::Retrieval(_)) {
            continue;
        }
        let mut run = run();
        run.per_cell = 2;
        // The items are only asked, so the three first stand for the examination.
        run_arm(&cache, &persona, arm, &provider, &few, &run).unwrap();
        let record = recorded(&exam_dir, arm.name()).unwrap().unwrap();
        digests.insert(record.digest().to_string());
        let Arm::Retrieval(setting) = arm else {
            unreachable!()
        };
        if setting == Retrieval::ORIGINAL {
            let text = std::fs::read_to_string(exam_dir.join("manifest.json")).unwrap();
            let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert!(
                doc["arms"]["retrieval"].get("retrieval").is_none(),
                "the original arm's record is as it was before the setting was recorded"
            );
            assert_eq!(record.retrieval, None);
        } else {
            let recorded_setting = record.retrieval.unwrap();
            assert_eq!(recorded_setting.over, setting.over);
            assert_eq!(recorded_setting.paragraphs, setting.passages.count());
        }
    }
    assert_eq!(digests.len(), 6, "six retrieval arms, six records");
}

#[test]
fn outcomes_over_records_are_never_appended_to_after_the_records_changed() {
    let (cache, persona, items) = fixture("changed");
    let provider: Arc<dyn ModelProvider> = Arc::new(FirstOption);
    let few: Vec<Item> = items.into_iter().take(3).collect();
    let arm = Arm::Retrieval(Retrieval {
        over: Over::Records,
        passages: Passages::Two,
    });
    run_arm(&cache, &persona, arm, &provider, &few, &run()).unwrap();
    jsonl::Log::open(&cache.dir().join("decisions.jsonl"))
        .unwrap()
        .append(&Mined {
            article: 1,
            case: "case later".into(),
            result: MinedResult::Kept(decision("case later", "Another matter.", all_facts())),
        })
        .unwrap();
    let message = format!(
        "{:#}",
        run_arm(&cache, &persona, arm, &provider, &few, &run()).unwrap_err()
    );
    assert!(message.contains("context"), "{message}");
}
