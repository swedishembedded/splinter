// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving which author, model and
// settings a result belongs to, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: what a run leaves behind says what made it, and a later run that
//! would contradict it is refused. A corpus names its author, and a command
//! given another author's seed is refused. An examination's outcomes are
//! bound to the items and the model they were made for, and are never
//! appended to under another manifest. Training records every setting that
//! made an adapter.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use serde_json::json;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use synthetic_author::arms::read_outcomes;
use synthetic_author::binding::{
    file_digest, recorded, verify_items, verify_outcomes, ModelIdentity,
};
use synthetic_author::cache::Cache;
use synthetic_author::exam::{draw, Item, Spec, Trained, Version};
use synthetic_author::lab::{
    arm_record, asked_items, exam_items, exam_items_of, outcomes_path, run_arm, Arm, ExamRun,
    Mined, MinedResult,
};
use synthetic_author::manifest::{
    author_seed, begin_or_continue, corpus_author_seed, CorpusManifest, Limits, PersonaSource,
    Recipe, SeedSource, SEED_DERIVATION,
};
use synthetic_author::persona::Persona;
use synthetic_author::training::TrainArgs;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "synthetic-author-manifest-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn recipe(author_seed: u64, writer: &str) -> CorpusManifest {
    CorpusManifest {
        seed_source: SeedSource::Given,
        recipe: Recipe {
            author_seed,
            persona_source: PersonaSource::Drawn,
            writer: writer.into(),
            reader: Some("reader-model".into()),
            allow_same_family: false,
            decoding: "sampled".into(),
            run_seed: Some(!author_seed),
            articles: 10,
            episodes_per_article: 3,
            min_words: 700,
        },
        seed_derivation: SEED_DERIVATION.into(),
        limits: Limits {
            attempts: 3,
            max_calls: 50,
            max_output_tokens: 10_000,
            workers: 1,
        },
        code_revision: "test".into(),
    }
}

#[test]
fn a_corpus_records_how_it_was_made() {
    let dir = scratch("corpus-record");
    begin_or_continue(&dir, recipe(7, "writer-model")).unwrap();
    let text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    for key in [
        "author_seed",
        "writer",
        "reader",
        "decoding",
        "run_seed",
        "seed_derivation",
        "articles",
        "limits",
        "code_revision",
    ] {
        assert!(!doc[key].is_null(), "{key} is recorded: {text}");
    }
}

#[test]
fn a_corpus_without_a_seed_draws_an_author_and_every_later_command_reads_it_back() {
    let dir = scratch("corpus-draw");
    let (first, source) = corpus_author_seed(&dir, None).unwrap();
    assert_eq!(source, SeedSource::Drawn);
    let (other, _) = corpus_author_seed(&dir, None).unwrap();
    assert_ne!(first, other, "nothing is recorded yet, so each call draws");
    let mut manifest = recipe(first, "writer-model");
    manifest.seed_source = source;
    begin_or_continue(&dir, manifest).unwrap();
    assert_eq!(author_seed(&dir, None).unwrap(), first);
    assert_eq!(author_seed(&dir, Some(first)).unwrap(), first);
    assert_eq!(corpus_author_seed(&dir, None).unwrap().0, first);
}

#[test]
fn a_seed_that_is_not_the_corpus_s_author_is_refused() {
    let dir = scratch("author-refused");
    begin_or_continue(&dir, recipe(7, "writer-model")).unwrap();
    for refused in [
        author_seed(&dir, Some(8)),
        corpus_author_seed(&dir, Some(8)).map(|(s, _)| s),
    ] {
        let message = format!("{:#}", refused.unwrap_err());
        assert!(message.contains('7') && message.contains('8'), "{message}");
    }
}

#[test]
fn a_corpus_from_before_manifests_is_known_by_its_persona_file() {
    let dir = scratch("author-legacy");
    Cache::open(dir.clone())
        .unwrap()
        .write_persona(&Persona::generate(5))
        .unwrap();
    assert_eq!(author_seed(&dir, None).unwrap(), 5);
    assert!(author_seed(&dir, Some(6)).is_err());
}

#[test]
fn a_directory_that_records_no_author_needs_a_seed() {
    assert!(author_seed(&scratch("author-none"), None).is_err());
}

#[test]
fn a_corpus_is_not_continued_with_another_writer() {
    let dir = scratch("corpus-writer");
    begin_or_continue(&dir, recipe(7, "writer-model")).unwrap();
    begin_or_continue(&dir, recipe(7, "writer-model")).unwrap();
    let message = format!(
        "{:#}",
        begin_or_continue(&dir, recipe(7, "another-model")).unwrap_err()
    );
    assert!(message.contains("writer"), "{message}");
}

/// A model that always takes the first option of the tool it is offered.
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

const SEED: u64 = 6; // seed: the exam items are those of one fixed author
const PER_CELL: usize = 2;

struct Study {
    cache: Cache,
    persona: Persona,
}

fn study(tag: &str) -> Study {
    Study {
        cache: Cache::open(scratch(tag)).unwrap(),
        persona: Persona::generate(SEED),
    }
}

fn model(adapter: Option<&str>) -> ModelIdentity {
    ModelIdentity {
        base_digest: "sha256:base".into(),
        adapter_digest: adapter.map(str::to_string),
        decoding: "greedy".into(),
        decoding_seed: None,
    }
}

impl Study {
    fn items(&self, articles: usize, per_cell: usize) -> Vec<Item> {
        exam_items(&self.persona, SEED, articles, per_cell)
    }

    fn exam(
        &self,
        articles: usize,
        per_cell: usize,
        adapter: Option<&str>,
        adopt: bool,
    ) -> anyhow::Result<()> {
        let provider: Arc<dyn ModelProvider> = Arc::new(FirstOption);
        let run = ExamRun {
            articles,
            per_cell,
            max_tokens: 64,
            model: model(adapter),
            exam: Version::V1,
            adopt,
        };
        run_arm(
            &self.cache,
            &self.persona,
            Arm::Tuned,
            &provider,
            &self.items(articles, per_cell),
            &run,
        )
        .map(|_| ())
    }

    fn manifest_digest(&self) -> String {
        let dir = self.cache.dir().join("exam");
        recorded(&dir, "tuned")
            .unwrap()
            .unwrap()
            .digest()
            .to_string()
    }
}

#[test]
fn an_exam_rerun_for_other_items_is_refused_naming_both_digests() {
    let s = study("exam-items");
    s.exam(20, PER_CELL, Some("sha256:a"), false).unwrap();
    s.exam(20, PER_CELL, Some("sha256:a"), false)
        .expect("the same manifest picks up where it stopped");
    let kept = s.manifest_digest();
    let rerun = ExamRun {
        articles: 21,
        per_cell: PER_CELL,
        max_tokens: 64,
        model: model(Some("sha256:a")),
        exam: Version::V1,
        adopt: false,
    };
    let other = arm_record(
        &s.cache,
        &s.persona,
        Arm::Tuned,
        &s.items(21, PER_CELL),
        &rerun,
    )
    .unwrap()
    .digest()
    .to_string();
    let message = format!(
        "{:#}",
        s.exam(21, PER_CELL, Some("sha256:a"), false).unwrap_err()
    );
    assert!(
        message.contains(&kept) && message.contains(&other),
        "{message}"
    );
    assert_eq!(s.manifest_digest(), kept, "the manifest is not rewritten");
}

#[test]
fn the_second_exam_is_a_new_exam_and_the_first_one_s_outcomes_are_refused_under_it() {
    let s = study("exam-v2");
    s.exam(20, PER_CELL, Some("sha256:a"), false).unwrap();
    let kept = s.manifest_digest();
    let told = synthetic_author::writer::plan(&s.persona, SEED, 20, 3)
        .into_iter()
        .flat_map(|(_, eps)| eps.into_iter().map(|e| e.case.name))
        .collect();
    let second = draw(
        &s.persona,
        SEED,
        &Spec {
            version: Version::V2,
            per_cell: PER_CELL,
            corpus_cases: &told,
            trained: &Trained::default(),
        },
    );
    let provider: Arc<dyn ModelProvider> = Arc::new(FirstOption);
    let run = ExamRun {
        articles: 20,
        per_cell: PER_CELL,
        max_tokens: 64,
        model: model(Some("sha256:a")),
        exam: Version::V2,
        adopt: false,
    };
    let message = format!(
        "{:#}",
        run_arm(&s.cache, &s.persona, Arm::Tuned, &provider, &second, &run).unwrap_err()
    );
    assert!(
        message.contains(&kept) && message.contains("exam") && message.contains("items"),
        "{message}"
    );
}

#[test]
fn an_exam_rerun_with_another_adapter_is_refused_naming_both_digests() {
    let s = study("exam-adapter");
    s.exam(20, PER_CELL, Some("sha256:a"), false).unwrap();
    let kept = s.manifest_digest();
    let message = format!(
        "{:#}",
        s.exam(20, PER_CELL, Some("sha256:b"), false).unwrap_err()
    );
    assert!(
        message.contains(&kept) && message.contains("adapter"),
        "{message}"
    );
}

#[test]
fn an_exam_rerun_with_an_adapter_retrained_at_the_same_path_is_refused() {
    let s = study("exam-retrained");
    let adapter = s.cache.dir().join("adapter.safetensors");
    std::fs::write(&adapter, b"first training").unwrap();
    let digest = |path: &std::path::Path| file_digest(path).unwrap();
    s.exam(20, PER_CELL, Some(&digest(&adapter)), false)
        .unwrap();
    std::fs::write(&adapter, b"second training").unwrap();
    let message = format!(
        "{:#}",
        s.exam(20, PER_CELL, Some(&digest(&adapter)), false)
            .unwrap_err()
    );
    assert!(message.contains("adapter"), "{message}");
}

#[test]
fn outcomes_that_predate_manifests_are_refused_until_adopted() {
    let s = study("exam-adopt");
    let path = outcomes_path(&s.cache, Arm::Tuned);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    // An outcome file as it was written before outcomes carried digests.
    let old = s.items(20, PER_CELL);
    let lines: Vec<String> = old
        .iter()
        .map(|i| {
            json!({"id": i.id, "correct": false, "topic_seen": i.topic_seen,
                   "combo_seen": i.combo_seen, "tool_seen": i.tool_seen, "call": null})
            .to_string()
        })
        .collect();
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();

    let message = format!("{:#}", s.exam(20, PER_CELL, None, false).unwrap_err());
    assert!(message.contains("--adopt"), "{message}");
    assert!(
        !s.cache.dir().join("exam").join("manifest.json").exists(),
        "refusing writes nothing"
    );

    s.exam(20, PER_CELL, None, true).unwrap();
    let record = recorded(&s.cache.dir().join("exam"), "tuned")
        .unwrap()
        .unwrap();
    assert!(record.adopted);
    let outcomes = read_outcomes(&path).unwrap();
    verify_outcomes(&record, &outcomes, &path).expect("adopted outcomes need no digests");
    verify_items(&record, &old, &outcomes, &path).expect("nor do they for the items");
}

#[test]
fn outcomes_made_by_another_model_or_for_other_items_do_not_join() {
    let s = study("exam-join");
    s.exam(20, PER_CELL, Some("sha256:a"), false).unwrap();
    let path = outcomes_path(&s.cache, Arm::Tuned);
    let record = recorded(&s.cache.dir().join("exam"), "tuned")
        .unwrap()
        .unwrap();
    let outcomes = read_outcomes(&path).unwrap();
    verify_outcomes(&record, &outcomes, &path).unwrap();
    verify_items(&record, &s.items(20, PER_CELL), &outcomes, &path).unwrap();

    let mut foreign = outcomes.clone();
    foreign[0].model = "blake3:other".into();
    assert!(verify_outcomes(&record, &foreign, &path).is_err());

    let message = format!(
        "{:#}",
        verify_items(&record, &s.items(20, PER_CELL + 1), &outcomes, &path).unwrap_err()
    );
    assert!(message.contains(&record.items_digest), "{message}");
}

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    train: TrainArgs,
}

#[test]
fn a_training_run_records_the_seed_and_every_setting_that_made_the_adapter() {
    let dir = scratch("train");
    let records = dir.join("train.jsonl");
    std::fs::write(&records, "{\"a\":1}\n").unwrap();
    let cli = Cli::parse_from([
        "x",
        "--cache",
        "c",
        "--base",
        "b",
        "--steps",
        "50",
        "--rank",
        "8",
        "--alpha",
        "16",
        "--lr",
        "0.0002",
        "--records-per-step",
        "2",
        "--bf16",
    ]);
    let settings = cli.train.settings(1234, &records).unwrap();
    let doc = serde_json::to_value(&settings).unwrap();
    assert_eq!(doc["seed"], 1234);
    assert_eq!(doc["steps"], 50);
    assert_eq!(doc["rank"], 8);
    assert_eq!(doc["alpha"], 16.0);
    assert!((doc["learning_rate"].as_f64().unwrap() - 0.0002).abs() < 1e-9);
    assert_eq!(doc["records_per_step"], 2);
    assert_eq!(doc["bf16"], true);
    let digest = doc["dataset_digest"].as_str().unwrap();
    assert!(digest.starts_with("blake3:"), "{digest}");
    std::fs::write(&records, "{\"a\":2}\n").unwrap();
    let changed = cli.train.settings(1234, &records).unwrap();
    assert_ne!(changed.dataset_digest, settings.dataset_digest);
}

#[test]
fn the_baselines_are_scored_only_on_the_items_the_arms_were_asked() {
    let s = study("baselines");
    s.cache.write_persona(&s.persona).unwrap();
    let decisions = s.cache.dir().join("decisions.jsonl");
    let line = |result| {
        let read = Mined {
            article: 0,
            case: "Any".into(),
            result,
        };
        format!("{}\n", serde_json::to_string(&read).unwrap())
    };
    // Read, and nothing kept: no item is trained on yet.
    std::fs::write(&decisions, line(MinedResult::Refused("unclear".into()))).unwrap();
    let items = exam_items_of(&s.cache, &s.persona, SEED, 20, PER_CELL, Version::V2).unwrap();
    let run = ExamRun {
        articles: 20,
        per_cell: PER_CELL,
        max_tokens: 64,
        model: model(Some("sha256:a")),
        exam: Version::V2,
        adopt: false,
    };
    let provider: Arc<dyn ModelProvider> = Arc::new(FirstOption);
    run_arm(&s.cache, &s.persona, Arm::Tuned, &provider, &items, &run).unwrap();
    let record = recorded(&s.cache.dir().join("exam"), "tuned")
        .unwrap()
        .unwrap();
    assert_eq!(asked_items(&s.cache, &record).unwrap().len(), items.len());

    // The corpus is read again and now holds a decision on the first item.
    let target = &items[0];
    let f = target.scenario.features;
    let kept = synthetic_author::mine::Verified {
        case: "Any".into(),
        briefing: "x".into(),
        offer_made: f.outsider_offers_help,
        facts: synthetic_author::mine::MinedFacts {
            exceeds_authority: Some(f.exceeds_authority),
            urgent: Some(f.urgent),
            binds_future: Some(f.binds_future),
            irreversible: Some(f.irreversible),
            members_harmed: Some(f.members_harmed),
        },
        chose: target.expected,
        open: target.scenario.options.clone(),
        reasons: vec![],
        dropped: 0,
    };
    let mut text = std::fs::read_to_string(&decisions).unwrap();
    text.push_str(&line(MinedResult::Kept(kept)));
    std::fs::write(&decisions, text).unwrap();
    let message = format!("{:#}", asked_items(&s.cache, &record).unwrap_err());
    assert!(
        message.contains("not scored") && message.contains(&record.items_digest),
        "{message}"
    );
}
