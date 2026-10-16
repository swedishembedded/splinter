// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in telling whether a model took on
// one author's judgment rather than any author's, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the author who holds every belief the other way can be written a
//! corpus of its own. The manifest says whose opposite it is, it must pass the
//! checkability rule of any author or be refused with the reason, every later
//! command reads it back as that author, and `cross` sets each trained
//! opposite beside the author it opposes.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use synthetic_author::binding::ModelIdentity;
use synthetic_author::cache::Cache;
use synthetic_author::exam::Version;
use synthetic_author::lab::{exam_items, run_arm, Arm, ExamRun};
use synthetic_author::manifest::{
    author, begin_or_continue, corpus_author, CorpusManifest, Limits, PersonaSource, Recipe,
    SeedSource, SEED_DERIVATION,
};
use synthetic_author::persona::Persona;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("synthetic-author-opposite-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn manifest(author_seed: u64, source: PersonaSource) -> CorpusManifest {
    CorpusManifest {
        seed_source: SeedSource::Given,
        recipe: Recipe {
            author_seed,
            persona_source: source,
            writer: "writer".into(),
            reader: None,
            allow_same_family: false,
            decoding: "greedy".into(),
            run_seed: None,
            articles: 10,
            episodes_per_article: 3,
            min_words: 700,
        },
        seed_derivation: SEED_DERIVATION.into(),
        limits: Limits {
            attempts: 1,
            max_calls: 1,
            max_output_tokens: 1,
            workers: 1,
        },
        code_revision: "test".into(),
    }
}

fn drawn_seed() -> u64 {
    splinter_sdk::model::seed::draw().value()
}

/// A seed whose opposite passes the rule; a draw does not always (about one in
/// eight fails), and the refusal has its own test.
fn opposable_seed() -> u64 {
    loop {
        let seed = drawn_seed();
        if Persona::opposite_of(seed).is_ok() {
            return seed;
        }
    }
}

#[test]
fn an_opposite_that_fails_the_rule_is_refused_naming_why() {
    let seed = loop {
        let seed = drawn_seed();
        if Persona::opposite_of(seed).is_err() {
            break seed;
        }
    };
    let dir = scratch("refused");
    let message = format!("{:#}", corpus_author(&dir, None, Some(seed)).unwrap_err());
    assert!(message.contains("cannot be checked against"), "{message}");
    assert!(
        message.contains("changes no decision") || message.contains("reference situations"),
        "{message}"
    );
    assert!(!dir.join("manifest.json").exists());
}

#[test]
fn the_opposite_holds_every_belief_the_other_way_and_passes_the_same_rule() {
    let seed = opposable_seed();
    let source = Persona::generate(seed);
    let opposite = Persona::opposite_of(seed).unwrap();
    assert_eq!(opposite, source.opposite());
    assert_eq!(opposite.beliefs.len(), source.beliefs.len());
    for (a, b) in source.beliefs.iter().zip(&opposite.beliefs) {
        assert_eq!(a.tenet, b.tenet);
        assert_ne!(a.polarity, b.polarity);
    }
    assert_ne!(opposite.name, source.name);
}

#[test]
fn an_author_who_fails_the_rule_is_refused_and_the_belief_that_decides_nothing_is_named() {
    let persona = Persona::generate(drawn_seed());
    // Over no situation at all, no belief changes any decision.
    let refusal = persona.checkability(&[]).unwrap_err().to_string();
    assert!(
        refusal.contains(persona.beliefs[0].statement()),
        "{refusal}"
    );
}

#[test]
fn a_corpus_of_the_opposite_records_whose_opposite_it_is_and_is_read_back_as_that_author() {
    let seed = opposable_seed();
    let dir = scratch("record");
    let begun = corpus_author(&dir, None, Some(seed)).unwrap();
    assert_eq!(begun.persona, Persona::opposite_of(seed).unwrap());
    assert_eq!(
        begun.seed, seed,
        "the opposite is set in its source's world"
    );
    assert_eq!(begun.persona_source, PersonaSource::OppositeOf { seed });
    begin_or_continue(&dir, manifest(begun.seed, begun.persona_source)).unwrap();

    let read = author(&dir, None).unwrap();
    assert_eq!(read.persona, begun.persona);
    assert_eq!(read.seed, seed);
    assert_eq!(
        corpus_author(&dir, None, Some(seed)).unwrap().persona,
        begun.persona,
        "the same request continues the corpus"
    );
    assert_eq!(
        corpus_author(&dir, None, None).unwrap().persona,
        begun.persona,
        "a corpus is continued as the author it was begun for"
    );
    let manifest_text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert!(manifest_text.contains("opposite-of"), "{manifest_text}");
}

#[test]
fn a_corpus_is_never_continued_as_another_kind_of_author() {
    let seed = opposable_seed();
    let drawn = scratch("drawn");
    begin_or_continue(&drawn, manifest(seed, PersonaSource::Drawn)).unwrap();
    let message = format!("{:#}", corpus_author(&drawn, None, Some(seed)).unwrap_err());
    assert!(message.contains("persona source"), "{message}");

    let opposite = scratch("other-source");
    begin_or_continue(
        &opposite,
        manifest(seed, PersonaSource::OppositeOf { seed }),
    )
    .unwrap();
    let message = format!(
        "{:#}",
        corpus_author(&opposite, None, Some(seed ^ 1)).unwrap_err()
    );
    assert!(message.contains(&seed.to_string()), "{message}");
}

#[test]
fn a_command_given_the_opposite_s_corpus_shows_the_opposite() {
    let seed = opposable_seed();
    let dir = scratch("cli");
    begin_or_continue(&dir, manifest(seed, PersonaSource::OppositeOf { seed })).unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_synthetic-author"))
        .args(["persona", "--show", "1", "--cache"])
        .arg(&dir)
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    let opposite = Persona::opposite_of(seed).unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.starts_with(&opposite.name), "{stdout}");
    assert!(stdout.contains(opposite.beliefs[0].statement()), "{stdout}");
}

#[test]
fn the_corpus_command_takes_either_a_seed_or_an_author_to_oppose_and_not_both() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_synthetic-author"))
        .args([
            "corpus",
            "--seed",
            "1",
            "--opposite-of",
            "2",
            "--cache",
            "none",
            "--writer",
            "w",
            "--max-calls",
            "1",
            "--max-output-tokens",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--opposite-of"),
        "{out:?}"
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

const ARTICLES: usize = 20;
const PER_CELL: usize = 2;

/// Examines `persona` in `dir` as the tuned arm, which takes the first option.
fn examined(dir: &std::path::Path, persona: &Persona, seed: u64) {
    let cache = Cache::open(dir.to_path_buf()).unwrap();
    let run = ExamRun {
        articles: ARTICLES,
        per_cell: PER_CELL,
        max_tokens: 64,
        model: ModelIdentity {
            base_digest: "sha256:base".into(),
            adapter_digest: Some("sha256:adapter".into()),
            decoding: "greedy".into(),
            decoding_seed: None,
        },
        exam: Version::V1,
        adopt: false,
    };
    let provider: Arc<dyn ModelProvider> = Arc::new(FirstOption);
    run_arm(
        &cache,
        persona,
        Arm::Tuned,
        &provider,
        &exam_items(persona, seed, ARTICLES, PER_CELL),
        &run,
    )
    .unwrap();
}

#[test]
fn cross_sets_the_trained_opposite_beside_the_author_it_opposes() {
    let seed = opposable_seed();
    let exams = scratch("cross");
    let source = Persona::generate(seed);
    let opposite = Persona::opposite_of(seed).unwrap();

    let source_dir = exams.join(format!("author-{seed}"));
    begin_or_continue(&source_dir, manifest(seed, PersonaSource::Drawn)).unwrap();
    examined(&source_dir, &source, seed);
    let opposite_dir = exams.join(format!("opposite-of-{seed}"));
    begin_or_continue(
        &opposite_dir,
        manifest(seed, PersonaSource::OppositeOf { seed }),
    )
    .unwrap();
    examined(&opposite_dir, &opposite, seed);

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_synthetic-author"))
        .args(["cross", "--exams"])
        .arg(&exams)
        .args(["--seeds", &seed.to_string()])
        .args(["--articles", &ARTICLES.to_string()])
        .args(["--per-cell", &PER_CELL.to_string()])
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(&format!("| {} | {} |", source.name, opposite.name)),
        "the author against the opposite oracle: {stdout}"
    );
    assert!(
        stdout.contains(&format!("| {} | {} |", opposite.name, source.name)),
        "the trained opposite against the author: {stdout}"
    );
}
