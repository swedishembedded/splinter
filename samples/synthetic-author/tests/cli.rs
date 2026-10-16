// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in reproducible command-line
// experiments, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the commands that show an author draw one when none is named, print
//! the seed, and reproduce the same output when it is passed back. Given a
//! corpus they take the author the corpus records and refuse another.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::{Command, Output};

use synthetic_author::manifest::{
    begin_or_continue, CorpusManifest, Limits, PersonaSource, Recipe, SeedSource,
};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_synthetic-author"))
        .args(args)
        .output()
        .unwrap()
}

/// The seed a command announced on stderr when it drew one.
fn announced_seed(out: &Output) -> u64 {
    let stderr = String::from_utf8_lossy(&out.stderr);
    let after = stderr
        .split("using seed ")
        .nth(1)
        .unwrap_or_else(|| panic!("no drawn seed announced on stderr: {stderr}"));
    after
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap_or_else(|_| panic!("not a seed: {stderr}"))
}

fn corpus_of(author_seed: u64) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "synthetic-author-cli-{author_seed}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let manifest = CorpusManifest {
        seed_source: SeedSource::Drawn,
        recipe: Recipe {
            author_seed,
            persona_source: PersonaSource::Drawn,
            writer: "writer".into(),
            reader: None,
            decoding: "greedy".into(),
            run_seed: None,
            articles: 10,
            episodes_per_article: 3,
            min_words: 700,
        },
        seed_derivation: String::new(),
        limits: Limits {
            attempts: 1,
            max_calls: 1,
            max_output_tokens: 1,
            workers: 1,
        },
        code_revision: "test".into(),
    };
    begin_or_continue(&dir, manifest).unwrap();
    dir
}

#[test]
fn persona_without_a_seed_draws_prints_and_reproduces_it() {
    let first = run(&["persona", "--show", "2"]);
    assert!(first.status.success(), "{first:?}");
    let seed = announced_seed(&first);
    let second = run(&["persona", "--show", "2"]);
    assert!(second.status.success(), "{second:?}");
    // Two draws of 64 bits colliding would be a defect of the draw itself.
    assert_ne!(announced_seed(&second), seed);

    let again = run(&["persona", "--show", "2", "--seed", &seed.to_string()]);
    assert!(again.status.success(), "{again:?}");
    assert_eq!(again.stdout, first.stdout);
    assert!(again.stderr.is_empty(), "a given seed is not announced");
}

#[test]
fn plan_show_and_items_draw_an_author_when_none_is_named() {
    for args in [
        &["plan", "--articles", "2"][..],
        &["show", "--articles", "2"][..],
        &[
            "items",
            "--articles",
            "2",
            "--per-cell",
            "2",
            "--exam",
            "v1",
        ][..],
    ] {
        let out = run(args);
        assert!(out.status.success(), "{args:?}: {out:?}");
        let seed = announced_seed(&out).to_string();
        let mut again: Vec<&str> = args.to_vec();
        again.extend(["--seed", &seed]);
        assert_eq!(run(&again).stdout, out.stdout, "{args:?}");
    }
}

#[test]
fn a_corpus_names_the_author_and_a_different_seed_is_refused() {
    let seed = splinter_sdk::model::seed::draw().value();
    let dir = corpus_of(seed);
    let cache = dir.to_str().unwrap();
    let recorded = run(&["persona", "--cache", cache, "--show", "1"]);
    assert!(recorded.status.success(), "{recorded:?}");
    let named = run(&["persona", "--seed", &seed.to_string(), "--show", "1"]);
    assert_eq!(recorded.stdout, named.stdout);

    for command in ["persona", "plan", "show"] {
        let other = run(&[command, "--cache", cache, "--seed", &(seed ^ 1).to_string()]);
        assert!(!other.status.success(), "{command}: {other:?}");
        let stderr = String::from_utf8_lossy(&other.stderr);
        assert!(stderr.contains(&seed.to_string()), "{command}: {stderr}");
    }
}

#[test]
fn survivors_reports_a_corpus_that_turned_nothing_away() {
    let seed = splinter_sdk::model::seed::draw().value();
    let dir = corpus_of(seed);
    let out = run(&[
        "survivors",
        "--cache",
        dir.to_str().unwrap(),
        "--articles",
        "2",
    ]);
    assert!(out.status.success(), "{out:?}");
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["articles_planned"], 2);
    assert_eq!(report["given_up"]["unattempted"], serde_json::json!([0, 1]));
}

#[test]
fn greedy_decoding_refuses_the_flags_of_sampling() {
    let seed = splinter_sdk::model::seed::draw().value();
    let dir = corpus_of(seed);
    let out = run(&[
        "exam",
        "--cache",
        dir.to_str().unwrap(),
        "--base",
        "none",
        "--arm",
        "base",
        "--draws",
        "3",
    ]);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--draws"),
        "{out:?}"
    );
}
