// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in keeping the model that writes a
// training corpus from also being the one that vouches for it, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the model that writes the essays, the one that reads them for the
//! gate and the one that reads the decisions out of them are of different
//! families, or the run says it was allowed to be otherwise and the manifest
//! records that it did.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::{Command, Output};

use synthetic_author::family::{check_distinct, family, Roles};
use synthetic_author::manifest::{
    begin_or_continue, begin_or_continue_mining, CorpusManifest, Limits, PersonaSource, Recipe,
    SeedSource, SEED_DERIVATION,
};

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "synthetic-author-protocol-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_family_is_the_letters_a_model_s_name_begins_with() {
    for (reference, expected) in [
        ("remote:north/aster-large-4", "aster"),
        ("remote:south/birch-4o", "birch"),
        ("remote:together/vendor-dune/Dune-3.1-70B", "dune"),
        ("local:/models/Cedar3-8B", "cedar"),
        (
            "local:/models/Cedar3-8B+/adapters/a.safetensors@4096",
            "cedar",
        ),
        ("policy:default", "policy:default"),
    ] {
        assert_eq!(family(reference).unwrap(), expected, "{reference}");
    }
    assert!(family("no scheme").is_err());
}

fn roles<'a>(writer: &'a str, reader: Option<&'a str>, miner: Option<&'a str>) -> Roles<'a> {
    Roles {
        writer,
        reader,
        miner,
    }
}

#[test]
fn roles_of_different_families_pass_and_two_of_one_family_are_refused_naming_them() {
    let (aster, birch, cedar) = (
        "remote:a/aster-xl-4",
        "remote:b/birch-5",
        "local:/models/Cedar3-8B",
    );
    check_distinct(&roles(aster, Some(birch), Some(cedar))).unwrap();
    check_distinct(&roles(aster, None, None)).unwrap();
    check_distinct(&roles(aster, None, Some(birch))).unwrap();

    let message = |r: Roles| format!("{:#}", check_distinct(&r).unwrap_err());
    let m = message(roles(aster, Some("remote:c/aster-small-4"), None));
    assert!(
        m.contains("writer") && m.contains("reader") && m.contains("aster"),
        "{m}"
    );
    let m = message(roles(aster, Some(birch), Some("local:/x/Aster-tuned")));
    assert!(m.contains("writer") && m.contains("miner"), "{m}");
    assert!(m.contains("--allow-same-family"), "{m}");
    let m = message(roles(aster, Some(birch), Some("remote:d/birch-4")));
    assert!(m.contains("reader") && m.contains("miner"), "{m}");
}

fn manifest(writer: &str, reader: Option<&str>, allow_same_family: bool) -> CorpusManifest {
    CorpusManifest {
        seed_source: SeedSource::Given,
        recipe: Recipe {
            author_seed: splinter_sdk::model::seed::draw().value(),
            persona_source: PersonaSource::Drawn,
            writer: writer.into(),
            reader: reader.map(str::to_string),
            allow_same_family,
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

#[test]
fn the_manifest_records_that_the_protocol_was_set_aside_and_a_corpus_is_continued_the_same_way() {
    let dir = scratch("manifest");
    let allowed = manifest("remote:a/aster-1", Some("remote:b/aster-2"), true);
    begin_or_continue(&dir, allowed.clone()).unwrap();
    let text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert!(text.contains("\"allow_same_family\": true"), "{text}");
    let message = format!(
        "{:#}",
        begin_or_continue(
            &dir,
            manifest("remote:a/aster-1", Some("remote:b/aster-2"), false)
        )
        .unwrap_err()
    );
    assert!(message.contains("allow same family"), "{message}");

    let plain = scratch("manifest-plain");
    begin_or_continue(
        &plain,
        manifest("remote:a/aster-1", Some("remote:b/birch-5"), false),
    )
    .unwrap();
    let text = std::fs::read_to_string(plain.join("manifest.json")).unwrap();
    assert!(
        !text.contains("allow_same_family"),
        "a corpus that kept to the protocol does not say so"
    );
}

#[test]
fn the_miner_is_held_to_the_families_of_the_writer_and_the_reader_the_corpus_records() {
    let dir = scratch("mine");
    begin_or_continue(
        &dir,
        manifest("remote:a/aster-1", Some("remote:b/birch-5"), false),
    )
    .unwrap();
    for same in ["remote:c/aster-9", "remote:c/birch-4"] {
        let message = format!(
            "{:#}",
            begin_or_continue_mining(&dir, same, None, false, false).unwrap_err()
        );
        assert!(message.contains("same model family"), "{same}: {message}");
        assert!(
            !dir.join("mine.json").exists(),
            "a refused run leaves no record"
        );
    }
    let record =
        begin_or_continue_mining(&dir, "local:/models/Cedar3-8B", Some(5), true, false).unwrap();
    assert!(!record.allow_same_family);

    let other = scratch("mine-allowed");
    begin_or_continue(&other, manifest("remote:a/aster-1", None, false)).unwrap();
    let record = begin_or_continue_mining(&other, "remote:c/aster-9", None, false, true).unwrap();
    assert!(record.allow_same_family);
    let text = std::fs::read_to_string(other.join("mine.json")).unwrap();
    assert!(text.contains("\"allow_same_family\": true"), "{text}");
    // Continuing the same miner needs no new permission: the record holds it.
    assert!(
        begin_or_continue_mining(&other, "remote:c/aster-9", None, false, false)
            .unwrap()
            .allow_same_family
    );
}

#[test]
fn a_corpus_that_predates_manifests_has_no_writer_to_compare_with() {
    let dir = scratch("legacy");
    let record = begin_or_continue_mining(&dir, "remote:c/aster-9", None, false, false).unwrap();
    assert!(!record.allow_same_family);
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_synthetic-author"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn the_corpus_command_refuses_a_reader_of_the_writer_s_family_before_it_starts_anything() {
    let dir = scratch("cli");
    let cache = dir.join("corpus");
    let args = |extra: &[&'static str]| -> Output {
        let mut argv = vec![
            "corpus",
            "--writer",
            "remote:a/aster-1",
            "--reader",
            "remote:b/aster-2",
            "--max-calls",
            "1",
            "--max-output-tokens",
            "1",
            "--cache",
            cache.to_str().unwrap(),
        ];
        argv.extend_from_slice(extra);
        run(&argv)
    };
    let refused = args(&[]);
    assert!(!refused.status.success());
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains("same model family") && stderr.contains("--allow-same-family"),
        "{stderr}"
    );
    assert!(!cache.join("manifest.json").exists());

    // With the flag the protocol no longer stops it; the network opt-in does.
    let allowed = args(&["--allow-same-family"]);
    let stderr = String::from_utf8_lossy(&allowed.stderr);
    assert!(!stderr.contains("same model family"), "{stderr}");
}
