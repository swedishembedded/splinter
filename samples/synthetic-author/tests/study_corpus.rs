// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in pre-registering a
// replication study so that no choice can follow its results, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the corpus stage of the study writes an author's corpus with the
//! protocol's three models, reads the decisions out of it, makes the records,
//! and freezes and hashes the examination before anything is trained. It
//! resumes what exists, asks no model twice for the same work, and refuses a
//! protocol that has changed or a corpus that was changed after the freeze.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod study_support;

use std::sync::atomic::Ordering;
use std::time::Duration;

use study_support::{frozen, responding, small_protocol};
use synthetic_author::binding::file_digest;
use synthetic_author::dataset::Briefings;
use synthetic_author::study::corpus::{run, Pace, Staff};
use synthetic_author::study::layout::{AUTHOR_MANIFEST, EXAM_ITEMS};

const PACE: Pace = Pace {
    transport_retries: 1,
    transport_backoff: Duration::ZERO,
};

#[test]
fn the_stage_writes_reads_and_freezes_the_examination_before_anything_is_trained() {
    let study = frozen("corpus", small_protocol(|_| {}), 1);
    let protocol = study.protocol();
    let seed = protocol.authors[0].seed;
    let (model, requests) = responding(seed, protocol);
    let staff = Staff {
        writer: &model,
        reader: &model,
        miner: &model,
    };

    let manifest = run(&study, seed, &staff, &PACE).unwrap();
    assert!(requests.load(Ordering::SeqCst) > 0);

    assert_eq!(manifest.protocol_digest, study.digest().as_str());
    assert_eq!(manifest.models, protocol.models);
    assert_eq!(manifest.corpus.articles_planned, protocol.corpus.articles);
    assert!(manifest.corpus.articles_kept > 0, "{manifest:?}");
    assert!(manifest.corpus.decisions_kept > 0, "{manifest:?}");

    // The records of every setting the protocol trains on exist and are hashed.
    let settings: Vec<Briefings> = manifest.datasets.iter().map(|d| d.briefings).collect();
    assert_eq!(
        settings,
        [
            Briefings::Both,
            Briefings::ReaderOnly,
            Briefings::FixedWordingOnly
        ]
    );

    // The examination is as large as the protocol says and its file is hashed.
    assert_eq!(manifest.exam.items, protocol.exam.per_cell * 4);
    assert_eq!(manifest.exam.per_cell, protocol.exam.per_cell);
    let dir = study.author_dir(seed);
    assert_eq!(
        manifest.exam.file_digest,
        file_digest(&dir.join(EXAM_ITEMS)).unwrap()
    );
    assert_eq!(
        std::fs::read_to_string(dir.join(EXAM_ITEMS))
            .unwrap()
            .lines()
            .count(),
        manifest.exam.items
    );
    // Nothing is trained or examined by this stage.
    assert!(!dir.join("runs").exists());
    assert!(!dir.join("exam").exists());
    // The survivors are accounted for in the manifest, with the kept share.
    assert_eq!(
        manifest.survivors.articles_planned,
        protocol.corpus.articles
    );

    let read = study.author_manifest(seed).unwrap().unwrap();
    assert_eq!(read, manifest);
}

#[test]
fn a_repeat_changes_nothing_and_asks_no_model_again() {
    let study = frozen("corpus-repeat", small_protocol(|_| {}), 1);
    let seed = study.protocol().authors[0].seed;
    let (model, requests) = responding(seed, study.protocol());
    let staff = Staff {
        writer: &model,
        reader: &model,
        miner: &model,
    };
    let first = run(&study, seed, &staff, &PACE).unwrap();
    let asked = requests.load(Ordering::SeqCst);
    let second = run(&study, seed, &staff, &PACE).unwrap();
    assert_eq!(second, first);
    assert_eq!(
        requests.load(Ordering::SeqCst),
        asked,
        "nothing is asked twice"
    );
}

#[test]
fn a_run_that_spends_its_budget_stops_without_freezing_and_the_next_resumes() {
    let study = frozen(
        "corpus-resume",
        small_protocol(|p| p.corpus.max_calls = 7),
        1,
    );
    let seed = study.protocol().authors[0].seed;
    let (model, _) = responding(seed, study.protocol());
    let staff = Staff {
        writer: &model,
        reader: &model,
        miner: &model,
    };
    let mut runs = 0;
    let manifest = loop {
        runs += 1;
        assert!(runs < 10, "the corpus never finished");
        match run(&study, seed, &staff, &PACE) {
            Ok(manifest) => break manifest,
            Err(e) => {
                let message = format!("{e:#}");
                assert!(
                    message.contains("not finished") && message.contains("study corpus"),
                    "{message}"
                );
                assert!(
                    !study.author_dir(seed).join(AUTHOR_MANIFEST).exists(),
                    "an unfinished corpus is not frozen"
                );
            }
        }
    };
    assert!(runs > 1, "the budget was meant to stop the first run");
    assert_eq!(
        manifest.corpus.articles_planned,
        study.protocol().corpus.articles
    );
}

#[test]
fn work_done_under_another_protocol_is_refused() {
    let study = frozen("corpus-protocol", small_protocol(|_| {}), 1);
    let seed = study.protocol().authors[0].seed;
    let (model, _) = responding(seed, study.protocol());
    let staff = Staff {
        writer: &model,
        reader: &model,
        miner: &model,
    };
    run(&study, seed, &staff, &PACE).unwrap();
    let path = study.author_dir(seed).join(AUTHOR_MANIFEST);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, text.replace(study.digest().as_str(), "blake3:other")).unwrap();
    let message = format!("{:#}", run(&study, seed, &staff, &PACE).unwrap_err());
    assert!(
        message.contains("another protocol") && message.contains("blake3:other"),
        "{message}"
    );
}

#[test]
fn a_corpus_changed_after_the_examination_was_frozen_is_refused() {
    let study = frozen("corpus-changed", small_protocol(|_| {}), 1);
    let seed = study.protocol().authors[0].seed;
    let (model, _) = responding(seed, study.protocol());
    let staff = Staff {
        writer: &model,
        reader: &model,
        miner: &model,
    };
    let first = run(&study, seed, &staff, &PACE).unwrap();
    // The examination frozen for the author was another one than the corpus
    // now makes: as if the corpus had been mined again after the freeze.
    let path = study.author_dir(seed).join(AUTHOR_MANIFEST);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        &path,
        text.replace(&first.exam.digest, "blake3:frozen-before"),
    )
    .unwrap();
    let message = format!("{:#}", run(&study, seed, &staff, &PACE).unwrap_err());
    assert!(
        message.contains("changed after the examination was frozen")
            && message.contains(&first.exam.digest)
            && message.contains("blake3:frozen-before"),
        "{message}"
    );
}

#[test]
fn the_command_names_a_model_that_runs_here_instead_of_starting_a_run() {
    let study = frozen(
        "corpus-local",
        small_protocol(|p| p.models.writer = "local:/models/aster-8b".into()),
        1,
    );
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_synthetic-author"))
        .args(["study", "corpus", "--study"])
        .arg(study.root())
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("writer") && stderr.contains("runs here"),
        "{stderr}"
    );
    assert!(!study.author_dir(study.protocol().authors[0].seed).exists());
}
