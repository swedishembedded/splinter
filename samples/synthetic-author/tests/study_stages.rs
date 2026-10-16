// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in training runs whose
// every setting is recorded and repeatable, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: the training and examination stages of the study run from the
//! protocol and the frozen examination, one manifest per adapter, resuming and
//! refusing what contradicts a manifest. They are driven here by a scripted
//! trainer and scripted models: no device is used.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod study_support;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_sdk::agent::sven::model::ModelProvider;
use study_support::{frozen, responding, small_protocol, FirstOption, Scripted};
use synthetic_author::binding::{file_digest, recorded, Base};
use synthetic_author::study::amend::{amend_base_hold, BaseHold};
use synthetic_author::study::arm_key::SHARED_SLOT;
use synthetic_author::study::corpus::{self, Staff};
use synthetic_author::study::examine::{self, Examinee, Loader};
use synthetic_author::study::layout::Study;
use synthetic_author::study::train;
use synthetic_author::transit::Transit;

/// Loads a scripted model and remembers which adapter each load attached.
struct Scripting {
    loads: Mutex<Vec<Option<PathBuf>>>,
    requests: Arc<AtomicUsize>,
}

struct Answers(Arc<AtomicUsize>);

impl Examinee for Answers {
    fn greedy(&self) -> Arc<dyn ModelProvider> {
        Arc::new(FirstOption {
            requests: self.0.clone(),
        })
    }
    fn sampled(&self, _: f32, _: u64) -> Arc<dyn ModelProvider> {
        self.greedy()
    }
}

impl Loader for Scripting {
    fn load(&self, adapter: Option<&Path>) -> anyhow::Result<Box<dyn Examinee>> {
        self.loads
            .lock()
            .unwrap()
            .push(adapter.map(Path::to_path_buf));
        Ok(Box::new(Answers(self.requests.clone())))
    }
}

fn scripting() -> Scripting {
    Scripting {
        loads: Mutex::new(Vec::new()),
        requests: Arc::new(AtomicUsize::new(0)),
    }
}

/// A study of one author whose corpus stage has run, and a stand-in for the
/// base checkpoint.
fn corpus_done(tag: &str) -> (Study, u64, Base) {
    let study = frozen(tag, small_protocol(|_| {}), 1);
    let seed = study.protocol().authors[0].seed;
    let (model, _) = responding(seed, study.protocol());
    corpus::run(
        &study,
        seed,
        &Staff {
            writer: &model,
            reader: &model,
            miner: &model,
        },
        &Transit::new(1, Duration::ZERO),
        None,
    )
    .unwrap();
    let base = study.root().join("base.bin");
    std::fs::write(&base, "base weights").unwrap();
    (study, seed, Base::open(&base).unwrap())
}

#[test]
fn every_adapter_the_protocol_asks_is_trained_once_under_its_seed_with_a_manifest() {
    let (study, author, base) = corpus_done("stages-train");
    let trainer = Scripted::default();
    let done = train::run(&study, author, &base, &trainer).unwrap();

    // The full adapter at both training seeds, each ablation at the first.
    let plan = study.author_plan(author).unwrap();
    assert_eq!((done.trained, done.already), (4, 0));
    let mut seen = trainer.seeds.lock().unwrap().clone();
    seen.sort();
    let mut expected = vec![
        (plan.training_seeds[0], String::new()),
        (plan.training_seeds[1], String::new()),
        (plan.training_seeds[0], "-reader-only".to_string()),
        (plan.training_seeds[0], "-fixed-wording-only".to_string()),
    ];
    expected.sort();
    assert_eq!(seen, expected);

    let frozen = study.require_author_manifest(author).unwrap();
    let t = &study.protocol().training;
    for training_seed in &plan.training_seeds {
        let run = study
            .run_manifest(
                author,
                synthetic_author::dataset::Briefings::Both,
                *training_seed,
            )
            .unwrap()
            .unwrap();
        assert_eq!(run.protocol_digest, study.digest().as_str());
        assert_eq!(
            run.exam_digest, frozen.exam.digest,
            "trained against the frozen examination"
        );
        assert_eq!(run.settings.seed, *training_seed);
        assert_eq!((run.settings.steps, run.settings.rank), (t.steps, t.rank));
        assert_eq!(run.base_digest, file_digest(base.path()).unwrap());
        assert_eq!(
            run.adapter_digest,
            file_digest(&study.root().join(&run.adapter)).unwrap()
        );
    }

    // A repeat leaves every trained adapter alone.
    let again = train::run(&study, author, &base, &trainer).unwrap();
    assert_eq!((again.trained, again.already), (0, 4));
    assert_eq!(trainer.seeds.lock().unwrap().len(), 4);
}

#[test]
fn a_training_that_fails_keeps_the_manifests_of_the_runs_before_it_and_the_next_call_resumes() {
    let (study, author, base) = corpus_done("stages-resume");
    let failing = Scripted {
        fail_on: Some(2),
        ..Scripted::default()
    };
    let message = format!(
        "{:#}",
        train::run(&study, author, &base, &failing).unwrap_err()
    );
    assert!(message.contains("device was lost"), "{message}");
    let resumed = train::run(&study, author, &base, &Scripted::default()).unwrap();
    assert_eq!((resumed.trained, resumed.already), (2, 2));
}

#[test]
fn nothing_is_trained_before_the_corpus_stage_froze_the_examination() {
    let study = frozen("stages-early", small_protocol(|_| {}), 1);
    let author = study.protocol().authors[0].seed;
    let trainer = Scripted::default();
    let base = study.root().join("base.bin");
    std::fs::write(&base, "base weights").unwrap();
    let message = format!(
        "{:#}",
        train::run(&study, author, &Base::open(&base).unwrap(), &trainer).unwrap_err()
    );
    assert!(message.contains("study corpus"), "{message}");
    assert!(trainer.seeds.lock().unwrap().is_empty());
}

#[test]
fn an_adapter_is_not_retrained_on_another_base_and_not_on_changed_records() {
    let (study, author, base) = corpus_done("stages-refuse");
    train::run(&study, author, &base, &Scripted::default()).unwrap();
    let other = study.root().join("other.bin");
    std::fs::write(&other, "other weights").unwrap();
    let other = Base::open(&other).unwrap();
    let message = format!(
        "{:#}",
        train::run(&study, author, &other, &Scripted::default()).unwrap_err()
    );
    assert!(message.contains("base"), "{message}");

    let records = study.author_dir(author).join("dataset").join("train.jsonl");
    std::fs::write(&records, "{}\n").unwrap();
    let message = format!(
        "{:#}",
        train::run(&study, author, &base, &Scripted::default()).unwrap_err()
    );
    assert!(message.contains("records were changed"), "{message}");
}

#[test]
fn the_examination_needs_every_adapter_trained_and_names_the_one_that_is_not() {
    let (study, author, base) = corpus_done("stages-untrained");
    let loader = scripting();
    let message = format!(
        "{:#}",
        examine::run(&study, author, &base, &loader, false).unwrap_err()
    );
    assert!(message.contains("study train"), "{message}");
    assert!(
        loader.loads.lock().unwrap().is_empty(),
        "no model is loaded for nothing"
    );
}

#[test]
fn each_arm_is_examined_in_its_own_slot_under_the_manifest_of_its_model() {
    let (study, author, base) = corpus_done("stages-exam");
    train::run(&study, author, &base, &Scripted::default()).unwrap();
    let loader = scripting();
    let examined = examine::run(&study, author, &base, &loader, false).unwrap();
    let frozen = study.require_author_manifest(author).unwrap();

    // One load for the arms without an adapter and one per adapter.
    assert_eq!(loader.loads.lock().unwrap().len(), 1 + 4);
    let exam = study.author_dir(author).join("exam");
    let plan = study.author_plan(author).unwrap();
    let shared = exam.join(SHARED_SLOT);
    for arm in [
        "base",
        "named",
        "principles",
        "principles-verdicts",
        "retrieval",
        "retrieval-records-6",
    ] {
        let record = recorded(&shared, arm).unwrap().unwrap();
        assert_eq!(record.items_digest, frozen.exam.digest, "{arm}");
        assert!(record.model.adapter_digest.is_none());
    }
    for training_seed in &plan.training_seeds {
        let slot = exam.join(format!("tuned-s{training_seed}"));
        let record = recorded(&slot, "tuned").unwrap().unwrap();
        let run = study
            .run_manifest(
                author,
                synthetic_author::dataset::Briefings::Both,
                *training_seed,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            record.model.adapter_digest.as_deref(),
            Some(run.adapter_digest.as_str())
        );
        assert_eq!(record.items_digest, frozen.exam.digest);
        assert_eq!(record.model.decoding, "greedy");
    }
    assert!(exam
        .join(format!("tuned-reader-only-s{}", plan.training_seeds[0]))
        .exists());
    assert!(!exam
        .join(format!("tuned-reader-only-s{}", plan.training_seeds[1]))
        .exists());
    assert!(examined
        .arms
        .iter()
        .all(|(_, _, answered)| *answered == frozen.exam.items));

    // A repeat asks nothing again.
    let asked = loader.requests.load(Ordering::SeqCst);
    assert!(asked > 0);
    examine::run(&study, author, &base, &loader, false).unwrap();
    assert_eq!(loader.requests.load(Ordering::SeqCst), asked);
}

#[test]
fn the_sampled_decoding_is_asked_beside_the_greedy_one_with_its_seed_recorded() {
    let (study, author, base) = corpus_done("stages-sampled");
    train::run(&study, author, &base, &Scripted::default()).unwrap();
    let loader = scripting();
    examine::run(&study, author, &base, &loader, true).unwrap();
    let shared = study.author_dir(author).join("exam").join(SHARED_SLOT);
    let record = recorded(&shared, "base-sampled").unwrap().unwrap();
    let plan = study.protocol().decoding.secondary.unwrap();
    assert_eq!(record.draws, Some(plan.draws));
    assert!(
        record.model.decoding_seed.is_some(),
        "the drawn seed is recorded"
    );
    assert!(shared.join("base-sampled.jsonl").exists());
}

#[test]
fn an_adapter_file_that_is_not_the_one_trained_is_not_examined() {
    let (study, author, base) = corpus_done("stages-swapped");
    train::run(&study, author, &base, &Scripted::default()).unwrap();
    let seed = study.author_plan(author).unwrap().training_seeds[0];
    let run = study
        .run_manifest(author, synthetic_author::dataset::Briefings::Both, seed)
        .unwrap()
        .unwrap();
    std::fs::write(study.root().join(&run.adapter), "another adapter").unwrap();
    let loader = scripting();
    let message = format!(
        "{:#}",
        examine::run(&study, author, &base, &loader, false).unwrap_err()
    );
    assert!(
        message.contains("not the adapter its manifest records"),
        "{message}"
    );
    assert!(loader.loads.lock().unwrap().is_empty());
}

#[test]
fn an_amendment_before_training_keeps_the_corpus_and_is_refused_once_an_adapter_exists() {
    let (study, author, base) = corpus_done("stages-amend");
    let before = study.digest().to_string();
    let held = study.protocol().training.clone();
    let hold = if held.int8 {
        BaseHold::Bf16
    } else {
        BaseHold::Int8
    };

    let refused = amend_base_hold(study.root(), hold, " ", false).unwrap_err();
    assert!(refused.to_string().contains("reason"), "{refused}");
    let amended =
        amend_base_hold(study.root(), hold, "the base does not fit one card", false).unwrap();
    assert_ne!(amended.digest.to_string(), before);
    let recorded = &amended.protocol.amendments[0];
    assert_eq!(recorded.replaces, before);
    assert_eq!(recorded.reason, "the base does not fit one card");
    assert_eq!(amended.protocol.training.int8, hold == BaseHold::Int8);

    // The corpus written before the amendment is still the study's, and what
    // is trained now records the amended protocol and its precision.
    let study = Study::open(study.root()).unwrap();
    let trainer = Scripted::default();
    train::run(&study, author, &base, &trainer).unwrap();
    let plan = study.author_plan(author).unwrap();
    let run = study
        .run_manifest(
            author,
            synthetic_author::dataset::Briefings::Both,
            plan.training_seeds[0],
        )
        .unwrap()
        .unwrap();
    assert_eq!(run.protocol_digest, amended.digest.to_string());
    assert_eq!(run.settings.int8, hold == BaseHold::Int8);

    let again = amend_base_hold(
        study.root(),
        BaseHold::Fp32,
        "late",
        study.any_trained().unwrap(),
    )
    .unwrap_err();
    assert!(again.to_string().contains("trained"), "{again}");
}
