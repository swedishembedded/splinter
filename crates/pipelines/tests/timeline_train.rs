// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training of risk models on
// participant-safe longitudinal data, with the checkpoint, the split and the
// source files tied together by digest, for its clients. If your team needs
// expertise in training audited time-to-event models, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the train stage turns the stored parts of one split into an
//! immutable candidate. Statistics are fitted on training units only and
//! certified; the packed file unpacks and loads in plain brain and predicts
//! what the model as trained predicted; knots are derived from the training
//! outcome times and recorded; terms that do not permit training, parts of
//! different splits and the wrong part are refused. Trains small models: run
//! under the device lock.
#![allow(clippy::unwrap_used)]

mod common;

use common::timeline::{context, prepare, write_synthetic, ABSORBING, CODES};
use splinter_core::terms::UsagePolicy;
use splinter_data::split::Part;
use splinter_model::timeline::{read_jsonl, TimelineModel, TimelineTraining};
use splinter_pipelines::timeline::train::{
    load_bundle, load_timeline_candidate, train_timeline_candidate, TimelineTrainRequest,
};

fn small(seed: u64) -> TimelineTraining {
    let mut t = TimelineTraining::new(CODES, ABSORBING);
    t.steps = 60;
    t.batch = 64;
    t.max_tokens = Some(16);
    t.eval_interval = 20;
    t.seed = seed;
    t
}

#[test]
fn a_candidate_is_trained_on_training_units_only_packed_and_recorded() {
    let (scratch, ctx) = context("timeline-train");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", 500, 11);
    let split = prepare(&ctx, &file, "cohort", UsagePolicy::Redistributable, 3);
    let request = TimelineTrainRequest::new(
        &split.train.to_string(),
        &split.validation.to_string(),
        small(5),
    );
    let trained = train_timeline_candidate(&ctx, &request).unwrap();
    let c = &trained.candidate;

    // Lineage and certification.
    let store = ctx.timeline_datasets();
    let train_part = store.get(&split.train).unwrap();
    assert_eq!(c.split, split.split);
    assert_eq!(c.train, split.train);
    assert_eq!(c.held_out, split.validation);
    assert_eq!(c.episodes, train_part.manifest.episodes);
    assert_eq!(c.sources, train_part.manifest.sources);
    assert_eq!(c.seed, 5);
    assert_eq!(
        c.fit.units,
        split.records[&Part::Train],
        "the fit consumed exactly the training units"
    );
    assert_eq!(c.fit.split, split.split);
    assert!(c.brain_commit.is_some());
    assert_eq!(c.config_digest, c.config.digest().unwrap());

    // Knots were derived from the training outcome times and recorded.
    let knots = c.config.knots.as_ref().unwrap();
    assert_eq!(knots[0], 0.0);
    assert!(knots.len() > 3 && knots.windows(2).all(|w| w[1] > w[0]));

    // The file is the immutable artifact, and plain brain loads what is in it.
    assert!(std::fs::metadata(&trained.checkpoint)
        .unwrap()
        .permissions()
        .readonly());
    assert!(c.round_trip_max_abs_diff <= 1e-6);
    let unpack = scratch.0.join("plain");
    let loaded = load_bundle(&trained.checkpoint, &unpack, "model").unwrap();
    assert_eq!(loaded.codes(), CODES);
    let held = read_jsonl(&store.get(&split.validation).unwrap().path).unwrap();
    let again = TimelineModel::load(unpack.join("model")).unwrap();
    let (a, b) = (
        loaded.predict(&held[..5]).unwrap(),
        again.predict(&held[..5]).unwrap(),
    );
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.cif("death:a", 5.0), y.cif("death:a", 5.0));
    }

    // Its record is found again by id and never rewritten.
    let (id, found) = load_timeline_candidate(&ctx, &trained.id).unwrap();
    assert_eq!((id.as_str(), &found), (trained.id.as_str(), c));
}

#[test]
fn what_cannot_be_trained_from_is_refused_before_anything_is_kept() {
    let (scratch, ctx) = context("timeline-train-refused");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", 300, 12);
    let split = prepare(&ctx, &file, "cohort", UsagePolicy::Redistributable, 3);
    let (t, v, x) = (
        split.train.to_string(),
        split.validation.to_string(),
        split.test.to_string(),
    );
    // The wrong part is named.
    let swapped = train_timeline_candidate(&ctx, &TimelineTrainRequest::new(&v, &t, small(1)))
        .err()
        .unwrap();
    assert!(
        swapped.to_string().contains("not the Train part"),
        "{swapped}"
    );
    let test_as_train =
        train_timeline_candidate(&ctx, &TimelineTrainRequest::new(&x, &v, small(1)))
            .err()
            .unwrap();
    assert!(
        test_as_train.to_string().contains("Test"),
        "{test_as_train}"
    );
    // A request that names an absorbing code it does not predict.
    let mut bad = small(1);
    bad.absorbing = vec!["death:zzz".into()];
    let refused = train_timeline_candidate(&ctx, &TimelineTrainRequest::new(&t, &v, bad))
        .err()
        .unwrap();
    assert!(refused.to_string().contains("death:zzz"), "{refused}");
    assert_eq!(
        ctx.workspace()
            .document_ids("timeline_candidate")
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        ctx.artifacts()
            .list()
            .unwrap()
            .iter()
            .filter(|a| a.role == "checkpoint")
            .count(),
        0
    );
}

#[test]
fn terms_that_do_not_permit_training_refuse_it_and_other_splits_do_not_mix() {
    let (scratch, ctx) = context("timeline-train-terms");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", 300, 13);
    let unknown = prepare(&ctx, &file, "cohort", UsagePolicy::Unknown, 3);
    let why = train_timeline_candidate(
        &ctx,
        &TimelineTrainRequest::new(
            &unknown.train.to_string(),
            &unknown.validation.to_string(),
            small(1),
        ),
    )
    .err()
    .unwrap();
    assert!(why.to_string().contains("training is refused"), "{why}");

    // Another seed, another split: its parts do not pair with these.
    let (scratch2, ctx2) = context("timeline-train-splits");
    let file2 = write_synthetic(&scratch2.0, "cohort.jsonl", 300, 13);
    let a = prepare(&ctx2, &file2, "cohort", UsagePolicy::Redistributable, 3);
    let b = prepare(&ctx2, &file2, "cohort", UsagePolicy::Redistributable, 4);
    assert_ne!(a.split, b.split);
    let mixed = train_timeline_candidate(
        &ctx2,
        &TimelineTrainRequest::new(&a.train.to_string(), &b.validation.to_string(), small(1)),
    )
    .err()
    .unwrap();
    assert!(mixed.to_string().contains("split"), "{mixed}");
}
