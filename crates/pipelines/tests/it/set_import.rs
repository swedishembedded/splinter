// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data lineage that survives moving
// experience between machines and stores, for its clients. If your team needs
// expertise in curating agent experience across runs, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: an experience set moves from one state root into another with
//! everything said about its experiences.
//!
//! * The set keeps its id and its members in order; every experience keeps its
//!   id (its content address, checked as it is read and as it is written),
//!   its annotations, and the experiences its relations name come along.
//! * The examiner's trial register comes along for the imported experiences,
//!   so a trial imported as examination material still cannot enter a
//!   dataset.
//! * Importing again changes nothing.
//! * It works from a loose database into a packed one and back.

#![allow(clippy::unwrap_used)]

use crate::common;

use common::bench::{export, Trial, NO_OPTIONS};
use common::Scratch;
use splinter_core::annotation::{Annotation, AnnotationBody, Producer, RelationKind};
use splinter_expdb::backend::Layout;
use splinter_expdb::{Config, Database};
use splinter_orchestrator::Context;
use splinter_pipelines::bench::ingest_bench_trials;
use splinter_pipelines::bench::split;
use splinter_pipelines::set_import::import_sets;
use splinter_store::StateRoot;

/// A context over a root of the given layout.
fn rooted(test: &str, layout: Layout) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let root = scratch.state();
    std::fs::create_dir_all(root.expdb()).unwrap();
    drop(Database::open_as(root.expdb(), layout, Config::default()).unwrap());
    let ctx = Context::new(common::config(&scratch), false).unwrap();
    (scratch, ctx)
}

/// A source holding a train set of two trials, one related to the other,
/// and an eval trial.
fn source(test: &str, layout: Layout) -> (Scratch, Context, String) {
    let (scratch, ctx) = rooted(test, layout);
    let dir = export(
        &scratch.0.join("export"),
        &[
            Trial::new("t-pass", "org/a", Some("train"), true),
            Trial::new("t-fail", "org/a", Some("train"), false),
            Trial::new("e-pass", "org/b", Some("eval"), true),
        ],
    );
    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    let train = done.train_set.unwrap();
    let members = ctx.experiences().get_set(&train).unwrap().members;
    ctx.experiences()
        .annotate(&Annotation {
            experience: members[0].clone(),
            producer: Producer {
                name: "test/relation".into(),
                version: "1".into(),
            },
            body: AnnotationBody::Relation {
                kind: RelationKind::RevisionOf,
                other: members[1].clone(),
            },
        })
        .unwrap();
    (scratch, ctx, train.to_string())
}

fn roundtrip(from: Layout, to: Layout, test: &str) {
    let (source_scratch, source_ctx, train) = source(&format!("{test}-src"), from);
    let (target_scratch, target) = rooted(&format!("{test}-dst"), to);
    let root: StateRoot = source_scratch.state();
    assert_eq!(Layout::of_root(&root.expdb()).unwrap(), Some(from));
    assert_eq!(
        Layout::of_root(&target_scratch.state().expdb()).unwrap(),
        Some(to)
    );

    let imported = import_sets(&target, &root, std::slice::from_ref(&train)).unwrap();
    assert_eq!(imported.sets.len(), 1);
    let set = &imported.sets[0];
    assert_eq!(set.to_string(), train, "the set keeps its id");
    let theirs = source_ctx.experiences();
    let ours = target.experiences();
    let members = theirs.get_set(set).unwrap().members;
    assert_eq!(
        ours.get_set(set).unwrap().members,
        members,
        "and its members, in order"
    );
    for id in &members {
        assert_eq!(ours.get(id).unwrap(), theirs.get(id).unwrap());
        assert_eq!(
            ours.annotations(id).unwrap().annotations,
            theirs.annotations(id).unwrap().annotations,
            "every annotation comes along, in order"
        );
    }
    assert!(imported.experiences_new >= 2);
    // The register comes along for what was imported, and only for it.
    let register = split::trials(&target).unwrap();
    assert_eq!(register.len(), members.len());
    assert!(register.iter().all(|r| members.contains(&r.experience)));

    let again = import_sets(&target, &root, &[train]).unwrap();
    assert_eq!(again.experiences_new, 0, "importing again changes nothing");
    assert_eq!(again.annotations_new, 0);
    assert_eq!(again.experiences_held, imported.experiences_new);
}

#[test]
fn a_set_moves_from_a_loose_database_into_a_packed_one() {
    roundtrip(Layout::Loose, Layout::Packed, "import-loose-packed");
}

#[test]
fn a_set_moves_from_a_packed_database_into_a_loose_one() {
    roundtrip(Layout::Packed, Layout::Loose, "import-packed-loose");
}

#[test]
fn an_examination_trial_imported_stays_out_of_every_dataset() {
    let (source_scratch, source_ctx, _) = source("import-eval-src", Layout::Packed);
    let (_target_scratch, target) = rooted("import-eval-dst", Layout::Packed);
    let eval = source_ctx
        .experiences()
        .list_sets()
        .unwrap()
        .into_iter()
        .find(|s| {
            source_ctx
                .experiences()
                .get_set(s)
                .unwrap()
                .name
                .contains("eval")
        })
        .unwrap();
    let imported = import_sets(&target, &source_scratch.state(), &[eval.to_string()]).unwrap();
    let member = target
        .experiences()
        .get_set(&imported.sets[0])
        .unwrap()
        .members[0]
        .clone();
    assert!(
        splinter_pipelines::bench::admit_projection(&target, [member]).is_err(),
        "the register says it is examination material"
    );
}

#[test]
fn a_set_the_source_does_not_hold_is_refused() {
    let (source_scratch, _source_ctx, _) = source("import-missing-src", Layout::Packed);
    let (_target_scratch, target) = rooted("import-missing-dst", Layout::Packed);
    let err = import_sets(&target, &source_scratch.state(), &["blake3:00".into()]).unwrap_err();
    assert!(err.to_string().contains("blake3:00"), "{err}");
}
