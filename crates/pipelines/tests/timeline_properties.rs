// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible, leakage-free data preparation
// and checkable behaviour of risk models trained on longitudinal records, for
// its clients. If your team needs expertise in validating risk models before
// they ship, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the properties the timeline pipeline promises before any model is
//! trained. Ingestion is deterministic and idempotent; every participant is
//! accounted for; a split never puts a group in two parts, and a temporal
//! split carries nothing of the future into training.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::{BTreeMap, BTreeSet, HashSet};

use common::timeline::{context, write_synthetic};
use splinter_core::longitudinal::ParticipantKeying;
use splinter_core::terms::UsagePolicy;
use splinter_data::split::Part;
use splinter_data::timeline_dataset::ProjectionSpec;
use splinter_model::timeline::read_jsonl;
use splinter_pipelines::timeline::data::{
    import_records, split_timeline, ImportRequest, SplitPlan, SplitReport, SplitRequest,
};
use splinter_store::longitudinal::LongitudinalStore;

fn import(
    ctx: &splinter_orchestrator::Context,
    file: &std::path::Path,
) -> splinter_store::longitudinal::ImportReport {
    import_records(
        ctx,
        &ImportRequest {
            file: file.to_path_buf(),
            dataset: "cohort".into(),
            terms: UsagePolicy::Redistributable.terms("cohort"),
            secret: b"secret".to_vec(),
        },
    )
    .unwrap()
}

fn split(ctx: &splinter_orchestrator::Context, plan: SplitPlan) -> SplitReport {
    split_timeline(
        ctx,
        &SplitRequest {
            dataset: Some("cohort".into()),
            plan,
            seed: 9,
            projection: ProjectionSpec::at_entry(),
        },
    )
    .unwrap()
}

#[test]
fn ingestion_is_deterministic_idempotent_and_keeps_every_participant() {
    let (scratch, a) = context("timeline-ingest-a");
    let (_other, b) = context("timeline-ingest-b");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", 400, 5);
    let first = import(&a, &file);
    let second = import(&b, &file);
    assert_eq!(
        first, second,
        "the same file gives the same report in another state root"
    );
    assert_eq!(
        (first.rows, first.participants, first.added),
        (400, 400, 400)
    );
    let addresses = |ctx: &splinter_orchestrator::Context| {
        LongitudinalStore::new(ctx.workspace()).addresses().unwrap()
    };
    assert_eq!(
        addresses(&a),
        addresses(&b),
        "the same episodes, by address"
    );
    assert_eq!(addresses(&a).len(), 400, "one episode per participant");
    let again = import(&a, &file);
    assert_eq!(
        (again.added, again.already_present),
        (0, 400),
        "importing again adds nothing"
    );

    // Another secret is another set of opaque keys, not the same people twice.
    let (_c, c) = context("timeline-ingest-c");
    import_records(
        &c,
        &ImportRequest {
            file: file.clone(),
            dataset: "cohort".into(),
            terms: UsagePolicy::Redistributable.terms("cohort"),
            secret: b"another secret".to_vec(),
        },
    )
    .unwrap();
    assert_ne!(addresses(&a), addresses(&c));
    let _ = ParticipantKeying::from_secret(b"unused");
}

/// The subject ids and group ids of one stored part.
fn part_units(
    ctx: &splinter_orchestrator::Context,
    id: &splinter_core::dataset::DatasetId,
) -> Vec<(String, String)> {
    let stored = ctx.timeline_datasets().get(id).unwrap();
    read_jsonl(&stored.path)
        .unwrap()
        .into_iter()
        .map(|s| {
            (
                s.subject_id.clone(),
                s.group_id.clone().unwrap_or(s.subject_id),
            )
        })
        .collect()
}

#[test]
fn every_split_accounts_for_every_participant_and_never_shares_a_group() {
    let (scratch, ctx) = context("timeline-split");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", 600, 6);
    import(&ctx, &file);
    let plans = [
        SplitPlan::Participants {
            locked_share: 0.3,
            folds: 4,
        },
        SplitPlan::Temporal {
            cutoff: 2007.5,
            validation_share: 0.2,
        },
        SplitPlan::LeaveSourceOut {
            source: "cycle-b".into(),
            validation_share: 0.2,
        },
    ];
    for plan in plans {
        let report = split(&ctx, plan.clone());
        let in_parts: usize = report.records.values().sum();
        let left_out: usize = report.excluded.values().sum();
        assert_eq!(
            in_parts + left_out + report.unprojected,
            report.participants,
            "{plan:?}: every participant is in a record or counted as left out"
        );
        assert_eq!(report.participants, 600);

        // Isolation, read from the stored files and not from the split's own claim.
        let mut part_of: BTreeMap<String, Part> = BTreeMap::new();
        let mut groups: BTreeMap<String, BTreeSet<Part>> = BTreeMap::new();
        for (part, id) in [
            (Part::Train, &report.train),
            (Part::Validation, &report.validation),
            (Part::Test, &report.test),
        ] {
            for (subject, group) in part_units(&ctx, id) {
                assert!(
                    part_of.insert(subject, part).is_none(),
                    "{plan:?}: a subject is in two parts"
                );
                groups.entry(group).or_default().insert(part);
            }
        }
        assert!(
            groups.values().all(|parts| parts.len() == 1),
            "{plan:?}: a household was divided"
        );
        let split_doc = ctx.timeline_datasets().split(&report.split).unwrap();
        for part in Part::ALL {
            let from_files: HashSet<&str> = part_of
                .iter()
                .filter(|(_, p)| **p == part)
                .map(|(s, _)| s.as_str())
                .collect();
            let assigned: HashSet<&str> = split_doc.ids(part).iter().map(String::as_str).collect();
            assert!(
                from_files.is_subset(&assigned),
                "{plan:?}: {part:?} holds a record its split does not assign it"
            );
        }
    }
}

#[test]
fn a_temporal_split_carries_nothing_of_the_future_into_training() {
    let (scratch, ctx) = context("timeline-temporal");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", 800, 7);
    import(&ctx, &file);
    let cutoff = 2007.5;
    let report = split(
        &ctx,
        SplitPlan::Temporal {
            cutoff,
            validation_share: 0.2,
        },
    );
    for (part, id) in [
        (Part::Train, &report.train),
        (Part::Validation, &report.validation),
        (Part::Test, &report.test),
    ] {
        let subjects = read_jsonl(&ctx.timeline_datasets().get(id).unwrap().path).unwrap();
        assert!(!subjects.is_empty());
        for s in &subjects {
            // Calendar time of anything the record holds.
            let calendar = |t: f64| s.calendar_at_entry + (t - s.entry);
            if part == Part::Test {
                assert!(
                    s.calendar_at_entry > cutoff,
                    "a test unit entered at or before the cutoff"
                );
            } else {
                assert!(
                    s.calendar_at_entry <= cutoff,
                    "a {part:?} unit entered after the cutoff"
                );
                let latest = s
                    .observations
                    .iter()
                    .map(|o| calendar(o.t))
                    .chain(s.events.iter().map(|e| calendar(e.t)))
                    .chain(s.at_risk.iter().map(|w| calendar(w.to)))
                    .fold(f64::MIN, f64::max);
                assert!(
                    latest <= cutoff + 1e-6,
                    "a {part:?} record reaches {latest}, past the cutoff {cutoff}"
                );
            }
        }
    }
}
