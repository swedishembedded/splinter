// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for turning longitudinal records
// into participant-safe training datasets with leakage gates, for its
// clients. If your team needs expertise in survival and risk-model data
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: longitudinal episodes project to `timeline-v1` records that hold
//! only what was known at the prediction point as inputs, keep left
//! truncation and right censoring, carry no raw identifier, and are written
//! with a manifest that traces them to their episodes, files and terms. A
//! split's parts never share a group, and a planted leak fails the write.
#![allow(clippy::unwrap_used)]

use std::collections::{BTreeMap, HashSet};

use serde_json::json;
use splinter_core::longitudinal::{ParticipantKeying, Value};
use splinter_core::terms::{Permission, UsagePolicy};
use splinter_data::split::{temporal_split, LeakageError, Part};
use splinter_data::timeline_dataset::{
    members, project, project_split, write_timeline_dataset, write_timeline_splits, Dropped,
    PredictionPoint, ProjectionSpec, TimelineExclusion, TimelineManifest,
};
use splinter_data::{manifest_path, Format, Unchecked, ViewError};
use splinter_store::longitudinal::{ImportSpec, LongitudinalStore};
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

fn root(name: &str) -> StateRoot {
    let path =
        std::env::temp_dir().join(format!("splinter-projection-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    StateRoot::new(path)
}

fn spec(dataset: &str, policy: UsagePolicy) -> ImportSpec {
    ImportSpec {
        dataset: dataset.into(),
        terms: policy.terms(dataset),
        keying: ParticipantKeying::from_secret(b"campaign secret"),
    }
}

fn import(
    root: &StateRoot,
    file: &str,
    lines: &[serde_json::Value],
    spec: &ImportSpec,
) -> LongitudinalStore {
    let path = root.path().join(file);
    let text: Vec<String> = lines.iter().map(ToString::to_string).collect();
    std::fs::write(&path, text.join("\n") + "\n").unwrap();
    let store = LongitudinalStore::new(&Workspace::at(root));
    store.import_jsonl(&path, spec).unwrap();
    store
}

/// A participant examined at age 50 in 2000.0 whose record also holds
/// measurements and an event after the examination.
fn subject(id: &str, future_sbp: f64) -> serde_json::Value {
    json!({
        "subject_id": id, "group_id": format!("home-{id}"), "weight": 2.0, "source": "cycle-a",
        "entry": 50.0, "calendar_at_entry": 2000.0,
        "observations": [
            {"t": 40.0, "var": "weight", "value": 80},
            {"t": 50.0, "var": "sbp", "value": 130},
            {"t": 50.0, "var": "crp", "value": {"below": 0.2}},
            {"t": 50.001, "var": "sbp", "value": future_sbp},
            {"t": 55.0, "var": "sbp", "value": future_sbp}],
        "events": [{"t": 44.0, "code": "dx:hypertension"}, {"t": 50.0, "code": "dx:same-day"}, {"t": 58.0, "code": "death:heart"}],
        "at_risk": [{"code": "*", "from": 50.0, "to": 58.0}],
        "interventions": [{"t": 50.0, "code": "statin", "randomised": true},
                          {"t": 45.0, "code": "smoking", "randomised": false},
                          {"t": 52.0, "code": "diet", "randomised": true}]
    })
}

#[test]
fn nothing_after_the_prediction_point_reaches_an_input_whatever_its_value() {
    let root = root("future");
    let a = import(
        &root,
        "a.jsonl",
        &[subject("p1", 120.0)],
        &spec("cohort", UsagePolicy::ResearchOnly),
    );
    let b = import(
        &root_other("future-b"),
        "b.jsonl",
        &[subject("p1", 999.0)],
        &spec("cohort", UsagePolicy::ResearchOnly),
    );
    let (pa, pb) = (
        project(&a, &ProjectionSpec::at_entry()).unwrap(),
        project(&b, &ProjectionSpec::at_entry()).unwrap(),
    );
    let (ra, rb) = (&pa.records[0].record, &pb.records[0].record);
    assert_eq!(ra, rb, "a future measurement cannot change the record");
    let sbp: Vec<f64> = ra
        .observations
        .iter()
        .filter(|o| o.var == "sbp")
        .map(|o| match o.value {
            Value::Number(v) => v,
            _ => panic!(),
        })
        .collect();
    assert_eq!(sbp, [130.0], "only the measurement at entry is an input");
    assert!(ra.observations.iter().all(|o| o.t <= ra.entry));
    assert_eq!(pa.dropped[&Dropped::ObservationAfterPrediction], 2);
    assert_eq!(pa.dropped[&Dropped::InterventionAfterPrediction], 1);
    // Strictly before entry is history; at entry it is neither history nor outcome.
    let codes: Vec<&str> = ra.events.iter().map(|e| e.code.as_str()).collect();
    assert_eq!(codes, ["dx:hypertension", "death:heart"]);
    assert!(ra
        .events
        .iter()
        .filter(|e| e.t < ra.entry)
        .all(|e| e.code != "death:heart"));
}

fn root_other(name: &str) -> StateRoot {
    root(name)
}

#[test]
fn randomised_and_observational_interventions_are_distinct_inputs() {
    let root = root("assignment");
    let store = import(
        &root,
        "a.jsonl",
        &[subject("p1", 1.0)],
        &spec("cohort", UsagePolicy::ResearchOnly),
    );
    let projection = project(&store, &ProjectionSpec::at_entry()).unwrap();
    let levels: BTreeMap<&str, &Value> = projection.records[0]
        .record
        .observations
        .iter()
        .filter(|o| o.var.starts_with("intervention:"))
        .map(|o| (o.var.as_str(), &o.value))
        .collect();
    assert_eq!(
        levels["intervention:statin"],
        &Value::Category("randomised".into())
    );
    assert_eq!(
        levels["intervention:smoking"],
        &Value::Category("observational".into())
    );
    assert!(
        !levels.contains_key("intervention:diet"),
        "a later assignment is not an input"
    );
}

#[test]
fn left_truncation_and_right_censoring_survive_a_later_cutoff() {
    let root = root("windows");
    let delayed = json!({"subject_id": "late", "source": "cycle-a", "entry": 50.0, "calendar_at_entry": 2000.0,
        "events": [{"t": 57.0, "code": "death:other"}], "at_risk": [{"code": "*", "from": 55.0, "to": 57.0}]});
    let ended = json!({"subject_id": "gone", "source": "cycle-a", "entry": 50.0, "calendar_at_entry": 2000.0,
        "at_risk": [{"code": "*", "from": 50.0, "to": 52.0}]});
    let store = import(
        &root,
        "a.jsonl",
        &[subject("p1", 1.0), delayed, ended],
        &spec("cohort", UsagePolicy::ResearchOnly),
    );
    // Predict at 53 on every participant's clock.
    let at = ProjectionSpec {
        prediction: PredictionPoint::SubjectTime { t: 53.0 },
        administrative_end: None,
    };
    let projection = project(&store, &at).unwrap();
    assert_eq!(
        projection.excluded[&TimelineExclusion::NotAtRiskAfterPrediction],
        1,
        "follow-up that ended before the cutoff"
    );
    let by_events: Vec<_> = projection.records.iter().map(|r| &r.record).collect();
    let late = by_events
        .iter()
        .find(|r| r.at_risk[0].to == 57.0 && r.at_risk[0].from == 55.0)
        .unwrap();
    assert_eq!(
        late.at_risk[0].from, 55.0,
        "a window that opens after the cutoff stays delayed"
    );
    assert_eq!(late.events.len(), 1);
    let p1 = by_events.iter().find(|r| r.at_risk[0].to == 58.0).unwrap();
    assert_eq!(
        (p1.entry, p1.at_risk[0].from),
        (53.0, 53.0),
        "a window open at the cutoff starts there"
    );
    assert_eq!(
        p1.calendar_at_entry, 2003.0,
        "the calendar follows the clock"
    );
    assert!(
        p1.events.iter().any(|e| e.code == "dx:hypertension")
            && p1.events.iter().any(|e| e.code == "death:heart")
    );

    // Calendar time reads through each participant's own clock mapping.
    let cal = ProjectionSpec {
        prediction: PredictionPoint::CalendarTime { t: 2003.0 },
        administrative_end: None,
    };
    assert_eq!(project(&store, &cal).unwrap().records, projection.records);

    // An administrative end censors windows and hides what comes after it.
    let ended_at = ProjectionSpec {
        prediction: PredictionPoint::AtEntry,
        administrative_end: Some(2002.0),
    };
    let censored = project(&store, &ended_at).unwrap();
    let p1 = censored
        .records
        .iter()
        .find(|r| r.record.at_risk[0].from == 50.0 && r.record.observations.len() > 3)
        .unwrap();
    assert_eq!(p1.record.at_risk[0].to, 52.0);
    assert!(p1.record.events.iter().all(|e| e.code != "death:heart"));
    assert!(censored.dropped[&Dropped::AfterAdministrativeEnd] > 0);
}

#[test]
fn a_record_carries_only_opaque_keys_and_the_dataset_has_a_traceable_manifest() {
    let root = root("manifest");
    let store = import(
        &root,
        "a.jsonl",
        &[subject("p1", 1.0), subject("p2", 2.0)],
        &spec("cohort-a", UsagePolicy::Redistributable),
    );
    let second = root.path().join("b.jsonl");
    std::fs::write(
        &second,
        json!({"subject_id": "q1", "source": "site-b", "entry": 60.0, "calendar_at_entry": 2010.0,
        "at_risk": [{"code": "*", "from": 60.0, "to": 65.0}]})
        .to_string()
            + "\n",
    )
    .unwrap();
    store
        .import_jsonl(&second, &spec("cohort-b", UsagePolicy::Noncommercial))
        .unwrap();

    let projection = project(&store, &ProjectionSpec::at_entry()).unwrap();
    let out = root.path().join("out").join("timeline.jsonl");
    let dataset = write_timeline_dataset(&out, &projection, &Unchecked).unwrap();
    let text = std::fs::read_to_string(&out).unwrap();
    for raw in ["p1", "p2", "q1", "home-p1"] {
        assert!(
            !text.contains(&format!("\"{raw}\"")),
            "raw identifier {raw} in the dataset"
        );
    }
    for line in text.lines() {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(record["subject_id"].as_str().unwrap().len(), 64);
        assert_eq!(record["group_id"].as_str().unwrap().len(), 64);
    }
    assert_eq!(dataset.format, Format::TimelineV1);
    assert_eq!(dataset.records, 3);

    let manifest: TimelineManifest =
        serde_json::from_slice(&std::fs::read(manifest_path(&out)).unwrap()).unwrap();
    assert_eq!(manifest.dataset, dataset.digest);
    assert_eq!(manifest.records, 3);
    assert_eq!(manifest.episodes, store.addresses().unwrap());
    assert_eq!(manifest.sources.len(), 2);
    assert!(manifest.sources.iter().any(|s| s.dataset == "cohort-b"));
    let terms = manifest.terms.unwrap();
    assert_eq!(
        terms.redistribution,
        Permission::Allowed,
        "noncommercial allows redistribution"
    );
    assert_eq!(
        terms.commercial_use,
        Permission::Forbidden,
        "the most restrictive source binds"
    );

    // Deterministic: the same episodes and spec yield the same bytes.
    let again = write_timeline_dataset(
        &root.path().join("out").join("again.jsonl"),
        &project(&store, &ProjectionSpec::at_entry()).unwrap(),
        &Unchecked,
    )
    .unwrap();
    assert_eq!(again.digest, dataset.digest);

    // The dataset's lineage reaches the episodes, once however often it is reported.
    store
        .record_dataset(
            &dataset.digest,
            json!({"spec": "at-entry"}),
            3,
            &manifest.episodes,
        )
        .unwrap();
    store
        .record_dataset(
            &dataset.digest,
            json!({"spec": "at-entry"}),
            3,
            &manifest.episodes,
        )
        .unwrap();
    let unknown = splinter_core::digest::Digest::of(b"other dataset");
    assert!(store
        .record_dataset(
            &unknown,
            json!({}),
            1,
            &[splinter_core::digest::Digest::of(b"nope")]
        )
        .is_err());
}

/// People examined across 2000-2009 in households of two.
fn cohort(n: usize) -> Vec<serde_json::Value> {
    (0..n)
        .map(|i| {
            let entry = 40.0 + (i % 20) as f64;
            let calendar = 2000.5 + (i % 10) as f64;
            let source = ["cycle-a", "cycle-b"][i % 2];
            let at = |years: f64| entry + years;
            json!({
                "subject_id": format!("s{i}"), "group_id": format!("h{}", i / 2), "source": source,
                "entry": entry, "calendar_at_entry": calendar,
                "observations": (0..12).map(|y| json!({"t": at(y as f64), "var": "sbp", "value": 100.0 + y as f64})).collect::<Vec<_>>(),
                "events": [{"t": at(9.0), "code": "death:other"}],
                "at_risk": [{"code": "*", "from": entry, "to": at(9.0)}],
                "interventions": [{"t": entry, "code": "x", "randomised": i % 3 == 0}]
            })
        })
        .collect()
}

#[test]
fn a_temporal_split_projects_training_without_a_trace_of_the_future() {
    let root = root("temporal");
    let store = import(
        &root,
        "a.jsonl",
        &cohort(120),
        &spec("cohort", UsagePolicy::ResearchOnly),
    );
    let all = members(&store, |_| String::new()).unwrap();
    let cutoff = 2004.9;
    let dataset = splinter_core::digest::Digest::of(b"cohort");
    let split = temporal_split(&all, cutoff, 0.25, 11, &dataset).unwrap();
    let parts = project_split(&store, &split, &ProjectionSpec::at_entry()).unwrap();

    for part in [Part::Train, Part::Validation] {
        assert!(!parts[&part].records.is_empty());
        for r in &parts[&part].records {
            let record = &r.record;
            let offset = record.calendar_at_entry - record.entry;
            assert!(record.calendar_at_entry <= cutoff);
            assert!(
                record
                    .observations
                    .iter()
                    .all(|o| o.t + offset <= cutoff + 1e-9),
                "a post-cutoff observation is an input"
            );
            assert!(record.events.iter().all(|e| e.t + offset <= cutoff + 1e-9));
            assert!(
                record
                    .at_risk
                    .iter()
                    .all(|w| w.to + offset <= cutoff + 1e-9),
                "censored at the cutoff"
            );
        }
    }
    assert!(parts[&Part::Test]
        .records
        .iter()
        .all(|r| r.record.calendar_at_entry > cutoff));
    assert!(
        parts[&Part::Test].records.iter().any(|r| r
            .record
            .events
            .iter()
            .any(|e| e.t > r.record.entry)),
        "test outcomes are kept whole"
    );

    // Written, the parts never share a group, and a manifest names the split.
    let out = root.path().join("temporal-out");
    let written = write_timeline_splits(&out, &split, &parts, &Unchecked).unwrap();
    assert_eq!(written.split, split.digest().unwrap());
    let train: TimelineManifest = serde_json::from_slice(
        &std::fs::read(manifest_path(&written.datasets[&Part::Train].path)).unwrap(),
    )
    .unwrap();
    assert_eq!(
        (train.part, train.split),
        (Some(Part::Train), Some(written.split.clone()))
    );
    let groups = |part: Part| -> HashSet<String> {
        parts[&part]
            .records
            .iter()
            .map(|r| r.record.group_id.clone())
            .collect()
    };
    assert!(groups(Part::Train).is_disjoint(&groups(Part::Test)));
    assert!(groups(Part::Train).is_disjoint(&groups(Part::Validation)));
    assert!(out.join("split.json").is_file());
}

#[test]
fn a_planted_leak_fails_the_write_with_a_count_and_leaves_no_file() {
    let root = root("leak");
    let store = import(
        &root,
        "a.jsonl",
        &cohort(40),
        &spec("cohort", UsagePolicy::ResearchOnly),
    );
    let all = members(&store, |_| String::new()).unwrap();
    let split = temporal_split(
        &all,
        2004.9,
        0.25,
        11,
        &splinter_core::digest::Digest::of(b"c"),
    )
    .unwrap();
    let mut parts = project_split(&store, &split, &ProjectionSpec::at_entry()).unwrap();
    // Plant: a training record's group reappears in the test part.
    let mut planted = parts[&Part::Train].records[0].clone();
    planted.record.subject_id = "planted".into();
    parts.get_mut(&Part::Test).unwrap().records.push(planted);
    let out = root.path().join("leak-out");
    let error = write_timeline_splits(&out, &split, &parts, &Unchecked).unwrap_err();
    assert!(
        matches!(
            error,
            ViewError::Leakage(LeakageError::SharedGroups {
                groups: 1,
                units: 0
            })
        ),
        "{error}"
    );
    assert!(
        !out.exists() || std::fs::read_dir(&out).unwrap().next().is_none(),
        "nothing was written"
    );
}
