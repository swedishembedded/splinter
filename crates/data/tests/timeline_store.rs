// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements versioned, content-addressed datasets of
// longitudinal records with their splits, for its clients. If your team
// needs expertise in reproducible cohort datasets for risk models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a written timeline dataset is stored under the digest of its
//! manifest and read back verified; a split is stored under its own address
//! and the parts written from it say which split they are of.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_core::longitudinal::ParticipantKeying;
use splinter_core::terms::UsagePolicy;
use splinter_data::split::{temporal_split, Part};
use splinter_data::timeline_dataset::{
    members, project_split, write_timeline_splits, ProjectionSpec,
};
use splinter_data::timeline_store::TimelineStore;
use splinter_data::Unchecked;
use splinter_store::longitudinal::{ImportSpec, LongitudinalStore};
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

fn line(i: usize) -> serde_json::Value {
    let entry = 50.0;
    json!({
        "subject_id": format!("s{i}"), "group_id": format!("h{}", i / 2), "source": "cycle",
        "entry": entry, "calendar_at_entry": 2000.0 + (i % 10) as f64,
        "observations": [{"t": entry, "var": "sbp", "value": 100.0 + i as f64}],
        "events": [{"t": entry + 3.0, "code": "death:other"}],
        "at_risk": [{"code": "*", "from": entry, "to": entry + 5.0}]
    })
}

#[test]
fn a_dataset_and_its_split_are_stored_by_address_and_read_back_verified() {
    let dir = tempfile::tempdir().unwrap();
    let root = StateRoot::new(dir.path().join("state"));
    let workspace = Workspace::at(&root);
    let file = dir.path().join("cohort.jsonl");
    let text: Vec<String> = (0..60).map(|i| line(i).to_string()).collect();
    std::fs::write(&file, text.join("\n") + "\n").unwrap();
    let longitudinal = LongitudinalStore::new(&workspace);
    longitudinal
        .import_jsonl(
            &file,
            &ImportSpec {
                dataset: "cohort".into(),
                terms: UsagePolicy::Redistributable.terms("cohort"),
                keying: ParticipantKeying::from_secret(b"secret"),
            },
        )
        .unwrap();
    let all = members(&longitudinal, |_| String::new()).unwrap();
    let basis = splinter_core::digest::Digest::of(b"cohort");
    let split = temporal_split(&all, 2004.5, 0.25, 3, &basis).unwrap();
    let parts = project_split(&longitudinal, &split, &ProjectionSpec::at_entry()).unwrap();
    let written =
        write_timeline_splits(&dir.path().join("parts"), &split, &parts, &Unchecked).unwrap();

    let store = TimelineStore::new(&workspace, &root);
    let address = store.put_split(&split).unwrap();
    assert_eq!(address, written.split);
    assert_eq!(store.split(&address).unwrap(), split);

    let train = store.put(&written.datasets[&Part::Train]).unwrap();
    assert_eq!(train.id.0, written.datasets[&Part::Train].manifest);
    assert_eq!(train.manifest.part, Some(Part::Train));
    assert_eq!(train.manifest.split, Some(address));
    assert_eq!(
        store.put(&written.datasets[&Part::Train]).unwrap(),
        train,
        "storing again finds it"
    );
    assert_eq!(store.get(&train.id).unwrap(), train);
    assert_eq!(store.list().unwrap(), vec![train.id.clone()]);

    let mut mode = std::fs::metadata(&train.path).unwrap().permissions();
    assert!(mode.readonly(), "a stored file is immutable");
    #[allow(clippy::permissions_set_readonly_false)]
    mode.set_readonly(false);
    std::fs::set_permissions(&train.path, mode).unwrap();
    std::fs::write(&train.path, "tampered\n").unwrap();
    assert!(store.get(&train.id).is_err(), "an altered file is refused");
}
