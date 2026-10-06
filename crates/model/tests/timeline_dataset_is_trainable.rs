// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training survival and risk
// models on longitudinal records without leaking identities or the future,
// for its clients. If your team needs expertise in leakage-free risk-model
// training data, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the `timeline-v1` datasets projected from longitudinal episodes are
//! what brain's own parser reads, at any prediction point. The writer's
//! belief about the format is not evidence; brain's parser, on the other
//! side of the repository boundary, is.

#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_core::longitudinal::ParticipantKeying;
use splinter_core::terms::UsagePolicy;
use splinter_data::timeline_dataset::{
    project, write_timeline_dataset, PredictionPoint, ProjectionSpec,
};
use splinter_model::timeline::read_jsonl;
use splinter_model::BrainDatasetCheck;
use splinter_store::longitudinal::{ImportSpec, LongitudinalStore};
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

#[test]
fn projected_records_are_read_whole_by_brains_parser() {
    let root = StateRoot::new(std::env::temp_dir().join(format!(
        "splinter-timeline-trainable-{}",
        std::process::id()
    )));
    let _ = std::fs::remove_dir_all(root.path());
    std::fs::create_dir_all(root.path()).unwrap();
    let lines: Vec<String> = (0..30)
        .map(|i| {
            let entry = 45.0 + (i % 10) as f64;
            json!({
                "subject_id": format!("s{i}"), "group_id": format!("h{}", i / 2), "weight": 1.5, "source": "cycle-a",
                "entry": entry, "calendar_at_entry": 2000.0 + (i % 5) as f64,
                "observations": [{"t": entry, "var": "sbp", "value": 120 + i},
                                 {"t": entry, "var": "crp", "value": {"below": 0.2}},
                                 {"t": entry, "var": "smoking", "value": "never"},
                                 {"t": entry + 2.0, "var": "sbp", "value": 150}],
                "events": [{"t": entry - 3.0, "code": "dx:hypertension"}, {"t": entry + 6.0, "code": "death:heart"}],
                "at_risk": [{"code": "*", "from": entry, "to": entry + 6.0}],
                "interventions": [{"t": entry, "code": "statin", "randomised": i % 2 == 0}]
            })
            .to_string()
        })
        .collect();
    let file = root.path().join("cohort.jsonl");
    std::fs::write(&file, lines.join("\n") + "\n").unwrap();
    let store = LongitudinalStore::new(&Workspace::at(&root));
    store
        .import_jsonl(
            &file,
            &ImportSpec {
                dataset: "cohort".into(),
                terms: UsagePolicy::ResearchOnly.terms("cohort"),
                keying: ParticipantKeying::from_secret(b"secret"),
            },
        )
        .unwrap();

    for (name, prediction) in [
        ("entry", PredictionPoint::AtEntry),
        ("later", PredictionPoint::SubjectTime { t: 52.0 }),
    ] {
        let spec = ProjectionSpec {
            prediction,
            administrative_end: None,
        };
        let projection = project(&store, &spec).unwrap();
        let out = root.path().join(format!("{name}.jsonl"));
        write_timeline_dataset(&out, &projection, &BrainDatasetCheck).unwrap();
        let subjects = read_jsonl(&out).unwrap();
        assert_eq!(subjects.len(), projection.records.len(), "{name}");
        assert!(
            subjects
                .iter()
                .all(|s| s.known_observations().all(|o| o.t <= s.entry)),
            "{name}"
        );
    }
}
