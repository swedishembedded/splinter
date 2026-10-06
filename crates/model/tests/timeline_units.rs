// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that stop a risk model from being
// fed a measurement in the wrong unit, for its clients. If your team needs
// expertise in unit-safe clinical prediction you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a model trained on records that state units records them in its
//! vocabulary, and a measurement stated in another unit is refused at
//! prediction (never converted), by brain itself. Trains a tiny model: run
//! under the device lock.
#![allow(clippy::unwrap_used)]

use splinter_model::timeline::{synthetic, train_timeline, TimelineModel, TimelineTraining};

fn with_unit(subjects: &mut [splinter_model::timeline::Subject], unit: &str) {
    for s in subjects {
        for o in s.observations.iter_mut().filter(|o| o.var == "x1") {
            o.unit = Some(unit.into());
        }
    }
}

#[test]
fn a_trained_model_records_the_unit_and_refuses_another() {
    let (mut all, _) = synthetic::population(300, 5);
    with_unit(&mut all, "mmol/L");
    let (train, held) = all.split_at(220);
    let mut config = TimelineTraining::new(["death:a", "death:b", "onset"], ["death:a", "death:b"]);
    config.steps = 3;
    config.batch = 32;
    config.max_tokens = Some(16);
    let trained = train_timeline(train, held, &config).unwrap();
    let dir = std::env::temp_dir().join(format!("splinter-units-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    trained.model.save(&dir).unwrap();
    let vocab = std::fs::read_to_string(dir.join("vocab.json")).unwrap();
    assert!(vocab.contains("mmol/L"), "the unit is recorded: {vocab}");

    let model = TimelineModel::load(&dir).unwrap();
    assert!(model.predict(&held[..2]).is_ok());
    let mut wrong = held[..2].to_vec();
    with_unit(&mut wrong, "mg/dL");
    let why = model.predict(&wrong).err().unwrap().to_string();
    assert!(
        why.contains("x1") && why.contains("mmol/L") && why.contains("mg/dL"),
        "{why}"
    );
}
