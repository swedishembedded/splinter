// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A variable's codebook entry goes to a model as a typed call; its mapping
//! comes back as a [`MappingProposal`], and a reply naming a concept it was
//! not offered is sent back for correction rather than accepted.

#![allow(clippy::unwrap_used)]

mod common;

use std::time::Duration;

use common::Scripted;
use serde_json::json;
use splinter_agent::mapper::SvenMapper;
use splinter_agent::solve::Model;
use splinter_knowledge::codebook::{Code, VariableDoc};
use splinter_knowledge::harmonize::{admit, ConceptSpec, MappingProposer};

fn doc() -> VariableDoc {
    VariableDoc {
        name: "SMD030".into(),
        label: "Age started smoking cigarettes regularly".into(),
        text: "How old were you when you first started to smoke cigarettes fairly regularly?"
            .into(),
        target: "18+".into(),
        codes: vec![
            Code {
                value: "7 to 76".into(),
                meaning: "Range of Values".into(),
                count: None,
            },
            Code {
                value: "777".into(),
                meaning: "Refused".into(),
                count: None,
            },
        ],
    }
}

fn concepts() -> Vec<ConceptSpec> {
    vec![ConceptSpec {
        name: "smoking_start_age".into(),
        unit: "years".into(),
        description: "age at regular smoking onset".into(),
        reference: None,
    }]
}

#[tokio::test]
async fn a_mapping_is_asked_for_corrected_and_admitted() {
    let invented = json!({"concept": "age_started", "factor": 1.0, "valid_min": 5.0, "valid_max": 85.0, "missing_codes": [777.0], "quote": "first started to smoke"});
    let good = json!({"concept": "smoking_start_age", "factor": 1.0, "valid_min": 5.0, "valid_max": 85.0, "missing_codes": [777.0], "quote": "first started to smoke"});
    let model = Scripted::new(vec![invented.to_string(), good.to_string()]);
    let mapper = SvenMapper::new(
        Model::new(model.clone(), "scripted/generator-1"),
        Duration::from_secs(30),
    );
    assert_eq!(mapper.identity(), "scripted/generator-1");
    let p = mapper.propose(&doc(), &concepts()).await.unwrap();
    assert_eq!(p.concept, "smoking_start_age");
    let seen = model.seen.lock().unwrap();
    assert_eq!(
        seen.len(),
        2,
        "the reply with an unoffered concept is sent back once"
    );
    let asked = format!("{:?}", seen[0]);
    assert!(
        asked.contains("first started to smoke") && asked.contains("smoking_start_age"),
        "the model sees the codebook entry and the concepts"
    );
    assert!(admit(&p, &doc(), &[Some(16.0), Some(777.0)], &concepts()).admitted);
}
