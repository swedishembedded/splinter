// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for curricula whose prerequisite
// order is provably right, for its clients. If your team needs expertise in
// skill graphs and learning curricula then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A skill graph's order against NetworkX
//! (`validation/oracles/graphs/skill_order.py`): acyclic graphs are ordered
//! in rounds exactly as `topological_generations` with each round in the
//! graph's order; a cyclic graph is refused naming exactly the skills on or
//! behind a cycle. Frozen, or drawn under
//! `scripts/conformance/external.py differential graphs/skill-order`.

#![allow(clippy::unwrap_used)]

use serde_json::{json, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_core::digest::Digest;
use splinter_core::source::SourceId;
use splinter_knowledge::concepts::SectionRef;
use splinter_knowledge::skills::{GraphError, Skill, SkillGraph};

fn graph(skills: &Value) -> SkillGraph {
    let skills = skills
        .as_array()
        .unwrap()
        .iter()
        .map(|s| Skill {
            id: s["id"].as_str().unwrap().into(),
            title: s["id"].as_str().unwrap().into(),
            sections: vec![SectionRef {
                source: SourceId(Digest::of(b"manual")),
                part: "manual.md".into(),
                section: 0,
            }],
            symbols: Vec::new(),
            constraints: Vec::new(),
            requirements: Vec::new(),
            prerequisites: serde_json::from_value(s["prerequisites"].clone()).unwrap(),
        })
        .collect();
    SkillGraph {
        objective: "o".into(),
        revision: 1,
        parent: None,
        sources: Vec::new(),
        skills,
    }
}

#[test]
fn a_skill_graph_is_ordered_and_its_cycles_named_as_networkx_finds_them() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "graphs/skill-order").unwrap();
    for case in fixture.cases() {
        let graph = graph(&case["skills"]);
        let found = match graph.order() {
            Ok(order) => json!({
                "skills": case["skills"], "acyclic": true,
                "order": order.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
            }),
            Err(GraphError::Cycle(stuck)) => {
                json!({ "skills": case["skills"], "acyclic": false, "stuck": stuck })
            }
            Err(other) => panic!("{other}"),
        };
        if let Err(mismatch) = compare(&found, case, Tolerance::EXACT) {
            panic!(
                "{} ({:?}): {mismatch}\ncase: {case}",
                fixture.name, fixture.source
            );
        }
        assert_eq!(graph.validate().is_ok(), case["acyclic"] == true, "{case}");
    }
}
