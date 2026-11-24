// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval measured against known answers,
// for its clients. If your team needs expertise in lexical, semantic and
// hybrid search, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec, against an independent reference (plan 3C): on a corpus whose
//! relevant passages are known (`validation/oracles/retrieval/benchmark.py`:
//! an exact match, a match by meaning only, several relevant passages,
//! distractors, a query nothing answers, a repeated passage, conflicting
//! versions, and the corpus reordered), BM25, dense search over the
//! benchmark's word vectors and their reciprocal-rank fusion rank every
//! passage exactly as the reference does. The benchmark's recall, precision,
//! reciprocal rank and nDCG of each ranking, and of rank_bm25's own, are
//! recorded with the fixture.
#![allow(clippy::unwrap_used)]

use serde_json::{json, Map, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_core::text::words;
use splinter_knowledge::retrieve::{fuse, Bm25, Dense, EmbedError, Embedder, Hit, Passage};

/// A text's vector: the sum of its words' vectors in the benchmark's table.
struct Table(Map<String, Value>);

impl Embedder for Table {
    fn name(&self) -> String {
        "retrieval-benchmark-table".into()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut total = vec![0.0f32; 5];
                for word in words(text) {
                    if let Some(v) = self.0.get(&word) {
                        for (axis, x) in v.as_array().unwrap().iter().enumerate() {
                            total[axis] += x.as_f64().unwrap() as f32;
                        }
                    }
                }
                total
            })
            .collect())
    }
}

fn order(hits: &[Hit]) -> Value {
    json!(hits.iter().map(|h| h.passage).collect::<Vec<_>>())
}

#[test]
fn bm25_dense_and_fused_rank_the_benchmark_as_the_reference_does() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "retrieval/benchmark").unwrap();
    let table = Table(fixture.value["vectors"].as_object().unwrap().clone());
    let texts: Vec<&str> = fixture.value["passages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    for case in fixture.cases() {
        let passages: Vec<Passage> = case["order"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(at, i)| Passage::of_text("notes", at, texts[i.as_u64().unwrap() as usize]))
            .collect();
        let query = case["query"].as_str().unwrap();
        let depth = passages.len();
        let lexical = Bm25::new(&passages).search(query, depth);
        let semantic = Dense::new(&passages, &table)
            .unwrap()
            .search(query, &table, depth)
            .unwrap();
        let hybrid = fuse(&[lexical.clone(), semantic.clone()], depth);
        let mut actual = case.clone();
        actual["rankings"] =
            json!({"bm25": order(&lexical), "dense": order(&semantic), "fused": order(&hybrid)});
        if let Err(mismatch) = compare(&actual, case, Tolerance::new(0.0, 0.0)) {
            panic!("{mismatch}\ncase: {case}");
        }
    }
}
