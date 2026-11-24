// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for retrieval whose rankings are
// provably what they claim, for its clients. If your team needs expertise in
// lexical and hybrid search then you can procure our services by sending an
// email to info@swedishembedded.com.

//! BM25 rankings against an independent implementation of the formula the
//! index states (`validation/oracles/retrieval/bm25.py`, itself checked
//! against rank_bm25 where their conventions agree): the same passages, in
//! the same order, with the same scores. Both sum a few logarithms and
//! quotients per term: 1e-12.

use serde_json::{json, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_knowledge::retrieve::{Bm25, Passage};

#[test]
fn bm25_ranks_as_its_stated_formula_does() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "retrieval/bm25").unwrap();
    for case in fixture.cases() {
        let passages: Vec<Passage> = case["passages"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, text)| Passage::of_text("corpus", i, text.as_str().unwrap()))
            .collect();
        let k = usize::try_from(case["k"].as_u64().unwrap()).unwrap();
        let hits: Vec<Value> = Bm25::new(&passages)
            .search(case["query"].as_str().unwrap(), k)
            .iter()
            .map(|hit| json!({ "passage": hit.passage, "score": hit.score }))
            .collect();
        let actual =
            json!({ "passages": case["passages"], "query": case["query"], "k": k, "hits": hits });
        if let Err(mismatch) = compare(&actual, case, Tolerance::new(1e-12, 1e-12)) {
            panic!(
                "{} ({:?}): {mismatch}\ncase: {case}",
                fixture.name, fixture.source
            );
        }
    }
}
