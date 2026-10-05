// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The transport reader against an independent implementation: `fixtures/
//! mixed.xpt` was written by pyreadstat and `fixtures/mixed.json` is what
//! pyreadstat reads back (`fixtures/make_xport.py` regenerates both).

use serde_json::Value;
use splinter_knowledge::tabular::{read_xport, Values};

#[test]
fn every_value_matches_the_reference_reader() {
    let table = read_xport(include_bytes!("fixtures/mixed.xpt")).unwrap();
    let want: Value = serde_json::from_str(include_str!("fixtures/mixed.json")).unwrap();
    assert_eq!(table.name, want["dataset"]);
    let cols = want["columns"].as_array().unwrap();
    assert_eq!(table.columns.len(), cols.len());
    assert_eq!(
        table.rows, 37,
        "the last row ends in a blank text field and is still a row"
    );
    assert_eq!(table.column("LBXGH").unwrap().label, "Glycohemoglobin (%)");
    for (col, w) in table.columns.iter().zip(cols) {
        assert_eq!(col.name, w["name"]);
        let wv = w["values"].as_array().unwrap();
        match &col.values {
            Values::Numeric(v) => {
                assert_eq!(v.len(), wv.len());
                for (i, (got, want)) in v.iter().zip(wv).enumerate() {
                    match (got, want.as_f64()) {
                        (None, None) => {}
                        (Some(g), Some(w)) => assert!(
                            (g - w).abs() <= 1e-14 * w.abs().max(1e-300),
                            "{}[{i}]: {g} vs {w}",
                            col.name
                        ),
                        (g, w) => panic!("{}[{i}]: {g:?} vs {w:?}", col.name),
                    }
                }
            }
            Values::Text(v) => {
                let wt: Vec<&str> = wv.iter().map(|x| x.as_str().unwrap()).collect();
                assert_eq!(v, &wt, "{}", col.name);
            }
        }
    }
}
