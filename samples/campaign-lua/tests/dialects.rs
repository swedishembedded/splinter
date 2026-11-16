// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a model acquire several
// exactly verifiable skills one after another, each checked by an interpreter
// the model cannot influence, for its clients. If your team needs expertise
// in counterfactual task families whose rules no pretrained model knows you
// can procure our services by sending an email to info@swedishembedded.com.

//! Each counterfactual Lua dialect differs from stock Lua 5.1 exactly as its
//! definition says: every case of `data/dialect-cases.json` is run by the stock
//! interpreter and by the dialect's, and both outputs are the recorded ones.
//!
//! The dialect interpreters are built by `make deps/lua-dialects`
//! (`LUA_DIALECTS_DIR=<that directory>`); without them these tests fail, they
//! do not skip.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use campaign_lua::default_interpreter;
use campaign_lua::language::{Language, Runner};
use serde_json::Value;
use splinter_sdk::vocabulary::digest::Digest;

fn dialects_dir() -> PathBuf {
    std::env::var_os("LUA_DIALECTS_DIR")
        .map(PathBuf::from)
        .expect(
        "LUA_DIALECTS_DIR names the directory `make deps/lua-dialects` installed the dialects into",
    )
}

/// The dialect named `name`, pinned to the digest of the file installed (the
/// pin's own refusal is tested apart).
fn dialect(name: &str) -> Language {
    let interpreter = dialects_dir().join(name).join("bin/lua");
    let digest = Digest::sha256_of(&std::fs::read(&interpreter).unwrap()).to_string();
    Language::dialect(name, &interpreter, &digest, &[]).unwrap()
}

fn outcome(runner: &Runner, program: &str) -> String {
    runner
        .output(program)
        .unwrap_or_else(|_| "ERROR".to_string())
}

#[test]
fn every_dialect_differs_from_stock_lua_exactly_as_recorded() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("../data/dialect-cases.json")).unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let stock = Runner::new(
        &Language::lua51(&default_interpreter(), &[]),
        scratch.path(),
    )
    .unwrap();
    let mut differing = std::collections::BTreeMap::<String, usize>::new();
    for case in &cases {
        let name = case["dialect"].as_str().unwrap();
        let program = case["program"].as_str().unwrap();
        let runner = Runner::new(&dialect(name), scratch.path()).unwrap();
        let (want_stock, want_dialect) = (
            case["stock"].as_str().unwrap().trim(),
            case["dialect_output"].as_str().unwrap().trim(),
        );
        assert_eq!(outcome(&stock, program), want_stock, "stock: {program}");
        assert_eq!(outcome(&runner, program), want_dialect, "{name}: {program}");
        if want_stock != want_dialect {
            *differing.entry(name.to_string()).or_default() += 1;
        }
    }
    for name in ["zero-index", "plus-concat"] {
        assert!(
            differing.get(name).copied().unwrap_or(0) >= 8,
            "{name} is exercised by at least 8 programs that differ from stock"
        );
    }
}

#[test]
fn a_dialect_interpreter_that_is_not_the_pinned_file_is_refused() {
    let interpreter = dialects_dir().join("zero-index/bin/lua");
    let wrong = Language::dialect(
        "zero-index",
        &interpreter,
        &format!("sha256:{}", "0".repeat(64)),
        &[],
    )
    .unwrap();
    let error = Runner::new(&wrong, Path::new(".")).err().expect("refused");
    assert!(format!("{error:#}").contains("is pinned to"), "{error:#}");
}

#[test]
fn a_dialect_reports_a_lua_5_1_version_line_the_runtime_accepts() {
    let scratch = tempfile::tempdir().unwrap();
    let runner = Runner::new(&dialect("plus-concat"), scratch.path()).unwrap();
    assert!(runner
        .runtime()
        .runtime()
        .version
        .as_deref()
        .unwrap()
        .starts_with("Lua 5.1"));
}
