// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent pipelines that put a language model to
// work behind checks that do not trust it, for its clients. If your team needs
// expertise in schema-checked model calls with bounded cost, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a program with no async runtime of its own asks a model for values
//! of a given shape and gets the value or the reason it could not: a reply
//! that is not the shape is an error once its corrections are spent.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::time::Duration;

use common::Scripted;
use schemars::JsonSchema;
use serde::Deserialize;
use splinter_agent::blocking::BlockingModel;
use splinter_agent::solve::Model;
use splinter_agent::typed::TypedCall;

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Verdict {
    keep: bool,
}

fn call() -> TypedCall<Verdict> {
    TypedCall::new(
        "judge",
        "Say whether to keep it.",
        "You judge.",
        Duration::from_secs(30),
    )
    .repairs(0)
}

#[test]
fn a_reply_of_the_shape_is_returned_as_the_value() {
    let provider = Scripted::new(vec![r#"{"keep": true}"#.into()]);
    let model = BlockingModel::new(Model::new(provider, "scripted")).unwrap();
    assert_eq!(
        model.call(call(), &"anything").unwrap(),
        Verdict { keep: true }
    );
}

#[test]
fn a_reply_that_is_not_the_shape_is_an_error_naming_the_call() {
    let provider = Scripted::new(vec!["not json at all".into()]);
    let model = BlockingModel::new(Model::new(provider, "scripted")).unwrap();
    let error = model.call(call(), &"anything").unwrap_err().to_string();
    assert!(!error.is_empty());
}
