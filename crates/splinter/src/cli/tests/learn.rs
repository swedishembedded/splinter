// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for `learn`: the model that plans, and the share of
//! passages and abstentions taken from the records.

use super::{command, parse, Command};

#[test]
fn learn_names_the_model_that_plans() {
    let Command::Learn(plain) = command(&["learn", "docs"]) else {
        panic!("learn");
    };
    assert_eq!(plain.planner, None, "the generator plans by default");
    let Command::Learn(named) = command(&["learn", "docs", "--planner", "local:./big"]) else {
        panic!("learn --planner");
    };
    assert_eq!(named.planner, Some("local:./big".parse().unwrap()));
}

#[test]
fn passages_are_a_share_in_zero_to_one_not_zero() {
    let Command::Learn(learn) = command(&["learn", "docs", "--with-passages", "0.4"]) else {
        panic!("learn");
    };
    assert_eq!(learn.with_passages, Some(0.4));
    assert!(parse(&["learn", "docs", "--with-passages", "0"]).is_err());
    assert!(parse(&["learn", "docs", "--with-passages", "1.5"]).is_err());
    assert!(parse(&["learn", "docs", "--with-passages", "most"]).is_err());
}

#[test]
fn abstentions_are_taken_from_the_records_given_passages() {
    let Command::Learn(learn) = command(&[
        "learn",
        "docs",
        "--with-passages",
        "0.5",
        "--abstain",
        "0.1",
    ]) else {
        panic!("learn");
    };
    assert_eq!(learn.abstain, Some(0.1));
    assert!(parse(&["learn", "docs", "--abstain", "0.1"]).is_err());
}
