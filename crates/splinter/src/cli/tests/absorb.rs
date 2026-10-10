// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for learning from sessions: `session`, `claims` and
//! `absorb`.

use super::{command, parse, Command};
use crate::cli::{ClaimsCommand, SessionCommand};
use splinter_sdk::vocabulary::model_ref::ModelRef;

#[test]
fn sessions_and_claims_are_one_stage_per_command() {
    let Command::Session(SessionCommand::Add { paths, max_bytes }) =
        command(&["session", "add", "a.atif.json", "dir"])
    else {
        panic!("session add");
    };
    assert_eq!(paths.len(), 2);
    assert_eq!(max_bytes, splinter_sdk::sessions::DEFAULT_MAX_SESSION_BYTES);
    assert!(parse(&["session", "add"]).is_err(), "a path is required");

    let Command::Claims(ClaimsCommand::Extract {
        sessions,
        generator,
        passes,
    }) = command(&["claims", "extract", "ab12", "cd34"])
    else {
        panic!("claims extract");
    };
    assert_eq!(sessions, ["ab12", "cd34"]);
    assert_eq!(generator, ModelRef::policy_default());
    assert_eq!(passes, 2, "two readings by default");
    assert!(
        parse(&["claims", "extract"]).is_err(),
        "a session is required"
    );
    assert!(matches!(
        command(&["claims", "gate", "ab12"]),
        Command::Claims(ClaimsCommand::Gate { .. })
    ));
    for verb in ["list", "ledger"] {
        assert!(matches!(command(&["claims", verb]), Command::Claims(_)));
    }
}

#[test]
fn absorb_takes_sessions_and_its_defaults_are_the_documented_ones() {
    let Command::Absorb(args) = command(&[
        "absorb",
        "./today",
        "./monday.atif.json",
        "--dry-run",
        "--sealed-probes",
        "a.jsonl",
        "b.jsonl",
        "--judge",
        "local:judge",
    ]) else {
        panic!("not an absorb");
    };
    assert_eq!(args.sessions.len(), 2);
    assert_eq!(args.sealed_probes.len(), 2);
    assert!(args.dry_run && !args.no_release && !args.continue_from_release);
    assert_eq!(args.epochs, splinter_sdk::absorb::DEFAULT_EPOCHS);
    assert_eq!(
        args.paraphrases as usize,
        splinter_sdk::absorb::kit::PARAPHRASES_WRITTEN
    );
    assert_eq!(
        args.rehearsal_share,
        splinter_sdk::rehearsal::DEFAULT_REHEARSAL_SHARE
    );
    assert!(args.judge.is_some() && args.generator.is_none());
    assert!(parse(&["absorb"]).is_err(), "sessions are needed");
    assert!(
        parse(&["absorb", "d", "--replay-fraction", "0.5"]).is_err(),
        "a replay fraction belongs to continuing a release"
    );
}

#[test]
fn absorb_trains_a_stance_pass_by_default_and_its_parameters_are_named() {
    let Command::Absorb(args) = command(&["absorb", "./today"]) else {
        panic!("not an absorb");
    };
    assert_eq!(
        args.stance_pairs as usize,
        splinter_sdk::absorb::stance::DEFAULT_PAIRS
    );
    assert_eq!(
        args.stance_tries,
        splinter_sdk::absorb::stance::DEFAULT_TRIES
    );
    assert!(args.stance_beta.is_none() && !args.no_stance);
    let Command::Absorb(args) = command(&[
        "absorb",
        "./today",
        "--stance-pairs",
        "12",
        "--stance-beta",
        "0.2",
    ]) else {
        panic!("not an absorb");
    };
    assert_eq!((args.stance_pairs, args.stance_beta), (12, Some(0.2)));
    let Command::Absorb(args) = command(&["absorb", "./today", "--no-stance"]) else {
        panic!("not an absorb");
    };
    assert!(args.no_stance);
    for conflicting in [
        ["--stance-pairs", "12"].as_slice(),
        ["--stance-beta", "0.2"].as_slice(),
    ] {
        let mut words = vec!["absorb", "./today", "--no-stance"];
        words.extend(conflicting);
        assert!(parse(&words).is_err(), "{conflicting:?} with --no-stance");
    }
    assert!(parse(&["absorb", "d", "--stance-beta", "0"]).is_err());
    assert!(parse(&["absorb", "d", "--stance-pairs", "0"]).is_err());
}
