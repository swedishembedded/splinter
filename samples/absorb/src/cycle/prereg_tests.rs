// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specifications of [`super::prereg`].

use super::plan::CycleRole;
use super::prereg::*;
use super::verdict::Rule;

fn prereg() -> Prereg {
    Prereg {
        schema: SCHEMA.into(),
        run_counter: 1,
        pipeline_commit: "0123456789abcdef".into(),
        dirty: false,
        role: CycleRole::Test,
        days: 8,
        arm: "main".into(),
        absorb: AbsorbSettings::default(),
        probes_sha256: "p".into(),
        manifest_sha256: "m".into(),
        plan_sha256: "l".into(),
        items_sha256: "i".into(),
        anchor_sha256: "a".into(),
        analysis_sha256: analysis_sha256(),
        rule: Rule::protocol(),
        registered_at: "2026-10-10T00:00:00Z".into(),
        hash: String::new(),
    }
    .sealed()
    .expect("sealed")
}

#[test]
fn a_registration_is_written_once_and_any_edit_after_it_is_refused() {
    let dir = tempfile::tempdir().expect("dir");
    let registered = prereg();
    registered.verify().expect("an honest registration");
    registered.write(dir.path()).expect("written");
    assert!(
        registered.write(dir.path()).is_err(),
        "never registered again"
    );
    assert_eq!(Prereg::read(dir.path()).expect("read"), registered);

    let text = std::fs::read_to_string(dir.path().join(FILE)).expect("text");
    for (from, to) in [
        ("\"days\": 8", "\"days\": 9"),
        ("\"sign_test_alpha\": 0.05", "\"sign_test_alpha\": 0.5"),
        ("\"epochs\": 6", "\"epochs\": 7"),
    ] {
        assert!(text.contains(from), "{from}");
        std::fs::write(dir.path().join(FILE), text.replace(from, to)).expect("edit");
        let error = format!(
            "{:#}",
            Prereg::read(dir.path())
                .expect("read")
                .verify()
                .expect_err("edited")
        );
        assert!(error.contains("edited after it was registered"), "{error}");
    }
}

#[test]
fn a_report_from_other_analysis_code_than_was_registered_is_refused() {
    let mut other = prereg();
    other.analysis_sha256 = "0".repeat(64);
    let error = format!(
        "{:#}",
        other
            .sealed()
            .expect("sealed")
            .verify()
            .expect_err("other code")
    );
    assert!(error.contains("registered the analysis"), "{error}");
    assert_eq!(analysis_sha256().len(), 64);
    assert_eq!(analysis_sha256(), analysis_sha256());
}

#[test]
fn every_full_run_is_counted_in_the_ledger_of_its_fact_pool_among_the_runs_of_its_role() {
    let dir = tempfile::tempdir().expect("dir");
    let start = |role, run| Entry::Start {
        role,
        run,
        dir: "a".into(),
    };
    assert_eq!(next_run(dir.path(), CycleRole::Test).expect("count"), 1);
    record(dir.path(), &start(CycleRole::Dev, 1)).expect("recorded");
    record(dir.path(), &start(CycleRole::Dev, 2)).expect("recorded");
    assert_eq!(
        next_run(dir.path(), CycleRole::Test).expect("count"),
        1,
        "development cycles do not use up the numbers of the pre-registered runs"
    );
    record(dir.path(), &start(CycleRole::Test, 1)).expect("recorded");
    assert_eq!(
        next_run(dir.path(), CycleRole::Test).expect("count"),
        2,
        "a run is counted when it starts"
    );
    assert_eq!(next_run(dir.path(), CycleRole::Dev).expect("count"), 3);
    record(
        dir.path(),
        &Entry::Finished {
            hash: "h".into(),
            met: false,
            text: "not met".into(),
        },
    )
    .expect("recorded");
    assert_eq!(read_ledger(dir.path()).expect("ledger").len(), 4);
}

#[test]
fn the_absorb_of_a_night_is_the_registered_settings_over_the_nights_files() {
    let settings = AbsorbSettings::default();
    let request = settings
        .request(
            std::path::Path::new("/n/sessions"),
            std::path::Path::new("/n/subjects.jsonl"),
            std::path::Path::new("/r/probes.jsonl"),
        )
        .expect("a request");
    assert_eq!(request.sessions, [std::path::PathBuf::from("/n/sessions")]);
    assert_eq!(
        request.sealed_probes,
        [std::path::PathBuf::from("/r/probes.jsonl")]
    );
    assert_eq!(
        request.subjects,
        [std::path::PathBuf::from("/n/subjects.jsonl")]
    );
    assert!(
        !request.continue_from_release,
        "the nightly update retrains from the base"
    );
    assert!(request.entail && request.tuning.bf16_base);
    assert_eq!(
        request.policy,
        splinter_sdk::vocabulary::model_ref::ModelRef::policy_default()
    );
}
