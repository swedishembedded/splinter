// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specifications of [`super::driver`]: a three-night cycle on invented
//! facts, run end to end with scripted models.

use std::path::Path;

use super::driver::{make_report, run, Repo, Request};
use super::fake::Fake;
use super::fixtures::{manifest, probes, Shape};
use super::items::{build, Item, Sources};
use super::plan::{generate, CanaryNights, CycleRole, PlanRequest};
use super::prereg::{read_ledger, AbsorbSettings, Entry};
use crate::facts::{Manifest, Origin, Sealed};
use crate::probes::{sealed_bytes, sha256_hex};
use crate::session::Style;

/// Writes the manifest sealed with its probes, and a screening record, into
/// `dir`; returns the manifest as written.
fn write(dir: &Path, mut manifest: Manifest) -> Manifest {
    let all = probes(&manifest);
    let bytes = sealed_bytes(&all).expect("bytes");
    std::fs::write(dir.join(crate::seal::SEALED), &bytes).expect("probes");
    manifest.probes = Some(Sealed {
        file: crate::seal::SEALED.into(),
        sha256: sha256_hex(&bytes),
        probes: all.len(),
        by_kind: Default::default(),
        unprobed: vec![],
        generator: "scripted".into(),
        sealed_at: "2026-01-01T00:00:00Z".into(),
    });
    manifest.write(dir).expect("manifest");
    // The day-0 wrong answer of an untaught fact is "port 1".
    let screening = serde_json::json!({
        "facts": manifest.facts.iter().map(|f| serde_json::json!({
            "id": f.id,
            "wrong_claim": if f.origin == Origin::Letters { vec!["nineteen".to_string()] } else { vec![] },
        })).collect::<Vec<_>>()
    });
    std::fs::write(dir.join("screening.json"), screening.to_string()).expect("screening");
    manifest
}

const ANCHOR: &str =
    "{\"instruction\": \"What is the capital of France?\", \"reference\": \"Paris\"}\n";

struct Pool {
    dir: tempfile::TempDir,
    manifest: Manifest,
}

fn pool() -> Pool {
    let dir = tempfile::tempdir().expect("dir");
    let manifest = write(
        dir.path(),
        manifest(&Shape {
            per_day: 2,
            days: 3,
            dev: 0,
            controls: 2,
        }),
    );
    std::fs::write(dir.path().join("anchor.jsonl"), ANCHOR).expect("anchor");
    let canary = manifest
        .facts
        .iter()
        .find(|f| f.canary.is_some())
        .expect("a canary")
        .id
        .clone();
    let plan = generate(&PlanRequest {
        manifest: &manifest,
        role: CycleRole::Test,
        nights: 3,
        per_night: 2,
        noise_per_night: 1,
        styles: vec![Style::Plain],
        null_night: None,
        sham_night: None,
        canary: Some(CanaryNights {
            fact: canary,
            night: 1,
            back: 3,
        }),
    })
    .expect("a plan");
    std::fs::write(
        dir.path().join("plan.json"),
        serde_json::to_string_pretty(&plan).expect("json"),
    )
    .expect("plan");
    Pool { dir, manifest }
}

fn request(pool: &Pool, out: &Path) -> Request {
    Request {
        dir: out.to_path_buf(),
        manifest_dir: pool.dir.path().to_path_buf(),
        plan: pool.dir.path().join("plan.json"),
        anchor: pool.dir.path().join("anchor.jsonl"),
        days: 3,
        arm: "main".into(),
        settings: AbsorbSettings::default(),
        repo: Repo {
            commit: "abc1234".into(),
            dirty: false,
        },
        now: "2026-10-10T00:00:00Z".into(),
    }
}

fn fake(pool: &Pool) -> Fake {
    let probes = crate::seal::verified_probes(pool.dir.path(), &pool.manifest).expect("probes");
    let items: Vec<Item> = build(&Sources {
        manifest: &pool.manifest,
        probes: &probes,
        wrong_claims: &Default::default(),
        anchor: ANCHOR,
        role: CycleRole::Test,
    })
    .expect("items");
    Fake::new(items)
}

#[test]
fn a_three_night_cycle_runs_back_to_back_and_reports_what_it_measured() {
    let pool = pool();
    let out = tempfile::tempdir().expect("out");
    let world = fake(&pool);
    let report = run(&request(&pool, out.path()), &world).expect("a cycle");

    assert_eq!(
        world.calls(),
        [
            "answers", "judge", "record:1", "absorb:1", "answers", "judge", "record:2", "absorb:2",
            "answers", "judge", "record:3", "absorb:3", "answers", "judge"
        ],
        "each night evaluates with one load of each model, after its update"
    );
    assert!(report.verdict.met, "{}", report.verdict.text);
    for file in [
        "prereg.json",
        "plan.json",
        "items.json",
        "report.json",
        "report.md",
    ] {
        assert!(out.path().join(file).exists(), "{file}");
    }
    assert_eq!(report.nights.len(), 4, "the base and three nights");
    assert_eq!(report.nights[3].release.as_deref(), Some("release-3"));
    assert_eq!(
        report.meta.iter().map(|m| m.sessions).collect::<Vec<_>>(),
        [3, 3, 4]
    );
    // The canary followed the false statement, then the correction.
    let canary: Vec<(usize, bool, bool)> = report
        .verdict
        .canary
        .iter()
        .map(|c| (c.night, c.adopted, c.true_recall))
        .collect();
    assert_eq!(
        canary,
        [(1, true, false), (2, true, false), (3, false, true)]
    );

    let text = std::fs::read_to_string(out.path().join("report.md")).expect("markdown");
    assert!(text.contains("PRIMARY CRITERION MET"), "{text}");
    assert!(text.contains("3/3 (0.29-1.00)"), "{text}");
    assert!(
        text.contains("not measured"),
        "no scoring call: absent, not zero"
    );
    let ledger = read_ledger(pool.dir.path()).expect("ledger");
    assert!(matches!(
        ledger.as_slice(),
        [
            Entry::Start { run: 1, .. },
            Entry::Registered { run: 1, .. },
            Entry::Finished {
                run: 1,
                met: true,
                ..
            }
        ]
    ));
}

#[test]
fn a_cycle_that_stops_in_the_middle_of_a_night_resumes_without_repeating_finished_work() {
    let pool = pool();
    let out = tempfile::tempdir().expect("out");
    let world = fake(&pool);
    world.fail_absorb_on(Some(2));
    let error = format!(
        "{:#}",
        run(&request(&pool, out.path()), &world).expect_err("lost the card")
    );
    assert!(
        error.contains("night 2") && error.contains("the card was lost"),
        "{error}"
    );
    assert!(out.path().join("nights/night-01/evaluation.json").exists());
    assert!(out.path().join("nights/night-02/record.json").exists());
    assert!(!out.path().join("nights/night-02/absorb.json").exists());

    world.fail_absorb_on(None);
    let report = run(&request(&pool, out.path()), &world).expect("resumed");
    assert!(report.verdict.met, "{}", report.verdict.text);
    let calls = world.calls();
    let count = |call: &str| calls.iter().filter(|c| *c == call).count();
    assert_eq!(count("record:1"), 1, "{calls:?}");
    assert_eq!(
        count("record:2"),
        1,
        "the recorded night is not recorded again"
    );
    assert_eq!(count("absorb:1"), 1);
    assert_eq!(count("absorb:2"), 2, "failed once, then done");
    assert_eq!(count("answers"), 4, "night 0 and each night once");
    assert_eq!(
        read_ledger(pool.dir.path()).expect("ledger").len(),
        3,
        "one run, not two"
    );
}

#[test]
fn a_registered_run_cannot_be_changed_and_a_report_is_refused_when_the_registration_does_not_hold()
{
    let pool = pool();
    let out = tempfile::tempdir().expect("out");
    let world = fake(&pool);
    run(&request(&pool, out.path()), &world).expect("a cycle");

    let mut other = request(&pool, out.path());
    other.settings.epochs += 1;
    let error = format!("{:#}", run(&other, &world).expect_err("other inputs"));
    assert!(error.contains("registered with other inputs"), "{error}");

    let prereg = out.path().join("prereg.json");
    let original = std::fs::read_to_string(&prereg).expect("prereg");
    std::fs::write(&prereg, original.replace("\"days\": 3", "\"days\": 2")).expect("edit");
    let error = format!(
        "{:#}",
        make_report(out.path(), pool.dir.path()).expect_err("edited")
    );
    assert!(error.contains("edited after it was registered"), "{error}");
    std::fs::write(&prereg, &original).expect("restore");
    make_report(out.path(), pool.dir.path()).expect("the honest report");

    let evaluation = out.path().join("nights/night-02/evaluation.json");
    let text = std::fs::read_to_string(&evaluation).expect("evaluation");
    let hash = serde_json::from_str::<serde_json::Value>(&original).expect("json")["hash"]
        .as_str()
        .expect("hash")
        .to_string();
    std::fs::write(&evaluation, text.replace(&hash, &"0".repeat(64))).expect("edit");
    let error = format!(
        "{:#}",
        make_report(out.path(), pool.dir.path()).expect_err("foreign night")
    );
    assert!(error.contains("night 2 was evaluated under"), "{error}");

    std::fs::write(&evaluation, &text).expect("restore");

    let plan = out.path().join("plan.json");
    let text = std::fs::read_to_string(&plan).expect("plan");
    std::fs::write(&plan, text.replace("\"noise\"", "\"fact\"")).expect("edit");
    let error = format!(
        "{:#}",
        make_report(out.path(), pool.dir.path()).expect_err("edited plan")
    );
    assert!(error.contains("plan.json hashes to"), "{error}");
}

#[test]
fn the_test_facts_are_not_run_from_an_unclean_tree_and_every_full_run_is_counted() {
    let pool = pool();
    let out = tempfile::tempdir().expect("out");
    let mut dirty = request(&pool, &out.path().join("a"));
    dirty.repo.dirty = true;
    let error = format!("{:#}", run(&dirty, &fake(&pool)).expect_err("dirty"));
    assert!(error.contains("clean tree"), "{error}");
    assert!(
        read_ledger(pool.dir.path()).expect("ledger").is_empty(),
        "nothing started"
    );

    run(&request(&pool, &out.path().join("b")), &fake(&pool)).expect("run 1");
    let report = run(&request(&pool, &out.path().join("c")), &fake(&pool)).expect("run 2");
    let runs: Vec<usize> = report.runs.iter().map(|r| r.run).collect();
    assert_eq!(runs, [1, 2], "every full run is listed");
    assert!(report.run.starts_with("run 2 at commit abc1234"));
    let text = std::fs::read_to_string(out.path().join("c/report.md")).expect("markdown");
    assert!(
        text.contains("run 1 in") && text.contains("run 2 in"),
        "{text}"
    );
}
