// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements promotion gates that move a champion only on
// signed, independent evidence, for its clients. If your team needs
// expertise in sealed evaluation of learning agents you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: on the agent protocol the evaluate stage examines through bench's
//! own run and verdict path.
//!
//! * Each of the dev split, the retention suite and the generation's sealed
//!   slice is run by bench (`bench run`) once for the champion and once for
//!   the candidate, each under a profile pointed at the endpoint brain serves
//!   that model under, both sides in one session on one serving build, the
//!   jobs in bench's private directory.
//! * Each pair of jobs becomes one `bench verdict` that names the pair by
//!   its candidate files, at the gate's level, the retention suite as its
//!   anchor; no signing key is ever passed (bench signs with its own).
//! * Splinter reads none of what bench keeps (jobs, items): only the signed
//!   aggregates come back, and the gate decides on them with the public key.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_eval::promotion::{decide, Submitted};
use splinter_orchestrator::context::ServedEndpoint;
use splinter_pipelines::campaign::examiner::{AgentSpec, BenchExaminer, ExamineSpec, Protocol};
use splinter_pipelines::campaign::{ExamRequest, Examined, Model};

use super::world;

const CANDIDATE: &str = "sha256:c1";

/// A bench checkout whose `./bench` records every call. `run` makes one job
/// whose item record no one may read; `verdict` answers with the signed
/// verdict canned for its `--out` name.
fn fake_bench(root: &Path, canned: &Path) {
    let script = format!(
        r#"#!/bin/sh
printf '%s\n' "$*" >> "{root}/calls.log"
case "$1" in
run)
  while [ $# -gt 0 ]; do [ "$1" = --jobs-dir ] && jobs="$2"; [ "$1" = --profile ] && profile="$2"; shift; done
  mkdir -p "$jobs/2026-11-18__00-00-00"
  echo '{{}}' > "$jobs/2026-11-18__00-00-00/bench-run.json"
  echo secret > "$jobs/2026-11-18__00-00-00/verdict-items.json"
  chmod 000 "$jobs/2026-11-18__00-00-00/verdict-items.json"
  cp "$profile" "{root}/profile-$(basename "$(dirname "$jobs")")-$(basename "$jobs").toml"
  ;;
verdict)
  while [ $# -gt 0 ]; do [ "$1" = --out ] && out="$2"; shift; done
  cp "{canned}/$(basename "$out")" "$out"
  ;;
*) exit 2 ;;
esac
"#,
        root = root.display(),
        canned = canned.display()
    );
    std::fs::write(root.join("bench"), script).unwrap();
    std::fs::set_permissions(root.join("bench"), std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// An agent-protocol verdict of the candidate against the base, with the
/// given overall interval and retention bound; sealed when it spends `exam`.
fn verdict(interval: [f64; 2], retention_lower: f64, exam: Option<&str>) -> Value {
    let section = |items: u64, low: f64, high: f64, lower: f64| {
        json!({"items": items, "families": 12, "delta": (low + high) / 2.0, "interval": [low, high],
               "one_sided_lower": lower})
    };
    let sealed: Vec<Value> = exam
        .map(|e| json!({"collection": "slice-1", "exam_sha256": e, "tasks": 40}))
        .into_iter()
        .collect();
    let burns: Vec<Value> = exam
        .map(|e| json!({"exam_sha256": e, "candidate": CANDIDATE, "predecessor": null}))
        .into_iter()
        .collect();
    world::sign(
        &ed25519_dalek::SigningKey::from_bytes(&[42; 32]),
        json!({
            "schema": "bench.verdict/3", "verdict": "Inconclusive", "role": "candidate", "reasons": [],
            "integrity": {"ok": true, "checks": []}, "policy": {"alpha": 0.05},
            "candidate": {"adapter_digest": CANDIDATE}, "predecessor": {"kind": "baseline", "adapter_digest": null},
            "overall": section(80, interval[0], interval[1], interval[0] + 0.01),
            "retention": section(60, retention_lower - 0.02, 0.03, retention_lower),
            "sealed_exams": sealed, "burns": burns,
        }),
    )
}

fn endpoint(model: &str) -> ServedEndpoint {
    ServedEndpoint {
        base_url: format!("http://127.0.0.1:9/{model}/v1"),
        key: "served-key".into(),
        model: model.into(),
        identity: model.into(),
        adapter: None,
    }
}

#[test]
fn the_evaluate_stage_runs_each_collection_through_bench_for_both_models_and_gates_on_the_signed_aggregates(
) {
    let dir = tempfile::tempdir().unwrap();
    let (bench, canned, stage, private) = (
        dir.path().join("bench"),
        dir.path().join("canned"),
        dir.path().join("stage"),
        dir.path().join("private"),
    );
    for d in [&bench, &canned, &stage] {
        std::fs::create_dir_all(d).unwrap();
    }
    fake_bench(&bench, &canned);
    let exam = "e".repeat(64);
    for (name, doc) in [
        ("dev.json", verdict([0.02, 0.2], 0.0, None)),
        ("retention.json", verdict([-0.02, 0.03], -0.01, None)),
        ("sealed.json", verdict([0.05, 0.3], 0.0, Some(&exam))),
    ] {
        std::fs::write(canned.join(name), doc.to_string()).unwrap();
    }
    let public = dir.path().join("verdict.key.pub");
    let hex: String = world::key_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    std::fs::write(&public, hex).unwrap();
    let profile = dir.path().join("profile.toml");
    std::fs::write(
        &profile,
        "[roles]\nmain = \"brain\"\n[upstreams.brain]\nbase_url = \"http://x/v1\"\nmodel = \"m\"\n[run]\nattempts = 2\n",
    )
    .unwrap();
    let examiner = BenchExaminer::new(ExamineSpec {
        bench: bench.clone(),
        public_key: public,
        protocol: Protocol::Agent,
        private: private.clone(),
        dev: "tasks/support".into(),
        retention: "tasks/anchor".into(),
        slices: vec![".bench-data/exams/slice-1".into()],
        base: "Qwen/Qwen3-4B".into(),
        proposer_commit: "p".into(),
        brain_commit: "b".into(),
        serving: Vec::new(),
        agent: Some(AgentSpec {
            profile,
            min_gain: 0.05,
            serving_build: "brain-e9044bbeb".into(),
            dev_set: Some("eval".into()),
            sven_build: None,
        }),
    })
    .unwrap();
    let champion = Model {
        generation: 0,
        model: "local:/m/base".into(),
        candidate: None,
        adapter: None,
    };
    let candidate = Model {
        generation: 1,
        model: "local:/m/base+/a/g1.safetensors".into(),
        candidate: Some("g1".into()),
        adapter: Some(CANDIDATE.into()),
    };
    let training = dir.path().join("train.jsonl");
    std::fs::write(&training, "{}\n").unwrap();
    let request = ExamRequest {
        campaign: "camp",
        generation: 1,
        candidate: &candidate,
        predecessor: &champion,
        slice: ".bench-data/exams/slice-1",
        training: &training,
        dir: &stage,
    };
    let examined = Examined {
        dev: stage.join("dev.json"),
        retention: stage.join("retention.json"),
        sealed: stage.join("sealed.json"),
    };
    examiner
        .examine_agentic(
            &request,
            (&endpoint("local/base"), &endpoint("local/g1")),
            &examined,
            &CancelToken::new(),
        )
        .unwrap();

    let calls = std::fs::read_to_string(bench.join("calls.log")).unwrap();
    let calls: Vec<&str> = calls.lines().collect();
    assert_eq!(
        calls.len(),
        9,
        "two runs and one verdict per collection: {calls:#?}"
    );
    let jobs = private.join("jobs/camp/g0001");
    assert!(calls[0].starts_with("run tasks/support --profile") && calls[0].contains("--set eval"));
    assert!(calls[0].ends_with(&format!(
        "--jobs-dir {}",
        jobs.join("dev/champion").display()
    )));
    assert!(calls[3].starts_with("run tasks/anchor") && !calls[3].contains("--set"));
    assert!(calls[2].starts_with("verdict --base") && calls[2].contains("--anchor none"));
    assert!(calls[5].contains("--anchor anchor"), "{}", calls[5]);
    assert!(calls[8].contains(&format!(
        "--candidate-file {}",
        stage.join("candidate.json").display()
    )));
    assert!(
        !calls[8].contains("--predecessor-file"),
        "the base is the baseline"
    );
    assert!(
        calls.iter().all(|c| !c.contains("key")),
        "no key ever reaches bench's command line"
    );
    let profiles: Vec<toml::Table> = ["champion", "candidate"]
        .iter()
        .map(|side| {
            toml::from_str(
                &std::fs::read_to_string(bench.join(format!("profile-sealed-{side}.toml")))
                    .unwrap(),
            )
            .unwrap()
        })
        .collect();
    assert_eq!(
        profiles[0]["upstreams"]["brain"]["model"].as_str(),
        Some("local/base")
    );
    assert_eq!(
        profiles[1]["upstreams"]["brain"]["model"].as_str(),
        Some("local/g1")
    );
    for p in &profiles {
        assert_eq!(p["run"]["session"].as_str(), Some("camp-g0001"));
        assert_eq!(p["run"]["serving_build"].as_str(), Some("brain-e9044bbeb"));
    }
    let candidate_file: Value =
        serde_json::from_str(&std::fs::read_to_string(stage.join("candidate.json")).unwrap())
            .unwrap();
    assert_eq!(candidate_file["adapter_digest"], CANDIDATE);

    let read =
        |p: &Path| -> Value { serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap() };
    let (dev, retention, sealed) = (
        read(&examined.dev),
        read(&examined.retention),
        read(&examined.sealed),
    );
    let promotion = decide(
        &world::key(),
        &candidate.against(&champion),
        Submitted {
            dev: &dev,
            retention: &retention,
            sealed: &sealed,
        },
    );
    assert!(promotion.promoted, "{:?}", promotion.reasons);
    assert!(promotion.exam_verified);
}
