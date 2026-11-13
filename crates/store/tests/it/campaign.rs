// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-running model-improvement
// campaigns that survive a crash and keep an honest account of what each
// experiment cost and found, for its clients. If your team needs expertise in
// resumable training orchestration or experiment ledgers you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a campaign's state is resumable and cannot be advanced by two
//! controllers at once; its ledger keeps every arm's hypothesis, cost and
//! verdict, with what was not measured absent.
#![allow(clippy::unwrap_used)]

use serde::{Deserialize, Serialize};
use splinter_core::digest::digest_of;
use splinter_store::campaign::{CampaignError, CampaignStore, Cost, LedgerEntry, Verdict};
use splinter_store::StateRoot;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct State {
    step: u32,
    adapter: String,
}

fn workspace() -> (StateRoot, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    (StateRoot::new(dir.path()), dir)
}

fn entry(campaign: &str, arm: &str, verdict: Verdict) -> LedgerEntry {
    LedgerEntry {
        campaign: campaign.into(),
        arm: arm.into(),
        generation: 1,
        strategy: "policy-gradient".into(),
        hypothesis: "dynamic sampling raises the informative fraction".into(),
        config_digest: digest_of(&"config"),
        at: "2026-11-13T08:00:00.000Z".into(),
        cost: Cost::default(),
        discoveries: None,
        verdict,
        measurements: Default::default(),
    }
}

#[test]
fn a_campaign_never_saved_has_no_state() {
    let (ws, _dir) = workspace();
    let store = CampaignStore::new(&ws, "lua").unwrap();
    assert!(store.load::<State>().unwrap().is_none());
}

#[test]
fn a_saved_state_is_what_a_restarted_controller_loads() {
    let (ws, _dir) = workspace();
    let store = CampaignStore::new(&ws, "lua").unwrap();
    let first = State {
        step: 1,
        adapter: "a1".into(),
    };
    let digest = store.save(None, &first, "t1").unwrap();
    let second = State {
        step: 2,
        adapter: "a2".into(),
    };
    store.save(Some(&digest), &second, "t2").unwrap();

    let loaded = CampaignStore::new(&ws, "lua")
        .unwrap()
        .load::<State>()
        .unwrap()
        .unwrap();
    assert_eq!(loaded.state, second);
    assert_eq!(loaded.version, 2);
}

#[test]
fn a_second_controller_that_loaded_the_same_state_loses_instead_of_overwriting() {
    let (ws, _dir) = workspace();
    let store = CampaignStore::new(&ws, "lua").unwrap();
    let start = store
        .save(
            None,
            &State {
                step: 0,
                adapter: "base".into(),
            },
            "t0",
        )
        .unwrap();
    let winner = State {
        step: 1,
        adapter: "winner".into(),
    };
    store.save(Some(&start), &winner, "t1").unwrap();
    let late = store.save(
        Some(&start),
        &State {
            step: 1,
            adapter: "late".into(),
        },
        "t2",
    );
    assert!(
        matches!(late, Err(CampaignError::Conflict { .. })),
        "{late:?}"
    );
    assert_eq!(store.load::<State>().unwrap().unwrap().state, winner);
}

#[test]
fn campaigns_are_isolated_from_each_other() {
    let (ws, _dir) = workspace();
    let a = CampaignStore::new(&ws, "a").unwrap();
    let b = CampaignStore::new(&ws, "b").unwrap();
    a.save(
        None,
        &State {
            step: 9,
            adapter: "a".into(),
        },
        "t",
    )
    .unwrap();
    assert!(b.load::<State>().unwrap().is_none());
    a.record(&entry("a", "arm-1", Verdict::Running)).unwrap();
    assert!(b.ledger().unwrap().is_empty());
    assert!(
        b.record(&entry("a", "arm-1", Verdict::Running)).is_err(),
        "an entry names its own campaign"
    );
}

#[test]
fn names_that_could_escape_the_pointer_are_refused() {
    let (ws, _dir) = workspace();
    for bad in ["", ".", "..", "a/b", "a b", "a\n"] {
        assert!(
            matches!(CampaignStore::new(&ws, bad), Err(CampaignError::Name(_))),
            "{bad:?}"
        );
    }
}

#[test]
fn the_ledger_keeps_every_event_in_order_and_unmeasured_stays_absent() {
    let (ws, _dir) = workspace();
    let store = CampaignStore::new(&ws, "lua").unwrap();
    store
        .record(&entry("lua", "arm-1", Verdict::Running))
        .unwrap();
    let mut finished = entry("lua", "arm-1", Verdict::Accepted { significant: false });
    finished.cost = Cost {
        gpu_hours: Some(1.5),
        ..Cost::default()
    };
    finished.discoveries = Some(3.0);
    finished.at = "2026-11-13T09:00:00.000Z".into();
    store.record(&finished).unwrap();
    store
        .record(&entry(
            "lua",
            "arm-2",
            Verdict::Refused {
                reason: "the exam cannot detect the target".into(),
            },
        ))
        .unwrap();

    let ledger = store.ledger().unwrap();
    assert_eq!(ledger.len(), 3);
    assert_eq!(ledger[0].verdict, Verdict::Running);
    assert_eq!(ledger[0].cost.gpu_hours, None, "unmeasured is not zero");
    assert_eq!(ledger[0].discoveries, None);
    let latest = LedgerEntry::latest_per_arm(&ledger);
    assert_eq!(latest.len(), 2);
    assert_eq!(latest[0].cost.gpu_hours, Some(1.5));
    assert_eq!(latest[0].discoveries, Some(3.0));
    assert!(matches!(latest[1].verdict, Verdict::Refused { .. }));
    let text = serde_json::to_string(&ledger[0]).unwrap();
    assert!(!text.contains("gpu_hours"), "{text}");
}

fn files_under(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let path = e.unwrap().path();
            if path.is_dir() {
                files_under(&path)
            } else {
                1
            }
        })
        .sum()
}

#[test]
fn a_campaign_is_one_file_however_many_states_and_entries_it_holds() {
    let (ws, dir) = workspace();
    let store = CampaignStore::new(&ws, "lua").unwrap();
    let mut previous = None;
    for step in 0..25 {
        let state = State {
            step,
            adapter: format!("a{step}"),
        };
        previous = Some(store.save(previous.as_ref(), &state, "t").unwrap());
        store
            .record(&entry("lua", &format!("arm-{step}"), Verdict::Running))
            .unwrap();
    }
    assert_eq!(
        files_under(dir.path()),
        1,
        "one pack, no file per state or entry"
    );
    assert_eq!(store.ledger().unwrap().len(), 25);
    assert_eq!(store.load::<State>().unwrap().unwrap().version, 25);
}

#[test]
fn a_torn_last_line_is_cut_off_and_the_previous_state_is_still_the_head() {
    let (ws, dir) = workspace();
    let store = CampaignStore::new(&ws, "lua").unwrap();
    let first = State {
        step: 1,
        adapter: "a1".into(),
    };
    let digest = store.save(None, &first, "t").unwrap();
    let pack = dir.path().join("campaigns/lua.pack");
    let mut text = std::fs::read(&pack).unwrap();
    text.extend_from_slice(br#"{"record":"state","at":"t","state":{"step":2,"ada"#);
    std::fs::write(&pack, text).unwrap();
    assert_eq!(store.load::<State>().unwrap().unwrap().state, first);
    let second = State {
        step: 2,
        adapter: "a2".into(),
    };
    store.save(Some(&digest), &second, "t2").unwrap();
    assert_eq!(store.load::<State>().unwrap().unwrap().state, second);
}

#[test]
fn compaction_keeps_the_ledger_and_the_newest_states() {
    let (ws, dir) = workspace();
    let store = CampaignStore::new(&ws, "lua").unwrap();
    let mut previous = None;
    for step in 0..10 {
        let state = State {
            step,
            adapter: "x".repeat(200),
        };
        previous = Some(store.save(previous.as_ref(), &state, "t").unwrap());
        store
            .record(&entry("lua", &format!("arm-{step}"), Verdict::Running))
            .unwrap();
    }
    let before = std::fs::metadata(dir.path().join("campaigns/lua.pack"))
        .unwrap()
        .len();
    store.compact(2).unwrap();
    let after = std::fs::metadata(dir.path().join("campaigns/lua.pack"))
        .unwrap()
        .len();
    assert!(after < before);
    assert_eq!(store.ledger().unwrap().len(), 10);
    let loaded = store.load::<State>().unwrap().unwrap();
    assert_eq!(loaded.state.step, 9);
    assert_eq!(loaded.version, 10, "the version survives compaction");
    store
        .save(
            Some(&loaded.digest),
            &State {
                step: 10,
                adapter: "y".into(),
            },
            "t",
        )
        .unwrap();
    assert_eq!(files_under(dir.path()), 1);
}
