// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for importing sensitive
// longitudinal datasets as immutable, traceable episodes, for its clients. If
// your team needs expertise in privacy-preserving data pipelines you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a longitudinal file is imported as one immutable content-addressed
//! episode per participant, on explicit clocks, every item tied to its file
//! line and terms, with no raw identifier kept. Importing is deterministic and
//! idempotent, and a file that cannot be imported whole imports nothing.
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use splinter_core::longitudinal::{Assignment, History, ParticipantKeying, Record, TICKS_PER_UNIT};
use splinter_core::terms::UsagePolicy;
use splinter_expdb::model::EpisodeKind;
use splinter_expdb::{Config, Database};
use splinter_store::longitudinal::{calendar_clock, ImportSpec, LongitudinalStore};
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

const ALICE: &str = r#"{"subject_id":"alice-17","group_id":"home-4","weight":2.5,"source":"cycle-a","entry":50.0,"calendar_at_entry":2003.5,"observations":[{"t":50.0,"var":"sbp","value":131},{"t":50.0,"var":"crp","value":{"below":0.2}},{"t":25.0,"var":"weight","value":70}],"events":[{"t":44.0,"code":"dx:hypertension"},{"t":62.5,"code":"death:heart"}],"at_risk":[{"code":"*","from":50.0,"to":62.5}],"interventions":[{"t":50.0,"code":"statin","randomised":true},{"t":49.0,"code":"smoking","randomised":false}]}"#;
const BOB: &str = r#"{"subject_id":"bob-3","weight":1.0,"source":"cycle-b","entry":60.0,"calendar_at_entry":2010.0,"observations":[{"t":60.0,"var":"sbp","value":120}],"at_risk":[{"code":"*","from":60.0,"to":70.0}]}"#;
const CARA: &str = r#"{"subject_id":"cara-9","group_id":"home-4","source":"cycle-a","entry":41.5,"calendar_at_entry":2001.0}"#;

fn root(name: &str) -> StateRoot {
    let path = std::env::temp_dir().join(format!(
        "splinter-longitudinal-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    StateRoot::new(path)
}

fn write(root: &StateRoot, name: &str, lines: &[&str]) -> PathBuf {
    std::fs::create_dir_all(root.path()).unwrap();
    let path = root.path().join(name);
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    path
}

fn spec() -> ImportSpec {
    ImportSpec {
        dataset: "cohort".into(),
        terms: UsagePolicy::ResearchOnly.terms("cohort data use agreement"),
        keying: ParticipantKeying::from_secret(b"campaign secret"),
    }
}

fn expected(
    line: &str,
    number: u64,
    file: &splinter_core::digest::Digest,
    spec: &ImportSpec,
) -> History {
    History::from_record(
        Record::from_json_line(line).unwrap(),
        &spec.keying,
        &spec.dataset,
        file,
        number,
        &spec.terms,
    )
    .unwrap()
}

#[test]
fn every_participant_becomes_one_episode_and_reads_back_whole() {
    let root = root("roundtrip");
    let file = write(&root, "cohort.jsonl", &[ALICE, "", BOB, CARA]);
    let store = LongitudinalStore::new(&Workspace::at(&root));
    let spec = spec();
    let report = store.import_jsonl(&file, &spec).unwrap();
    assert_eq!((report.rows, report.participants), (3, 3));
    assert_eq!((report.added, report.already_present), (3, 0));

    let mut found = Vec::new();
    store
        .for_each_history(|address, history| {
            found.push((address.clone(), history));
            Ok(())
        })
        .unwrap();
    assert_eq!(found.len(), 3);
    // Blank lines count toward line numbers: provenance points at the real line.
    for (line, number) in [(ALICE, 1), (BOB, 3), (CARA, 4)] {
        let want = expected(line, number, &report.file, &spec);
        let got = found
            .iter()
            .find(|(_, h)| h.participant == want.participant)
            .unwrap();
        assert_eq!(got.1, want);
        assert_eq!(got.0, want.address().unwrap());
    }
}

#[test]
fn absent_measurements_stay_absent() {
    let root = root("absent");
    let file = write(&root, "cohort.jsonl", &[BOB]);
    let store = LongitudinalStore::new(&Workspace::at(&root));
    store.import_jsonl(&file, &spec()).unwrap();
    store
        .for_each_history(|_, h| {
            assert_eq!(h.observations.len(), 1);
            assert!(h.events.is_empty() && h.interventions.is_empty());
            Ok(())
        })
        .unwrap();
}

#[test]
fn reimporting_the_same_file_adds_nothing_and_yields_the_same_episodes() {
    let root = root("idempotent");
    let file = write(&root, "cohort.jsonl", &[ALICE, BOB]);
    let store = LongitudinalStore::new(&Workspace::at(&root));
    let first = store.import_jsonl(&file, &spec()).unwrap();
    let before = store.addresses().unwrap();
    let again = store.import_jsonl(&file, &spec()).unwrap();
    assert_eq!((again.added, again.already_present), (0, 2));
    assert_eq!(again.file, first.file);
    assert_eq!(store.addresses().unwrap(), before);

    // A second process writing the same file into a fresh state gets the same ids.
    let other = root_for_other(&file);
    assert_eq!(other, before);
}

fn root_for_other(file: &Path) -> Vec<splinter_core::digest::Digest> {
    let root = root("idempotent-other");
    let store = LongitudinalStore::new(&Workspace::at(&root));
    store.import_jsonl(file, &spec()).unwrap();
    store.addresses().unwrap()
}

#[test]
fn a_file_with_a_bad_row_or_a_repeated_participant_imports_nothing() {
    let root = root("refused");
    let store = LongitudinalStore::new(&Workspace::at(&root));
    let bad = write(
        &root,
        "bad.jsonl",
        &[
            ALICE,
            r#"{"subject_id":"x","source":"s","entry":1.0,"calendar_at_entry":2000.0,"interventions":[{"t":1.0,"code":"c"}]}"#,
        ],
    );
    let error = store.import_jsonl(&bad, &spec()).unwrap_err().to_string();
    assert!(
        error.contains("line 2") && error.contains("randomised"),
        "{error}"
    );
    let repeated = write(&root, "twice.jsonl", &[ALICE, BOB, ALICE]);
    let error = store
        .import_jsonl(&repeated, &spec())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("line 3") && error.contains("line 1"),
        "{error}"
    );
    assert!(
        !error.contains("alice-17"),
        "an error never names a participant: {error}"
    );
    assert!(store.addresses().unwrap().is_empty(), "nothing was written");
}

#[test]
fn no_raw_identifier_is_stored_and_keys_follow_the_secret() {
    let root = root("identity");
    let file = write(&root, "cohort.jsonl", &[ALICE, BOB, CARA]);
    let workspace = Workspace::at(&root);
    let store = LongitudinalStore::new(&workspace);
    store.import_jsonl(&file, &spec()).unwrap();
    workspace.commit().unwrap();
    // The database compresses what it writes, so look at what it hands back:
    // every episode, with every stream chunk, as a reader sees it.
    let snapshot = Database::open(root.expdb(), Config::default())
        .unwrap()
        .snapshot()
        .unwrap();
    let mut dump = String::new();
    for kind in [EpisodeKind::Observational, EpisodeKind::Experimental] {
        for id in snapshot.episodes_of_kind(kind).unwrap() {
            let view = snapshot.episode(id).unwrap().unwrap();
            dump.push_str(&format!("{view:?}"));
            for stream in &view.streams {
                for chunk in &stream.chunks {
                    dump.push_str(&String::from_utf8(snapshot.read_chunk(chunk).unwrap()).unwrap());
                }
            }
        }
    }
    assert!(
        dump.contains("cycle-a") && dump.contains("death:heart"),
        "the dump holds the episodes"
    );
    for raw in ["alice-17", "bob-3", "cara-9", "home-4"] {
        assert!(
            !dump.contains(raw),
            "an episode holds the raw identifier {raw}"
        );
    }
    let mut groups = Vec::new();
    let mut participants = Vec::new();
    store
        .for_each_history(|_, h| {
            groups.push((h.source.clone(), h.group.clone()));
            participants.push(h.participant.clone());
            Ok(())
        })
        .unwrap();
    participants.sort();
    participants.dedup();
    assert_eq!(participants.len(), 3);
    let cycle_a: Vec<_> = groups.iter().filter(|(s, _)| s == "cycle-a").collect();
    assert_eq!(
        cycle_a[0].1, cycle_a[1].1,
        "alice and cara share a household"
    );
    let bob = groups.iter().find(|(s, _)| s == "cycle-b").unwrap();
    assert_ne!(bob.1, cycle_a[0].1);

    let other = root_with_secret("identity-other", &file, b"another secret");
    assert!(
        other.iter().all(|k| !participants.contains(k)),
        "another secret, other keys"
    );
}

fn root_with_secret(
    name: &str,
    file: &Path,
    secret: &[u8],
) -> Vec<splinter_core::longitudinal::ParticipantKey> {
    let root = root(name);
    let store = LongitudinalStore::new(&Workspace::at(&root));
    let mut spec = spec();
    spec.keying = ParticipantKeying::from_secret(secret);
    store.import_jsonl(file, &spec).unwrap();
    let mut keys = Vec::new();
    store
        .for_each_history(|_, h| {
            keys.push(h.participant);
            Ok(())
        })
        .unwrap();
    keys
}

#[test]
fn episodes_sit_on_the_participants_clock_and_map_to_the_calendar() {
    let root = root("clocks");
    let file = write(&root, "cohort.jsonl", &[ALICE, BOB]);
    let workspace = Workspace::at(&root);
    let store = LongitudinalStore::new(&workspace);
    store.import_jsonl(&file, &spec()).unwrap();
    workspace.commit().unwrap();

    let snapshot = Database::open(root.expdb(), Config::default())
        .unwrap()
        .snapshot()
        .unwrap();
    let randomised = snapshot
        .episodes_of_kind(EpisodeKind::Experimental)
        .unwrap();
    let observed = snapshot
        .episodes_of_kind(EpisodeKind::Observational)
        .unwrap();
    assert_eq!(
        (randomised.len(), observed.len()),
        (1, 1),
        "a randomised assignment makes an experimental episode"
    );

    let calendar = calendar_clock().id().unwrap();
    let alice = snapshot.episode(randomised[0]).unwrap().unwrap();
    assert_ne!(
        alice.episode.clock, calendar,
        "the participant's clock is its own"
    );
    let to_calendar = snapshot
        .resolve_clock(alice.episode.clock, calendar)
        .unwrap()
        .unwrap();
    let entry_ticks = (50.0 * TICKS_PER_UNIT) as i64;
    assert_eq!(
        to_calendar.apply(entry_ticks),
        (2003.5 * TICKS_PER_UNIT) as i64
    );
    assert!(alice
        .streams
        .iter()
        .all(|s| s.stream.clock == alice.episode.clock));
    assert_eq!(alice.streams.len(), 3, "one stream per measured variable");
    assert_eq!(alice.actions.len(), 2, "interventions are actions");
    let flags: Vec<_> = alice
        .actions
        .iter()
        .map(|a| match &a.action.payload {
            splinter_expdb::model::Content::Text { text } => text.clone(),
            other => panic!("{other:?}"),
        })
        .collect();
    assert!(
        flags.iter().any(|f| f.contains("\"randomised\""))
            && flags.iter().any(|f| f.contains("\"observational\""))
    );
    for text in &flags {
        assert!(
            text.contains("blake3:") && text.contains("\"line\":1"),
            "provenance on every action: {text}"
        );
    }
    assert!(alice
        .events
        .iter()
        .all(|e| e.event.payload.to_string().contains("\"terms\"")));
    let _ = Assignment::Randomised;
}

#[test]
fn a_larger_file_imports_in_groups_and_again_adds_nothing() {
    let root = root("large");
    let lines: Vec<String> = (0..700)
        .map(|n| {
            format!(
                r#"{{"subject_id":"p{n}","source":"cycle-a","entry":{}.5,"calendar_at_entry":2000.0,"observations":[{{"t":{}.5,"var":"sbp","value":{}}}]}}"#,
                30 + n % 40,
                30 + n % 40,
                100 + n % 50
            )
        })
        .collect();
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let file = write(&root, "large.jsonl", &refs);
    let store = LongitudinalStore::new(&Workspace::at(&root));
    let first = store.import_jsonl(&file, &spec()).unwrap();
    assert_eq!((first.rows, first.added), (700, 700));
    assert_eq!(store.addresses().unwrap().len(), 700);
    let again = store.import_jsonl(&file, &spec()).unwrap();
    assert_eq!((again.added, again.already_present), (0, 700));
    let mut seen = 0;
    store
        .for_each_history(|_, _| {
            seen += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, 700);
}
