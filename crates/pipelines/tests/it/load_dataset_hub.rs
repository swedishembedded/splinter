// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that read any dataset
// the same way, for its clients. If your team needs expertise in loading,
// sampling and versioning datasets for agent training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a dataset of a Hub repository loads like a local one, against a
//! local emulator of the Hub.
//!
//! * `owner/name` loads the splits of the repository's `data/` directory;
//! * the dataset says the commit it was read at, also when a tag or a branch
//!   was asked for;
//! * a tag keeps naming the revision it named, after the branch moves on;
//! * a second load is served from the cache and downloads nothing.

#![allow(clippy::unwrap_used)]

use splinter_data::load::{load_dataset, ChatRecord, LoadOptions};
use splinter_hub::{Hub, RepoId, RepoPath, Upload};

use crate::common::hub::{Emulator, Quirks};
use crate::common::Scratch;

const REPO: &str = "lab/records";

fn record(task: &str) -> String {
    format!(
        r#"{{"messages":[{{"role":"user","content":"x","train":true}}],"metadata":{{"task":"{task}"}}}}"#
    )
}

fn commit(hub: &Hub, repo: &RepoId, scratch: &Scratch, tasks: &[(&str, &[&str])]) -> String {
    let mut adds = Vec::new();
    for (name, rows) in tasks {
        let local = scratch.0.join(format!("up-{name}.jsonl"));
        let text: String = rows.iter().map(|t| format!("{}\n", record(t))).collect();
        std::fs::write(&local, text).unwrap();
        adds.push(Upload {
            path: RepoPath::parse(&format!("data/{name}.jsonl")).unwrap(),
            local,
        });
    }
    let parent = hub.resolve(repo, "main").ok();
    hub.commit(repo, "main", parent.as_ref(), "records", &adds, &[])
        .unwrap()
        .commit
        .as_str()
        .to_owned()
}

#[test]
fn a_hub_dataset_loads_at_a_commit_and_is_cached() {
    let emulator = Emulator::start(Quirks::default());
    let hub = Hub::new(emulator.url(), Some("test-token".into())).unwrap();
    let repo = RepoId::parse(REPO).unwrap();
    hub.create_private_repo(&repo).unwrap();
    let scratch = Scratch::new("load-dataset-hub");
    let first = commit(
        &hub,
        &repo,
        &scratch,
        &[("train", &["a", "b"]), ("eval", &["c"])],
    );
    hub.create_tag(
        &repo,
        &splinter_hub::CommitId::parse(&first).unwrap(),
        "v1",
        "first",
    )
    .unwrap();
    // The branch moves on; the tag does not.
    let second = commit(
        &hub,
        &repo,
        &scratch,
        &[("train", &["a", "b", "d"]), ("eval", &["c"])],
    );
    assert_ne!(first, second);

    let cache = scratch.0.join("cache");
    let options = LoadOptions {
        hub: Some(&hub),
        cache: Some(&cache),
        ..LoadOptions::default()
    };

    let latest = load_dataset::<ChatRecord>(REPO, None, &options).unwrap();
    assert_eq!(latest.get("train").unwrap().len(), 3);
    assert_eq!(
        latest.get("eval").unwrap().len(),
        1,
        "eval is the validation split"
    );
    assert_eq!(
        latest
            .get("train")
            .unwrap()
            .provenance()
            .revision
            .as_deref(),
        Some(second.as_str()),
        "the commit, not the name `main`"
    );

    let tagged = load_dataset::<ChatRecord>(&format!("hf:{REPO}@v1"), None, &options).unwrap();
    assert_eq!(tagged.get("train").unwrap().len(), 2);
    assert_eq!(
        tagged
            .get("train")
            .unwrap()
            .provenance()
            .revision
            .as_deref(),
        Some(first.as_str())
    );

    // A second load reads the cache: no download is requested.
    let before = emulator
        .log()
        .iter()
        .filter(|l| l.contains("/resolve/"))
        .count();
    let again = load_dataset::<ChatRecord>(REPO, None, &options).unwrap();
    assert_eq!(again.get("train").unwrap().len(), 3);
    let after = emulator
        .log()
        .iter()
        .filter(|l| l.contains("/resolve/"))
        .count();
    assert_eq!(before, after, "cached files are not fetched again");
}
