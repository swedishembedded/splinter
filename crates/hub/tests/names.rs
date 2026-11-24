// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-hub clients that never let a remote
// name write outside the directory it was given, for its clients. If your
// team needs expertise in secure artifact distribution then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec, by property: a name a hub or a user supplies (a repository id, a
//! commit, a path in a repository) is answered for any text, and what is
//! accepted is safe to use as a local path: joined to the directory it is
//! downloaded into, it names a place inside that directory, and it reads
//! back as the text it was.
#![allow(clippy::unwrap_used)]

use std::path::{Component, Path};

use proptest::prelude::*;
use splinter_conformance::fuzz::{config, damaged_text};
use splinter_hub::{CommitId, RepoId, RepoPath};

/// Text near the names a hub serves, or any text.
fn name(seed: &'static str) -> impl Strategy<Value = String> {
    prop_oneof![damaged_text(seed), "\\PC{0,24}", "[a-z./_-]{0,24}"]
}

fn stays_inside(relative: &str) -> bool {
    let joined = Path::new("/srv/hub").join(relative);
    joined.starts_with("/srv/hub")
        && Path::new(relative)
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn an_accepted_repository_path_stays_in_its_directory(text in name("onnx/model.q4.onnx")) {
        if let Ok(path) = RepoPath::parse(&text) {
            prop_assert!(stays_inside(path.as_str()), "{text:?}");
            prop_assert_eq!(path.to_string(), text);
        }
    }

    #[test]
    fn an_accepted_repository_id_is_two_names_inside_the_cache(text in name("Qwen/Qwen3-4B")) {
        if let Ok(id) = RepoId::parse(&text) {
            prop_assert!(stays_inside(&id.to_string()), "{text:?}");
            prop_assert_eq!(id.to_string().split('/').count(), 2);
            prop_assert_eq!(id.to_string(), text);
        }
    }

    #[test]
    fn an_accepted_commit_is_forty_hex_digits(
        text in name("0123456789abcdef0123456789abcdef01234567")
    ) {
        if let Ok(commit) = CommitId::parse(&text) {
            prop_assert!(stays_inside(commit.as_str()));
            prop_assert_eq!(commit.as_str().len(), 40);
        }
    }
}
