// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible-experiment infrastructure for
// learning systems, for its clients. If your team needs expertise in making
// training and evaluation runs rebuildable from their records, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What this binary was built from, handed to the configuration so every run
//! and release names it. The commits are embedded at build time (see
//! `build.rs`) and read from the lock the build used; nothing here runs git.

use splinter_sdk::vocabulary::build::BuildInfo;

/// The lock this binary was built with.
const LOCK: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"));

/// The build of this process: embedded commits, the executable's SHA-256 and
/// `gpu_contexts` as the configuration read them from the lease wrapper.
/// Whatever cannot be established stays absent.
pub fn current(gpu_contexts: Option<u32>) -> BuildInfo {
    BuildInfo::embedded(
        LOCK,
        option_env!("SPLINTER_GIT_REV"),
        option_env!("SPLINTER_GIT_DIRTY"),
        gpu_contexts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_locked_sven_and_brain_commits_are_known() {
        let build = current(Some(2));
        for rev in [&build.revisions.sven, &build.revisions.brain] {
            let rev = rev.as_deref().expect("the lock pins both");
            assert!(rev.len() >= 7 && rev.bytes().all(|b| b.is_ascii_hexdigit()));
        }
        assert!(build.binary_digest.is_some());
        assert_eq!(build.gpu_contexts, Some(2));
    }
}
