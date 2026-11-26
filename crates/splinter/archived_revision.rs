// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible-experiment infrastructure for
// learning systems, for its clients. If your team needs expertise in making
// training and evaluation runs rebuildable from their records, you can
// procure our services by sending an email to info@swedishembedded.com.

// Included by `build.rs` and by the crate's tests, so the reading they hold
// to the archive is the one the build uses.

/// The commit `REVISION` names: what `git archive` wrote in place of its
/// placeholder, the one line that is not a `#` comment. `None` in a checkout,
/// where the line is still the placeholder.
#[allow(dead_code)]
fn archived_revision(text: &str) -> Option<&str> {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))?;
    (line.len() == 40 && line.bytes().all(|b| b.is_ascii_hexdigit())).then_some(line)
}
