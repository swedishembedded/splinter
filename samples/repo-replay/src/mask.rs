// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The diff as the replaying agent sees it: positions removed, content kept
//! where content is the only way to find a place.
//!
//! Hunk headers become `@@ @@` (no offsets, no function heading), the `index`
//! lines (which name the target's blobs) are dropped, and unchanged lines are
//! shown as `xxx` - except the few that anchor a change. A hunk with removed
//! lines is located by searching for them; an insertion has no removed lines,
//! so the unchanged lines next to it are kept as they are. The masker widens
//! the anchor, line by line on both sides, until the lines it shows together
//! occur exactly once in the parent's file; if even the whole hunk's unchanged
//! lines do not make the place unique, the commit cannot be replayed from its
//! masked diff and is refused ([`MaskError`], counted as `ambiguous_hunk`).
//! Offsets are never shown, so finding the place still takes a search and a
//! read.

/// A diff that went through [`mask_diff`]: the only form the grounded harness document
/// accepts, so no caller can hand the agent offsets or a bare unmasked diff.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaskedDiff(String);

impl MaskedDiff {
    /// The masked text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Why a diff cannot be masked so that every place is determinable.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("a hunk of {path} cannot be located from its masked form: its lines occur more than once in the file")]
pub struct MaskError {
    /// The file of the hunk.
    pub path: String,
}

struct Hunk<'a> {
    lines: Vec<&'a str>,
}

/// How many times `needle` occurs as a run of consecutive lines of `file`.
fn occurrences(file: &[&str], needle: &[&str]) -> usize {
    if needle.is_empty() || needle.len() > file.len() {
        return 0;
    }
    file.windows(needle.len()).filter(|w| *w == needle).count()
}

/// For one hunk, which of its lines to show: `shown[i]` for each line of `lines`.
/// Removed and added lines are always shown; unchanged lines next to the change
/// block are added, one at a time, until the shown old-side lines are unique.
fn anchored(hunk: &Hunk<'_>, parent: Option<&[&str]>) -> Option<Vec<bool>> {
    let lines = &hunk.lines;
    let mut shown: Vec<bool> = lines.iter().map(|l| !l.starts_with(' ')).collect();
    let Some(file) = parent else {
        return Some(shown);
    };
    let first = lines.iter().position(|l| !l.starts_with(' '))?;
    let last = lines.iter().rposition(|l| !l.starts_with(' '))?;
    // Unchanged lines before the first change and after the last, nearest first.
    let before: Vec<usize> = (0..first).rev().collect();
    let after: Vec<usize> = (last + 1..lines.len()).collect();
    let (mut k, mut m) = (0usize, 0usize);
    loop {
        // The old-side lines shown, as one contiguous run: the nearest `k`
        // unchanged lines before the change block, the block's own removed
        // lines and unchanged lines inside it, and the nearest `m` after.
        let from = if k == 0 { first } else { before[k - 1] };
        let to = if m == 0 { last } else { after[m - 1] };
        let run: Vec<&str> = lines[from..=to]
            .iter()
            .filter(|l| l.starts_with(' ') || l.starts_with('-'))
            .map(|l| &l[1..])
            .collect();
        // A pure insertion has no old line of its own: the run is its anchors.
        let removed = lines[first..=last].iter().any(|l| l.starts_with('-'));
        let determinable = occurrences(file, &run) == 1 && (removed || !run.is_empty());
        if determinable {
            shown[from..=to].fill(true);
            return Some(shown);
        }
        // Widen: alternate sides, nearest first.
        if k <= m && k < before.len() {
            k += 1;
        } else if m < after.len() {
            m += 1;
        } else if k < before.len() {
            k += 1;
        } else {
            return None;
        }
    }
}

/// `diff` masked; `parent` gives the parent's content of a path (the old
/// path of a rename), `None` for a file it does not have or cannot read.
/// Without parent content the anchors cannot be checked and only removed and
/// added lines are shown.
pub fn mask_diff(
    diff: &str,
    parent: &dyn Fn(&str) -> Option<String>,
) -> Result<MaskedDiff, MaskError> {
    let mut out = String::new();
    let mut lines = diff.lines().peekable();
    let mut new_path = String::new();
    let mut content: Option<String> = None;
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("diff --git a/") {
            let (o, n) = rest.split_once(" b/").unwrap_or((rest, rest));
            new_path = n.to_string();
            content = parent(o);
            out.push_str(line);
            out.push('\n');
        } else if line.starts_with("@@") {
            let mut hunk = Vec::new();
            while let Some(next) = lines.peek() {
                if next.starts_with("diff --git ") || next.starts_with("@@") {
                    break;
                }
                hunk.push(lines.next().unwrap_or_default());
            }
            let file_lines: Option<Vec<&str>> = content.as_deref().map(|c| c.lines().collect());
            let body: Vec<&str> = hunk
                .iter()
                .copied()
                .filter(|l| !l.starts_with('\\'))
                .collect();
            let shown = anchored(
                &Hunk {
                    lines: body.clone(),
                },
                file_lines.as_deref(),
            )
            .ok_or_else(|| MaskError {
                path: new_path.clone(),
            })?;
            out.push_str("@@ @@\n");
            let mut at = 0;
            for l in &hunk {
                if l.starts_with('\\') {
                    out.push_str(l);
                } else {
                    if shown[at] {
                        out.push_str(l);
                    } else {
                        out.push_str(" xxx");
                    }
                    at += 1;
                }
                out.push('\n');
            }
        } else if line.starts_with("index ") {
            continue;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    Ok(MaskedDiff(out))
}
