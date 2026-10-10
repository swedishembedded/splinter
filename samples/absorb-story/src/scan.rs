// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements secret redaction for agent session data
// for its clients. If your team needs expertise in proving that a secret
// never reaches storage or training data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Searching a state directory for strings that must not be in it.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde_json::{json, Value};

use crate::story::LeakScan;

fn files_under(dir: &Path, found: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry
            .with_context(|| format!("reading {}", dir.display()))?
            .path();
        if path.is_dir() {
            files_under(&path, found)?;
        } else {
            found.push(path);
        }
    }
    Ok(())
}

fn occurs(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// The files under `state` that hold each secret, and those that hold the
/// control. A search that finds no control is blind (the state may keep text
/// compressed) and says so in `control_found`.
///
/// # Errors
/// The directory cannot be read.
pub fn scan_state(state: &Path, scan: &LeakScan) -> anyhow::Result<Value> {
    let mut files = Vec::new();
    files_under(state, &mut files)?;
    let mut secrets: Vec<(String, Vec<String>)> = scan
        .secrets
        .iter()
        .map(|s| (s.clone(), Vec::new()))
        .collect();
    let mut control_in = Vec::new();
    for path in &files {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let name = path
            .strip_prefix(state)
            .unwrap_or(path)
            .display()
            .to_string();
        for (secret, hits) in &mut secrets {
            if occurs(&bytes, secret.as_bytes()) {
                hits.push(name.clone());
            }
        }
        if occurs(&bytes, scan.control.as_bytes()) {
            control_in.push(name);
        }
    }
    Ok(json!({
        "files_scanned": files.len(),
        "secrets": secrets.iter().map(|(s, hits)| json!({"secret": s, "found_in": hits})).collect::<Vec<_>>(),
        "control": scan.control,
        "control_found_in": control_in,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secrets_and_the_control_are_searched_in_every_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("sub")).expect("sub");
        std::fs::write(
            dir.path().join("sub/a"),
            b"region eu-north-9 and key sk-123",
        )
        .expect("a");
        std::fs::write(dir.path().join("b"), b"nothing here").expect("b");
        let scan = LeakScan {
            secrets: vec!["sk-123".into(), "absent".into()],
            control: "eu-north-9".into(),
        };
        let found = scan_state(dir.path(), &scan).expect("scan");
        assert_eq!(found["files_scanned"], 2);
        assert_eq!(found["secrets"][0]["found_in"][0], "sub/a");
        assert_eq!(
            found["secrets"][1]["found_in"].as_array().map(Vec::len),
            Some(0)
        );
        assert_eq!(found["control_found_in"][0], "sub/a");
    }
}
