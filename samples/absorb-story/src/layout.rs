// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable, content-addressed experiment
// runs for its clients. If your team needs expertise in running long
// measurement campaigns that survive interruption, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Where a storyline's files live, and the frozen definition that makes a
//! resumed run refuse a storyline that changed since it began.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _};
use serde_json::{json, Value};

use crate::story::Storyline;

/// The directories and files of one storyline under the output directory.
#[derive(Clone, Debug)]
pub struct Layout {
    root: PathBuf,
}

impl Layout {
    /// The layout of storyline `name` under `out`.
    #[must_use]
    pub fn new(out: &Path, name: &str) -> Self {
        Self {
            root: out.join(name),
        }
    }

    /// The storyline's directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The Splinter state root of the storyline.
    #[must_use]
    pub fn state(&self) -> PathBuf {
        self.root.join("state")
    }

    /// The directory the sessions of night `night` are written to.
    #[must_use]
    pub fn sessions(&self, night: u32) -> PathBuf {
        self.root.join("sessions").join(format!("night-{night}"))
    }

    /// The directory of run `run` (run 0 is the base before any night).
    #[must_use]
    pub fn run(&self, run: u32) -> PathBuf {
        self.root.join("runs").join(format!("run-{run}"))
    }

    /// The frozen definition.
    #[must_use]
    pub fn frozen(&self) -> PathBuf {
        self.root.join("storyline.json")
    }
}

/// The definition of `story` as JSON: items, sessions (with their content
/// hash), lessons, forgettings, the extra arguments and every prediction.
#[must_use]
pub fn definition(story: &Storyline) -> Value {
    let nights: Vec<Value> = story
        .nights
        .iter()
        .map(|night| {
            let sessions: Vec<Value> = night
                .sessions
                .iter()
                .map(|s| {
                    let text = s.script.to_json().to_string();
                    json!({
                        "id": s.script.id,
                        "content": blake3::hash(text.as_bytes()).to_hex().to_string(),
                        "lessons": s.lessons.iter().map(|l| json!({"item": l.item, "value": l.value})).collect::<Vec<_>>(),
                    })
                })
                .collect();
            json!({
                "number": night.number,
                "forgets": night.forgets,
                "sessions": sessions,
            })
        })
        .collect();
    json!({
        "name": story.name,
        "items": story.items,
        "nights": nights,
        "absorb_args": story.absorb_args,
        "scenarios": story.scenarios,
        "leak_scan": story.leak_scan,
    })
}

/// Writes the definition of `story` beside the run, with the hash of the
/// `splinter` binary that will run the nights, or checks that the definition
/// already there is the same.
///
/// # Errors
/// The files cannot be accessed, or a run was begun with another definition.
pub fn freeze(story: &Storyline, layout: &Layout, binary: &Path) -> anyhow::Result<()> {
    let definition = definition(story);
    let hash = blake3::hash(definition.to_string().as_bytes())
        .to_hex()
        .to_string();
    let path = layout.frozen();
    if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let frozen: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let before = frozen["definition_blake3"].as_str().unwrap_or_default();
        if before != hash {
            bail!(
                "{}: the storyline {} changed since this run began (definition {before} then, {hash} now); \
                 use another output directory",
                path.display(),
                story.name
            );
        }
        return Ok(());
    }
    std::fs::create_dir_all(layout.root())
        .with_context(|| format!("creating {}", layout.root().display()))?;
    let bytes = std::fs::read(binary).with_context(|| format!("reading {}", binary.display()))?;
    let frozen = json!({
        "definition_blake3": hash,
        "splinter_binary": binary.display().to_string(),
        "splinter_binary_blake3": blake3::hash(&bytes).to_hex().to_string(),
        "definition": definition,
    });
    std::fs::write(&path, serde_json::to_string_pretty(&frozen)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storylines::by_name;

    #[test]
    fn a_storyline_that_changed_since_it_began_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let binary = dir.path().join("splinter");
        std::fs::write(&binary, b"binary").expect("binary");
        let layout = Layout::new(dir.path(), "main");
        let mut story = by_name("main").expect("main");
        freeze(&story, &layout, &binary).expect("first freeze");
        freeze(&story, &layout, &binary).expect("same definition");
        story.scenarios[0].expectation.push_str(" (edited)");
        let refused = freeze(&story, &layout, &binary).expect_err("edited");
        assert!(refused.to_string().contains("changed since this run began"));
    }
}
