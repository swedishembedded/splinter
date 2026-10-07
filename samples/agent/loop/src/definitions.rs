// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The definitions in effect for a run: the Markdown subagents, skills and
//! commands sven finds for the repository the work is done in.
//!
//! Which ones apply, and which override which, is sven's rule (a project's
//! own definitions over its parents' and the user's), so this module asks
//! sven's discovery rather than reading directories itself. It hashes what
//! was found, so a run can say exactly which definitions produced it and a
//! resumed run can notice that they changed.

use std::path::Path;

use serde::{Deserialize, Serialize};
use splinter_sdk::agent::sven::workspace::{
    discover_agents, discover_commands, discover_skills, find_project_context_file,
};
use splinter_sdk::vocabulary::digest::Digest;

/// One definition in effect.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Entry {
    /// `agent`, `skill`, `command` or `context`.
    pub kind: String,
    /// The name it is invoked by.
    pub name: String,
    /// The file it came from.
    pub path: String,
    /// SHA-256 of its text.
    pub digest: String,
}

/// Everything in effect, in a fixed order, and one digest over it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Definitions {
    /// The definitions, sorted by kind then name.
    pub entries: Vec<Entry>,
    /// SHA-256 over the kind, name and text digest of every entry. The path
    /// is left out: the same definitions found somewhere else are the same
    /// definitions.
    pub digest: String,
}

/// The definitions in effect for work in `repository`.
#[must_use]
pub fn effective(repository: &Path) -> Definitions {
    let root = Some(repository);
    let mut entries: Vec<Entry> = Vec::new();
    for agent in discover_agents(root) {
        entries.push(entry(
            "agent",
            &agent.name,
            &agent.agent_md_path,
            &agent.content,
        ));
    }
    for skill in discover_skills(root) {
        entries.push(entry(
            "skill",
            &skill.command,
            &skill.skill_md_path,
            &skill.content,
        ));
    }
    for command in discover_commands(root) {
        entries.push(entry(
            "command",
            &command.command,
            &command.skill_md_path,
            &command.content,
        ));
    }
    if let Some(path) = find_project_context_file(repository) {
        if let Ok(text) = std::fs::read_to_string(&path) {
            entries.push(entry("context", "project", &path, &text));
        }
    }
    entries.sort_by(|a, b| (&a.kind, &a.name).cmp(&(&b.kind, &b.name)));
    let mut summary = String::new();
    for e in &entries {
        summary.push_str(&format!("{}\t{}\t{}\n", e.kind, e.name, e.digest));
    }
    Definitions {
        entries,
        digest: Digest::sha256_of(summary.as_bytes()).to_string(),
    }
}

fn entry(kind: &str, name: &str, path: &Path, text: &str) -> Entry {
    Entry {
        kind: kind.to_string(),
        name: name.to_string(),
        path: path.display().to_string(),
        digest: Digest::sha256_of(text.as_bytes()).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(agent_body: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let agents = dir.path().join(".sven").join("agents");
        std::fs::create_dir_all(&agents).unwrap();
        std::fs::write(
            agents.join("reviewer.md"),
            format!("---\nname: reviewer\ndescription: reviews a diff\n---\n{agent_body}\n"),
        )
        .unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "project rules\n").unwrap();
        dir
    }

    #[test]
    fn the_agent_and_the_project_context_are_listed_with_their_text_digests() {
        let dir = project("Read the diff.");
        let found = effective(dir.path());
        let kinds: Vec<(&str, &str)> = found
            .entries
            .iter()
            .filter(|e| e.path.starts_with(&dir.path().display().to_string()))
            .map(|e| (e.kind.as_str(), e.name.as_str()))
            .collect();
        assert!(kinds.contains(&("agent", "reviewer")), "{kinds:?}");
        assert!(kinds.contains(&("context", "project")), "{kinds:?}");
    }

    #[test]
    fn the_digest_follows_the_text_and_not_the_place() {
        let a = effective(project("Read the diff.").path());
        let again = effective(project("Read the diff.").path());
        let edited = effective(project("Read the diff twice.").path());
        assert_eq!(
            a.digest, again.digest,
            "the same definitions in another place"
        );
        assert_ne!(a.digest, edited.digest, "an edited definition");
    }
}
