// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The lineage graph in memory: every artifact the stores hold, the edges
//! between them, and the one way an id a person typed names a node.

use std::collections::{BTreeMap, BTreeSet};

use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::ids::{strip_algorithm, MIN_PREFIX};

use super::{Node, NodeKind, Relation};

/// See the module documentation.
#[derive(Debug, Default)]
pub(crate) struct Graph {
    nodes: BTreeMap<String, Node>,
    /// Each node's edges up: `(relation, to)`.
    up: BTreeMap<String, BTreeSet<(Relation, String)>>,
    /// Each node's edges down: `(relation, from)`.
    down: BTreeMap<String, BTreeSet<(Relation, String)>>,
}

impl Graph {
    /// Records `id` as a `kind` described by `label`, replacing what a
    /// mention of it guessed.
    pub(crate) fn record(&mut self, id: &str, kind: NodeKind, label: String) {
        self.nodes.insert(
            id.to_string(),
            Node {
                id: id.to_string(),
                kind,
                label,
            },
        );
    }

    /// Whether `id` was recorded or mentioned.
    pub(crate) fn contains(&self, id: &str) -> bool {
        self.nodes.contains_key(id)
    }

    /// Makes sure `id` is a node: an artifact named by another before its
    /// own record is read (or one whose record is not stored) gets
    /// `label`, which its record replaces.
    pub(crate) fn mention(&mut self, id: &str, kind: NodeKind, label: impl FnOnce() -> String) {
        if !self.contains(id) {
            self.record(id, kind, label());
        }
    }

    /// Adds the edge `from` -> `to`: `from` was derived from `to`.
    pub(crate) fn link(&mut self, from: &str, relation: Relation, to: &str) {
        self.up
            .entry(from.to_string())
            .or_default()
            .insert((relation, to.to_string()));
        self.down
            .entry(to.to_string())
            .or_default()
            .insert((relation, from.to_string()));
    }

    /// The node `id`.
    pub(crate) fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// What `id` was derived from, ordered by relation, then id.
    pub(crate) fn up(&self, id: &str) -> Vec<(Relation, String)> {
        self.up.get(id).into_iter().flatten().cloned().collect()
    }

    /// What was derived from `id`, ordered by relation, then id.
    pub(crate) fn down(&self, id: &str) -> Vec<(Relation, String)> {
        self.down.get(id).into_iter().flatten().cloned().collect()
    }

    /// The one node `given` names: its exact id; or, for a content
    /// address, `blake3:<hex>` or the hex alone, or at least
    /// [`MIN_PREFIX`] hex digits of it; or a prefix of any other id (a
    /// candidate's). A prefix naming more than one node is refused with
    /// every node it names.
    pub(crate) fn resolve(&self, given: &str) -> Result<&Node, OrchestratorError> {
        if let Some(node) = self.nodes.get(given) {
            return Ok(node);
        }
        let hex = strip_algorithm(given);
        let is_hex = !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit());
        let matches: Vec<&Node> = if is_hex {
            if hex.len() < MIN_PREFIX {
                return Err(OrchestratorError::Refused(format!(
                    "{given:?} is too short: give blake3:<hex>, or at least {MIN_PREFIX} hex \
                     digits of an id"
                )));
            }
            let hex = hex.to_ascii_lowercase();
            let algorithms: &[&str] = match given.split_once(':') {
                Some(("sha256", _)) => &["sha256"],
                Some(_) => &["blake3"],
                None => &["blake3", "sha256"],
            };
            algorithms
                .iter()
                .flat_map(|algorithm| {
                    let prefix = format!("{algorithm}:{hex}");
                    self.nodes
                        .range(prefix.clone()..)
                        .take_while(move |(id, _)| id.starts_with(&prefix))
                        .map(|(_, node)| node)
                })
                .collect()
        } else if given.starts_with("blake3:") || given.starts_with("sha256:") || given.is_empty() {
            Vec::new()
        } else {
            self.nodes
                .range(given.to_string()..)
                .take_while(|(id, _)| id.starts_with(given))
                .map(|(_, node)| node)
                .collect()
        };
        match matches.as_slice() {
            [] => Err(OrchestratorError::NotFound {
                what: "artifact",
                id: given.to_string(),
            }),
            [one] => Ok(*one),
            many => Err(OrchestratorError::AmbiguousArtifact {
                id: given.to_string(),
                candidates: many
                    .iter()
                    .map(|n| format!("{} {}", n.kind.as_str(), n.id))
                    .collect(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> Graph {
        let mut graph = Graph::default();
        graph.record(
            &format!("blake3:abcd{}", "1".repeat(60)),
            NodeKind::Source,
            "s".into(),
        );
        graph.record(
            &format!("blake3:abcd{}", "2".repeat(60)),
            NodeKind::Task,
            "t".into(),
        );
        graph.record(
            "candidate-20260930T120000.000-0001",
            NodeKind::Candidate,
            "c".into(),
        );
        graph
    }

    #[test]
    fn an_id_is_resolved_whole_by_hex_prefix_or_by_prefix_and_never_guessed() {
        let graph = graph();
        let task = format!("blake3:abcd{}", "2".repeat(60));
        assert_eq!(graph.resolve(&task).unwrap().kind, NodeKind::Task);
        assert_eq!(graph.resolve("ABCD2").unwrap().kind, NodeKind::Task);
        assert_eq!(
            graph.resolve("blake3:abcd1").unwrap().kind,
            NodeKind::Source
        );
        assert_eq!(
            graph.resolve("candidate-2026").unwrap().kind,
            NodeKind::Candidate
        );
        let Err(OrchestratorError::AmbiguousArtifact { candidates, .. }) = graph.resolve("abcd")
        else {
            panic!("abcd names two artifacts");
        };
        assert_eq!(candidates.len(), 2);
        assert!(candidates[0].starts_with("source blake3:abcd1"));
        assert!(candidates[1].starts_with("task blake3:abcd2"));
        assert!(matches!(
            graph.resolve("abc"),
            Err(OrchestratorError::Refused(_))
        ));
        assert!(matches!(
            graph.resolve("ffff"),
            Err(OrchestratorError::NotFound { .. })
        ));
        assert!(matches!(
            graph.resolve("release-x"),
            Err(OrchestratorError::NotFound { .. })
        ));
    }
}
