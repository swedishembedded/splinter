// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What `lineage` prints as text: the artifact, then a tree of where it
//! came from and a tree of what came from it, one artifact per line with
//! the relation that reached it.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use splinter_campaign::lineage::{Branch, Lineage, Node};

use crate::output::Report;

impl Report for Lineage {
    fn human(&self) -> String {
        let mut out = String::new();
        let Some(root) = self.nodes.first() else {
            return out;
        };
        let _ = writeln!(out, "{} {}  {}", root.kind.as_str(), root.id, root.label);
        let nodes: BTreeMap<&str, &Node> = self.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
        for (title, arrow, tree) in [
            ("where it came from", "->", &self.up),
            ("what came from it", "<-", &self.down),
        ] {
            let Some(tree) = tree else {
                continue;
            };
            let _ = writeln!(out, "{title}:");
            if tree.cut {
                let _ = writeln!(out, "  (more beyond --depth)");
            } else if tree.children.is_empty() {
                let _ = writeln!(out, "  (nothing recorded)");
            }
            for child in &tree.children {
                branch(&nodes, &mut out, child, arrow, 1);
            }
        }
        out
    }
}

/// Writes `node` and what the walk reached from it, `level` deep.
fn branch(
    nodes: &BTreeMap<&str, &Node>,
    out: &mut String,
    node: &Branch,
    arrow: &str,
    level: usize,
) {
    let relation = node.relation.map_or("", |r| r.as_str());
    let (kind, label) = nodes
        .get(node.id.as_str())
        .map_or(("?", ""), |n| (n.kind.as_str(), n.label.as_str()));
    let note = if node.repeat {
        "  (shown above)"
    } else if node.cut {
        "  (more beyond --depth)"
    } else {
        ""
    };
    let _ = writeln!(
        out,
        "{}{arrow} {relation} {kind} {}  {label}{note}",
        "  ".repeat(level),
        node.id
    );
    for child in &node.children {
        branch(nodes, out, child, arrow, level + 1);
    }
}
