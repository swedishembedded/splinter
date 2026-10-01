// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Compressed sparse rows for adjacency, and binary-lifting ancestry.

use crate::error::{Error, Result};

/// A node that is not there.
pub(crate) const NONE: u32 = u32::MAX;

/// Adjacency in compressed sparse row form: the neighbours of node `n` are
/// `targets[offsets[n]..offsets[n + 1]]`.
#[derive(Debug, Default)]
pub(crate) struct Csr {
    offsets: Vec<u32>,
    targets: Vec<u32>,
    labels: Vec<u8>,
}

impl Csr {
    /// Builds from `(from, to, label)` triples over `nodes` nodes.
    pub(crate) fn build(nodes: usize, mut triples: Vec<(u32, u32, u8)>) -> Self {
        triples.sort_unstable();
        triples.dedup();
        let mut offsets = vec![0u32; nodes + 1];
        for (from, _, _) in &triples {
            offsets[*from as usize + 1] += 1;
        }
        for i in 0..nodes {
            offsets[i + 1] += offsets[i];
        }
        let targets = triples.iter().map(|t| t.1).collect();
        let labels = triples.iter().map(|t| t.2).collect();
        Self {
            offsets,
            targets,
            labels,
        }
    }

    /// The neighbours of `node` with their labels.
    pub(crate) fn neighbours(&self, node: u32) -> impl Iterator<Item = (u32, u8)> + '_ {
        let (from, to) = match (
            self.offsets.get(node as usize),
            self.offsets.get(node as usize + 1),
        ) {
            (Some(a), Some(b)) => (*a as usize, *b as usize),
            _ => (0, 0),
        };
        self.targets[from..to]
            .iter()
            .copied()
            .zip(self.labels[from..to].iter().copied())
    }
}

/// Ancestors by binary lifting: `up[j][n]` is the `2^j`th ancestor of `n`.
/// Reaching any ancestor, or the meeting point of two paths, takes a number
/// of steps logarithmic in the depth instead of linear.
#[derive(Debug, Default)]
pub(crate) struct Ancestry {
    up: Vec<Vec<u32>>,
    depth: Vec<u32>,
}

impl Ancestry {
    /// Builds from each node's parent (or [`NONE`]). A cycle is corruption.
    pub(crate) fn build(parent: &[u32]) -> Result<Self> {
        let n = parent.len();
        let mut depth = vec![u32::MAX; n];
        let mut path = Vec::new();
        for start in 0..n {
            let mut node = start;
            path.clear();
            while depth[node] == u32::MAX {
                path.push(node);
                if path.len() > n {
                    return Err(Error::corrupt("index", "the parent links form a cycle"));
                }
                match parent[node] {
                    NONE => {
                        depth[node] = 0;
                        path.pop();
                        break;
                    }
                    next => node = next as usize,
                }
            }
            let mut d = depth[node];
            for visited in path.iter().rev() {
                d += 1;
                depth[*visited] = d;
            }
        }
        let mut up = vec![parent.to_vec()];
        while up
            .last()
            .is_some_and(|level| level.iter().any(|p| *p != NONE))
            && up.len() < 33
        {
            let last = &up[up.len() - 1];
            let next = last
                .iter()
                .map(|p| if *p == NONE { NONE } else { last[*p as usize] })
                .collect();
            up.push(next);
        }
        Ok(Self { up, depth })
    }

    pub(crate) fn depth(&self, node: u32) -> u32 {
        self.depth[node as usize]
    }

    /// The `steps`th ancestor of `node`, if the path is that long.
    pub(crate) fn nth(&self, node: u32, steps: u32) -> Option<u32> {
        if steps > self.depth(node) {
            return None;
        }
        let mut node = node;
        for (level, table) in self.up.iter().enumerate() {
            if steps >> level & 1 == 1 {
                node = table[node as usize];
            }
        }
        (steps >> self.up.len() == 0).then_some(node)
    }

    /// The deepest node both `a` and `b` descend from, or are.
    pub(crate) fn common(&self, a: u32, b: u32) -> Option<u32> {
        let (mut a, mut b) = (a, b);
        let (da, db) = (self.depth(a), self.depth(b));
        if da > db {
            a = self.nth(a, da - db)?;
        } else {
            b = self.nth(b, db - da)?;
        }
        if a == b {
            return Some(a);
        }
        for table in self.up.iter().rev() {
            if table[a as usize] != table[b as usize] {
                a = table[a as usize];
                b = table[b as usize];
            }
        }
        match self.up[0][a as usize] {
            NONE => None,
            common => Some(common),
        }
    }
}
