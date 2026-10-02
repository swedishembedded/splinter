// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The in-memory read model assembled from runs: where records are, which
//! group they belong to, and how they link.

use std::collections::HashMap;

use super::csr::{Ancestry, Csr, NONE};
use super::run::Run;
use crate::error::Result;
use crate::id::{ContentId, RecordId};
use crate::model::{RecordKind, Rel};

/// Where a record is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Loc {
    /// The segment's content id.
    pub segment: ContentId,
    /// The block within the segment.
    pub block: u32,
    /// The row within the block.
    pub row: u32,
}

/// Adjacency, groups and ancestry over a snapshot's records.
///
/// Nodes are the record ids in order; neighbours are stored as compressed
/// sparse rows, forwards and backwards, and ancestry as binary-lifting tables.
pub struct Index {
    nodes: Vec<RecordId>,
    locs: Vec<Option<Loc>>,
    kinds: Vec<u8>,
    parent: Vec<u32>,
    children: Csr,
    out: Csr,
    inc: Csr,
    ancestry: Ancestry,
    by_family: HashMap<ContentId, Vec<u32>>,
    by_instance: HashMap<ContentId, Vec<u32>>,
    by_attempt: HashMap<RecordId, Vec<u32>>,
    by_kind: HashMap<u8, Vec<u32>>,
    by_entity: HashMap<ContentId, u32>,
    by_class: HashMap<u64, Vec<u32>>,
    entity_of: HashMap<u32, ContentId>,
}

/// A node that has no stored record: an edge endpoint written elsewhere.
const UNSTORED: u8 = 255;

impl Index {
    pub(super) fn assemble(runs: &[Run]) -> Result<Self> {
        // First copy of a record wins: a record can sit in two segments while
        // a compaction's inputs and output are both named.
        let mut rows = HashMap::new();
        for run in runs {
            for row in &run.rows {
                rows.entry(row.id).or_insert((run, row));
            }
        }
        let mut edges = Vec::new();
        for run in runs {
            edges.extend(run.edges.iter().map(|e| e.edge));
        }
        let mut ids: Vec<RecordId> = rows.keys().copied().collect();
        ids.extend(edges.iter().flat_map(|e| [e.from, e.to]));
        ids.extend(rows.values().filter_map(|(_, r)| r.parent));
        ids.sort_unstable();
        ids.dedup();
        let at = |id: &RecordId| ids.binary_search(id).map(|i| i as u32).unwrap_or(NONE);

        let n = ids.len();
        let (mut locs, mut kinds, mut parent) = (vec![None; n], vec![UNSTORED; n], vec![NONE; n]);
        let (mut by_family, mut by_instance, mut by_attempt, mut by_kind) = (
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        let mut by_entity: HashMap<ContentId, u32> = HashMap::new();
        let mut by_class: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut entity_of: HashMap<u32, ContentId> = HashMap::new();
        for (id, (run, row)) in &rows {
            let i = at(id) as usize;
            locs[i] = run.covers.get(row.seg as usize).map(|segment| Loc {
                segment: *segment,
                block: row.block,
                row: row.row,
            });
            kinds[i] = row.kind;
            parent[i] = row.parent.as_ref().map_or(NONE, at);
            if let Some(f) = row.family {
                by_family.entry(f).or_insert_with(Vec::new).push(i as u32);
            }
            if let Some(t) = row.instance {
                by_instance.entry(t).or_insert_with(Vec::new).push(i as u32);
            }
            if let Some(a) = row.attempt {
                by_attempt.entry(a).or_insert_with(Vec::new).push(i as u32);
            }
            by_kind
                .entry(row.kind)
                .or_insert_with(Vec::new)
                .push(i as u32);
            if let Some((entity, class)) = row.entity {
                // The same entity can be stored by several writers: the one
                // with the lowest record id is the entity's record.
                by_entity
                    .entry(entity)
                    .and_modify(|first| *first = (*first).min(i as u32))
                    .or_insert(i as u32);
                entity_of.insert(i as u32, entity);
                if class != 0 {
                    by_class.entry(class).or_default().push(i as u32);
                }
            }
        }
        for list in by_class.values_mut() {
            list.sort_unstable();
        }
        for list in by_family
            .values_mut()
            .chain(by_instance.values_mut())
            .chain(by_attempt.values_mut())
            .chain(by_kind.values_mut())
        {
            list.sort_unstable();
        }
        let child_pairs = (0..n)
            .filter(|i| parent[*i] != NONE)
            .map(|i| (parent[i], i as u32, 0))
            .collect();
        let (mut out_pairs, mut in_pairs) = (Vec::new(), Vec::new());
        for e in &edges {
            let (from, to) = (at(&e.from), at(&e.to));
            out_pairs.push((from, to, e.rel.code()));
            in_pairs.push((to, from, e.rel.code()));
        }
        Ok(Self {
            ancestry: Ancestry::build(&parent)?,
            children: Csr::build(n, child_pairs),
            out: Csr::build(n, out_pairs),
            inc: Csr::build(n, in_pairs),
            nodes: ids,
            locs,
            kinds,
            parent,
            by_family,
            by_instance,
            by_attempt,
            by_kind,
            by_entity,
            by_class,
            entity_of,
        })
    }

    /// The record that defines the entity `id`: of the records that define
    /// it, the one with the lowest id.
    pub fn entity(&self, id: &ContentId) -> Option<RecordId> {
        self.by_entity.get(id).map(|n| self.nodes[*n as usize])
    }

    /// The entities of the class with hash `class`, each once, with the
    /// record that defines it, in record order.
    pub fn entities_of_class(&self, class: u64) -> Vec<(ContentId, RecordId)> {
        let mut seen = std::collections::HashSet::new();
        self.by_class
            .get(&class)
            .into_iter()
            .flatten()
            .filter_map(|n| {
                let id = self.entity_of.get(n)?;
                let first = *self.by_entity.get(id)?;
                (first == *n && seen.insert(*id)).then(|| (*id, self.nodes[first as usize]))
            })
            .collect()
    }

    /// How many distinct entities are indexed.
    pub fn entity_count(&self) -> usize {
        self.by_entity.len()
    }

    /// Whether the entity `id` is of the class with hash `class`.
    pub fn entity_has_class(&self, id: &ContentId, class: u64) -> bool {
        self.by_entity.get(id).is_some_and(|n| {
            self.by_class
                .get(&class)
                .is_some_and(|nodes| nodes.binary_search(n).is_ok())
        })
    }

    fn node(&self, id: RecordId) -> Option<u32> {
        self.nodes.binary_search(&id).ok().map(|i| i as u32)
    }

    fn ids(&self, nodes: impl Iterator<Item = u32>) -> Vec<RecordId> {
        nodes.map(|n| self.nodes[n as usize]).collect()
    }

    /// How many records are indexed.
    pub fn len(&self) -> usize {
        self.locs.iter().filter(|l| l.is_some()).count()
    }

    /// Whether nothing is indexed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the record is stored in the snapshot.
    pub fn contains(&self, id: RecordId) -> bool {
        self.loc(id).is_some()
    }

    /// Where the record is stored.
    pub fn loc(&self, id: RecordId) -> Option<Loc> {
        self.node(id).and_then(|n| self.locs[n as usize])
    }

    /// The kind of a stored record.
    pub fn kind_of(&self, id: RecordId) -> Option<RecordKind> {
        self.node(id)
            .and_then(|n| RecordKind::from_code(self.kinds[n as usize]))
    }

    /// The record before this one on its path.
    pub fn parent(&self, id: RecordId) -> Option<RecordId> {
        let p = self.parent[self.node(id)? as usize];
        (p != NONE).then(|| self.nodes[p as usize])
    }

    /// The records that follow this one, in id order.
    pub fn children(&self, id: RecordId) -> Vec<RecordId> {
        self.node(id).map_or_else(Vec::new, |n| {
            self.ids(self.children.neighbours(n).map(|(c, _)| c))
        })
    }

    /// How many steps the record is from the start of its path.
    pub fn depth(&self, id: RecordId) -> Option<usize> {
        self.node(id).map(|n| self.ancestry.depth(n) as usize)
    }

    /// The ancestor `steps` back along the path, found in a number of
    /// lookups logarithmic in `steps`.
    pub fn nth_ancestor(&self, id: RecordId, steps: usize) -> Option<RecordId> {
        let ancestor = self
            .ancestry
            .nth(self.node(id)?, u32::try_from(steps).ok()?)?;
        Some(self.nodes[ancestor as usize])
    }

    /// The last record two paths share, which may be one of them.
    pub fn common_prefix(&self, a: RecordId, b: RecordId) -> Option<RecordId> {
        let common = self.ancestry.common(self.node(a)?, self.node(b)?)?;
        Some(self.nodes[common as usize])
    }

    fn typed(&self, csr: &Csr, id: RecordId, rel: Option<Rel>) -> Vec<RecordId> {
        let Some(n) = self.node(id) else {
            return Vec::new();
        };
        self.ids(
            csr.neighbours(n)
                .filter(|(_, code)| rel.is_none_or(|r| r.code() == *code))
                .map(|(t, _)| t),
        )
    }

    /// The records `id` links to, optionally of one relation.
    pub fn edges_from(&self, id: RecordId, rel: Option<Rel>) -> Vec<RecordId> {
        self.typed(&self.out, id, rel)
    }

    /// The records that link to `id`, optionally of one relation.
    pub fn edges_to(&self, id: RecordId, rel: Option<Rel>) -> Vec<RecordId> {
        self.typed(&self.inc, id, rel)
    }

    fn walk(&self, csr: &Csr, start: RecordId, rels: Option<&[Rel]>) -> Vec<RecordId> {
        let Some(start) = self.node(start) else {
            return Vec::new();
        };
        let mut seen = vec![false; self.nodes.len()];
        seen[start as usize] = true;
        let mut pending = vec![start];
        let mut found = Vec::new();
        while let Some(node) = pending.pop() {
            for (next, code) in csr.neighbours(node) {
                if rels.is_none_or(|r| r.iter().any(|r| r.code() == code)) && !seen[next as usize] {
                    seen[next as usize] = true;
                    found.push(next);
                    pending.push(next);
                }
            }
        }
        found.sort_unstable();
        self.ids(found.into_iter())
    }

    /// Everything reachable by following links away from `id`.
    pub fn reachable_from(&self, id: RecordId, rels: Option<&[Rel]>) -> Vec<RecordId> {
        self.walk(&self.out, id, rels)
    }

    /// Everything that reaches `id` by following links.
    pub fn reaching(&self, id: RecordId, rels: Option<&[Rel]>) -> Vec<RecordId> {
        self.walk(&self.inc, id, rels)
    }

    fn group(&self, group: Option<&Vec<u32>>) -> Vec<RecordId> {
        group.map_or_else(Vec::new, |g| self.ids(g.iter().copied()))
    }

    /// Records of an episode family.
    pub fn by_family(&self, family: &ContentId) -> Vec<RecordId> {
        self.group(self.by_family.get(family))
    }

    /// Records of a task instance.
    pub fn by_task_instance(&self, instance: &ContentId) -> Vec<RecordId> {
        self.group(self.by_instance.get(instance))
    }

    /// Records of an attempt.
    pub fn by_attempt(&self, attempt: RecordId) -> Vec<RecordId> {
        self.group(self.by_attempt.get(&attempt))
    }

    /// Records of a kind.
    pub fn by_kind(&self, kind: RecordKind) -> Vec<RecordId> {
        self.group(self.by_kind.get(&kind.code()))
    }
}
