// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A session: one process's handle for writing entities, evaluations and
//! links and reading them back.
//!
//! A snapshot is a fixed point in time, and the lookups over it are built from
//! every record it holds, so a process that writes one record and then looks
//! one up must not take a fresh snapshot each time. A session keeps one
//! snapshot and the few records written since, answers from both, and takes
//! a new snapshot only when those few have grown large, so the cost of
//! rebuilding is shared by many writes. Its own writes are visible to it at
//! once; other writers' appear when it refreshes.

use std::collections::{HashMap, HashSet};

use crate::analyze::{EvalFilter, EvaluationView};
use crate::blob::BlobRef;
use crate::database::Database;
use crate::error::{Error, Result};
use crate::id::{ContentId, RecordId, WriterIdentity};
use crate::index::StoredEntity;
use crate::ingest::Collector;
use crate::manifest::Snapshot;
use crate::model::{Body, Entity, Evaluation, EvaluatorRef, Rel};

/// How many records a session writes before it refreshes its snapshot.
const DEFAULT_REFRESH_EVERY: usize = 4096;

/// A writer and the snapshot it reads through.
pub struct Session {
    collector: Collector,
    base: Snapshot,
    db: Database,
    entities: HashMap<ContentId, (RecordId, Entity)>,
    entity_log: Vec<ContentId>,
    by_class: HashMap<String, Vec<ContentId>>,
    evaluations: Vec<EvaluationView>,
    retracted: HashSet<EvaluatorRef>,
    retracted_log: Vec<EvaluatorRef>,
    refresh_every: usize,
    base_size: usize,
    delta_bytes: usize,
}

/// How much of a session's own writes it keeps in memory before it refreshes,
/// however few records that is.
const REFRESH_BYTES: usize = 64 << 20;

/// Where a session stood, to return to with [`Session::rollback`].
#[derive(Debug, Clone, Copy)]
pub struct Mark {
    writer: (usize, usize),
    entities: usize,
    evaluations: usize,
    retracted: usize,
    delta_bytes: usize,
}

impl Session {
    /// A session that writes as `identity` and reads from the database as it
    /// is now.
    pub fn open(db: &Database, identity: &WriterIdentity) -> Result<Self> {
        let collector = db.collector(identity)?;
        let base = db.snapshot()?;
        Ok(Self {
            collector,
            db: db.clone(),
            base_size: base.entity_count()?,
            base,
            entities: HashMap::new(),
            entity_log: Vec::new(),
            by_class: HashMap::new(),
            evaluations: Vec::new(),
            retracted: HashSet::new(),
            retracted_log: Vec::new(),
            refresh_every: DEFAULT_REFRESH_EVERY,
            delta_bytes: 0,
        })
    }

    /// The same session, refreshing after `n` unrefreshed writes instead of the
    /// default (but never before half the size of its snapshot, so refreshing
    /// stays amortised).
    #[must_use]
    pub fn refreshing_every(mut self, n: usize) -> Self {
        self.refresh_every = n.max(1);
        self
    }

    fn written(&self) -> usize {
        self.entities.len() + self.evaluations.len() + self.retracted.len()
    }

    /// Whether the session has written enough since its snapshot that it
    /// should take a new one, by number of records or by bytes held.
    pub fn refresh_due(&self) -> bool {
        self.written() >= self.refresh_every.max(self.base_size / 2)
            || self.delta_bytes >= REFRESH_BYTES
    }

    /// Refreshes if [`Session::refresh_due`]. It is a call of its own, made
    /// between writes and never inside one, because a refresh commits: a write
    /// that spans several records would otherwise be committed in pieces.
    pub fn refresh_if_due(&mut self) -> Result<()> {
        if self.refresh_due() {
            self.refresh()?;
        }
        Ok(())
    }

    /// Where the session stands now, so a write that fails halfway can be
    /// undone with [`Session::rollback`].
    pub fn begin(&self) -> Mark {
        Mark {
            writer: self.collector.mark(),
            entities: self.entity_log.len(),
            evaluations: self.evaluations.len(),
            retracted: self.retracted_log.len(),
            delta_bytes: self.delta_bytes,
        }
    }

    /// Forgets everything written since `mark`: it will not be committed and
    /// the session no longer reads it. Only valid while nothing was flushed
    /// since the mark was taken.
    pub fn rollback(&mut self, mark: Mark) {
        self.collector.rollback(mark.writer);
        for id in self.entity_log.split_off(mark.entities) {
            if let Some((_, entity)) = self.entities.remove(&id) {
                if let Some(ids) = self.by_class.get_mut(&entity.class) {
                    ids.retain(|kept| *kept != id);
                }
            }
        }
        self.evaluations.truncate(mark.evaluations);
        for evaluator in self.retracted_log.split_off(mark.retracted) {
            self.retracted.remove(&evaluator);
        }
        self.delta_bytes = mark.delta_bytes;
    }

    /// Makes everything written so far durable and visible to other
    /// readers.
    pub fn flush(&mut self) -> Result<()> {
        self.collector.flush()
    }

    /// Flushes and hands this session's history to the catalog, so a process
    /// that is done leaves no ref of its own behind. The session may be used
    /// again; its next commit starts a new chain.
    pub fn close(&mut self) -> Result<()> {
        self.flush()?;
        let job = self.collector.job().to_owned();
        self.db.retire_job(&job)?;
        Ok(())
    }

    /// Flushes, then reads from the database as it is now, so other
    /// writers' records appear.
    pub fn refresh(&mut self) -> Result<()> {
        self.flush()?;
        self.base = self.db.snapshot()?;
        self.base_size = self.base.entity_count()?;
        self.entities.clear();
        self.entity_log.clear();
        self.by_class.clear();
        self.evaluations.clear();
        self.retracted.clear();
        self.retracted_log.clear();
        self.delta_bytes = 0;
        Ok(())
    }

    /// A snapshot holding everything this session and others have flushed,
    /// for queries the session does not answer itself.
    pub fn snapshot(&mut self) -> Result<&Snapshot> {
        self.refresh()?;
        Ok(&self.base)
    }

    /// Stores an application entity and returns its content id. An entity
    /// the session or its snapshot already holds is not written again.
    pub fn put_entity(&mut self, entity: &Entity) -> Result<ContentId> {
        let id = entity.id()?;
        if self.entities.contains_key(&id) || self.base.entity(&id)?.is_some() {
            return Ok(id);
        }
        let record = self.collector.record(Body::Entity(entity.clone()))?;
        self.delta_bytes += entity.value.to_string().len();
        self.entities.insert(id, (record, entity.clone()));
        self.entity_log.push(id);
        self.by_class
            .entry(entity.class.clone())
            .or_default()
            .push(id);
        Ok(id)
    }

    /// The application entity with content id `id`.
    pub fn entity(&self, id: &ContentId) -> Result<Option<Entity>> {
        if let Some((_, entity)) = self.entities.get(id) {
            return Ok(Some(entity.clone()));
        }
        self.base.entity_body(id)
    }

    /// Whether the session or its snapshot holds an application entity of
    /// `class` with content id `id`, without reading any body.
    pub fn has(&self, id: &ContentId, class: &str) -> Result<bool> {
        if let Some((_, entity)) = self.entities.get(id) {
            return Ok(entity.class == class);
        }
        self.base.entity_in_class(id, class)
    }

    /// The ids of every application entity of `class`, in the order first
    /// written, without reading any body.
    pub fn entity_ids(&self, class: &str) -> Result<Vec<ContentId>> {
        let mut ids: Vec<ContentId> = self
            .base
            .entity_ids(class)?
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        ids.extend(self.by_class.get(class).into_iter().flatten().copied());
        Ok(ids)
    }

    fn entity_record(&self, id: &ContentId) -> Result<RecordId> {
        if let Some((record, _)) = self.entities.get(id) {
            return Ok(*record);
        }
        self.base.entity(id)?.ok_or_else(|| Error::NotFound {
            what: format!("entity {id}"),
        })
    }

    /// Every entity of `class`, each once, in the order first written.
    pub fn entities(&self, class: &str) -> Result<Vec<StoredEntity>> {
        let mut found = self.base.entities(class)?;
        for id in self.by_class.get(class).into_iter().flatten() {
            if let Some((record, entity)) = self.entities.get(id) {
                found.push(StoredEntity {
                    id: *id,
                    record: *record,
                    entity: entity.clone(),
                });
            }
        }
        Ok(found)
    }

    /// Records an evaluation of something that exists.
    pub fn evaluate(&mut self, evaluation: Evaluation) -> Result<RecordId> {
        let id = self.collector.evaluate(evaluation.clone())?;
        self.evaluations.push(EvaluationView {
            id,
            evaluation,
            timestamp_ns: self.db.clock().now_ns(),
        });
        Ok(id)
    }

    /// Withdraws every judgement by `evaluator`.
    pub fn retract(&mut self, evaluator: EvaluatorRef, reason: &str) -> Result<RecordId> {
        let id = self.collector.retract(evaluator.clone(), reason)?;
        self.retracted.insert(evaluator.clone());
        self.retracted_log.push(evaluator);
        Ok(id)
    }

    /// The evaluations matching `filter`, in write order: the snapshot's,
    /// then this session's.
    pub fn evaluations(&self, filter: &EvalFilter) -> Result<Vec<EvaluationView>> {
        let mut found = self.base.evaluations(filter)?;
        if !self.retracted.is_empty() {
            found.retain(|v| !self.retracted.contains(&v.evaluation.evaluator));
        }
        for view in &self.evaluations {
            let evaluator = &view.evaluation.evaluator;
            let retracted =
                self.retracted.contains(evaluator) || self.base.is_retracted(evaluator)?;
            if filter.accepts(&view.evaluation, retracted) {
                found.push(view.clone());
            }
        }
        Ok(found)
    }

    /// Runs `f` with the session's collector, for records that are not
    /// entities or evaluations, such as the attempts of an agent run. What it
    /// writes is flushed with everything else; the session does not read it
    /// back itself, so a reader takes a snapshot after a refresh.
    pub fn with_collector<R>(&mut self, f: impl FnOnce(&mut Collector) -> Result<R>) -> Result<R> {
        f(&mut self.collector)
    }

    /// Links two records, such as two attempts.
    pub fn link_records(&mut self, from: RecordId, rel: Rel, to: RecordId) -> Result<()> {
        self.collector.link(from, rel, to)
    }

    /// Links two entities.
    pub fn link(&mut self, from: &ContentId, rel: Rel, to: &ContentId) -> Result<()> {
        let (from, to) = (self.entity_record(from)?, self.entity_record(to)?);
        self.collector.link(from, rel, to)
    }

    /// Stores `bytes` and returns their reference.
    pub fn put_blob(&mut self, bytes: &[u8]) -> Result<BlobRef> {
        self.collector.put_blob(bytes)
    }

    /// The bytes with content id `id`: written by this session, or flushed
    /// by anyone.
    pub fn read_blob(&mut self, id: &ContentId) -> Result<Vec<u8>> {
        if let Ok(bytes) = self.collector.read_blob(id) {
            return Ok(bytes);
        }
        if let Ok(bytes) = self.base.read_blob(id) {
            return Ok(bytes);
        }
        self.refresh()?;
        self.base.read_blob(id)
    }
}
