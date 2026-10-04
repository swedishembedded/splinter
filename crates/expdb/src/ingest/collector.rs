// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The collector: the simple write API an agent runtime uses. It knows
//! nothing of segments, manifests or training; it records what happened.

use std::collections::HashSet;

use super::run::{DecisionRef, Run, Standing};
use super::writer::{Destination, Writer};
use crate::blob::BlobRef;
use crate::database::Database;
use crate::error::{Error, Result};
use crate::id::{ContentId, RecordId, WriterIdentity};
use crate::model::{
    family_key, Attempt, Body, Content, CounterfactualSet, Edge, Entity, EpisodeFamily, Evaluation,
    PolicyRef, Record, Rel, State, TaskDefinition, TaskInstance,
};

impl Database {
    /// A collector that publishes directly.
    pub fn collector(&self, identity: &WriterIdentity) -> Result<Collector> {
        self.collector_to(identity, Destination::Publish)
    }

    /// A collector that publishes directly or through the spool.
    pub fn collector_to(
        &self,
        identity: &WriterIdentity,
        destination: Destination,
    ) -> Result<Collector> {
        Ok(Collector {
            writer: Writer::open(self, identity, destination)?,
            entities: HashSet::new(),
        })
    }
}

/// Records experience through one writer.
pub struct Collector {
    pub(super) writer: Writer,
    /// Entities already written by this collector, so a task or a state
    /// shared by many attempts is written once per writer.
    entities: HashSet<ContentId>,
}

impl Collector {
    /// This collector's writer id, unique to this start of it.
    pub fn writer_id(&self) -> crate::id::WriterId {
        self.writer.id()
    }

    /// Records waiting to be sealed.
    pub fn buffered(&self) -> usize {
        self.writer.buffered()
    }

    /// Makes everything recorded so far durable and visible. Dropping a
    /// collector without flushing discards what was not flushed.
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().map(|_| ())
    }

    /// Records a body with no place in any attempt: a skill, a derivation, a
    /// dataset, a retraction.
    pub fn record(&mut self, body: Body) -> Result<RecordId> {
        let record = self.make(body);
        self.writer.push(record)
    }

    /// Records an evaluation of something that already exists.
    pub fn evaluate(&mut self, evaluation: Evaluation) -> Result<RecordId> {
        self.record(Body::Evaluation(evaluation))
    }

    /// Links two records.
    pub fn link(&mut self, from: RecordId, rel: Rel, to: RecordId) -> Result<()> {
        self.writer.link(Edge::new(from, rel, to))
    }

    /// Content as a record carries it: inline if small text, else stored as
    /// a blob and referenced.
    pub fn content(&mut self, bytes: &[u8]) -> Result<Content> {
        if bytes.len() <= self.writer.database().config().inline_payload_limit {
            if let Ok(text) = std::str::from_utf8(bytes) {
                return Ok(Content::text(text));
            }
        }
        Ok(Content::Blob {
            blob: self.writer.blobs().put(bytes)?,
        })
    }

    pub(super) fn make(&mut self, body: Body) -> Record {
        let id = self.writer.next_id();
        Record::new(id, self.writer.database().clock().now_ns(), body)
    }

    pub(crate) fn mark(&self) -> (usize, usize) {
        self.writer.mark()
    }

    pub(crate) fn rollback(&mut self, mark: (usize, usize)) {
        self.writer.rollback(mark);
    }

    /// The name of the job this collector publishes under, for
    /// [`Database::retire_job`].
    pub fn job(&self) -> &str {
        self.writer.job()
    }

    /// Stores `bytes` in the blob store and returns their reference. Whole
    /// content already stored, or shared in part with what is, is not stored
    /// again. The bytes are readable from a snapshot once flushed.
    pub fn put_blob(&mut self, bytes: &[u8]) -> Result<BlobRef> {
        self.writer.blobs().put(bytes)
    }

    /// The bytes with content id `id`, if this collector wrote them or they
    /// are in a pack it has seen.
    pub fn read_blob(&mut self, id: &ContentId) -> Result<Vec<u8>> {
        self.writer.blobs().get_by_id(id)
    }

    /// Stores an application entity and returns its content id. Putting the
    /// same entity again through this collector writes nothing; two
    /// collectors that put it write two records that name one entity.
    pub fn put_entity(&mut self, entity: &Entity) -> Result<ContentId> {
        let id = entity.id()?;
        self.entity(Body::Entity(entity.clone()))?;
        Ok(id)
    }

    /// Writes an entity (task, instance, state, family) once per collector.
    pub(super) fn entity(&mut self, body: Body) -> Result<()> {
        if let Some(id) = body.entity_id()? {
            if !self.entities.insert(id) {
                return Ok(());
            }
        }
        let record = self.make(body);
        self.writer.push(record).map(|_| ())
    }

    /// Begins an attempt at a task instance from an initial state. Every
    /// attempt at the same instance from the same state is in one family,
    /// however many writers run them.
    pub fn start_attempt(
        &mut self,
        definition: &TaskDefinition,
        instance: &TaskInstance,
        initial: &State,
        policy: &PolicyRef,
        seed: Option<u64>,
    ) -> Result<Run<'_>> {
        let definition_id = definition.id()?;
        if instance.definition != definition_id {
            return Err(Error::invalid(
                "task instance",
                "its definition is not the definition given",
            ));
        }
        let (instance_id, state_id) = (instance.id()?, initial.id()?);
        let family = family_key(&instance_id, &state_id);
        self.entity(Body::TaskDefinition(definition.clone()))?;
        self.entity(Body::TaskInstance(instance.clone()))?;
        self.entity(Body::State(initial.clone()))?;
        self.entity(Body::Family(EpisodeFamily {
            task_instance: instance_id,
            initial_state: state_id,
        }))?;

        let attempt_id = self.writer.next_id();
        let mut attempt = Record::new(
            attempt_id,
            self.writer.database().clock().now_ns(),
            Body::Attempt(Attempt {
                family,
                policy: policy.clone(),
                seed,
            }),
        );
        attempt = attempt.in_attempt(attempt_id, family, instance_id);
        self.writer.push(attempt)?;
        Ok(Run::new(
            self,
            Standing {
                attempt: attempt_id,
                family,
                task_instance: instance_id,
                state: state_id,
                head: attempt_id,
            },
            policy.clone(),
            None,
        ))
    }

    /// Branches a new attempt from a decision already made, in the world the
    /// decision was made in, sharing everything before it. The alternative
    /// is recorded against the original in a counterfactual set.
    pub fn fork(&mut self, origin: &DecisionRef, policy: &PolicyRef) -> Result<Run<'_>> {
        let attempt_id = self.writer.next_id();
        let now = self.writer.database().clock().now_ns();
        let mut attempt = Record::new(
            attempt_id,
            now,
            Body::Attempt(Attempt {
                family: origin.family,
                policy: policy.clone(),
                seed: None,
            }),
        );
        attempt = attempt.in_attempt(attempt_id, origin.family, origin.task_instance);
        if let Some(prefix) = origin.parent {
            attempt = attempt.with_parent(prefix);
        }
        self.writer.push(attempt)?;
        let set = self.make(Body::CounterfactualSet(CounterfactualSet {
            origin_state: origin.state,
            origin_decision: origin.id,
            shared_prefix: origin.parent,
        }));
        let set_id = self.writer.push(set)?;
        self.writer
            .link(Edge::new(set_id, Rel::Alternative, origin.id))?;
        let head = origin.parent.unwrap_or(attempt_id);
        Ok(Run::new(
            self,
            Standing {
                attempt: attempt_id,
                family: origin.family,
                task_instance: origin.task_instance,
                state: origin.state,
                head,
            },
            policy.clone(),
            Some((set_id, origin.id)),
        ))
    }
}
