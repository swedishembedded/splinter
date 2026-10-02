// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Turning references into text, only when a sample is actually used.

use std::collections::HashMap;
use std::sync::Arc;

use super::plan::DataRef;
use crate::blob::BlobStore;
use crate::error::{Error, Result};
use crate::id::{ContentId, RecordId};
use crate::index::locked;
use crate::manifest::Snapshot;
use crate::model::{Body, Content, Decision, Record, RecordKind};

impl Snapshot {
    pub(crate) fn blob_store(&self) -> Result<Arc<BlobStore>> {
        if let Some(found) = locked(&self.cache().blobs).as_ref() {
            return Ok(Arc::clone(found));
        }
        let packs: Vec<ContentId> = self.blob_packs().iter().map(|p| p.id).collect();
        let store = Arc::new(BlobStore::with_packs(self.database(), &packs)?);
        *locked(&self.cache().blobs) = Some(Arc::clone(&store));
        Ok(store)
    }

    pub(crate) fn entity(&self, id: &ContentId) -> Result<Option<RecordId>> {
        if let Some(found) = locked(&self.cache().entities).as_ref() {
            return Ok(found.get(id).copied());
        }
        let index = self.index()?;
        let mut map = HashMap::new();
        for kind in [
            RecordKind::TaskDefinition,
            RecordKind::TaskInstance,
            RecordKind::ModalitySchema,
            RecordKind::ClockDomain,
        ] {
            for record_id in index.by_kind(kind) {
                if let Some(record) = self.get(record_id)? {
                    if let Some(entity) = record.body.entity_id()? {
                        map.entry(entity).or_insert(record_id);
                    }
                }
            }
        }
        let map = Arc::new(map);
        *locked(&self.cache().entities) = Some(Arc::clone(&map));
        Ok(map.get(id).copied())
    }

    fn content_text(&self, content: &Content) -> Result<String> {
        match content {
            Content::Text { text } => Ok(text.clone()),
            Content::Blob { blob } => {
                Ok(String::from_utf8_lossy(&self.blob_store()?.get(blob)?).into_owned())
            }
        }
    }

    fn record(&self, id: RecordId) -> Result<Record> {
        self.get(id)?.ok_or_else(|| Error::NotFound {
            what: format!("record {id}"),
        })
    }

    fn decision(&self, id: RecordId) -> Result<(Record, Decision)> {
        let record = self.record(id)?;
        match record.body.clone() {
            Body::Decision(d) => Ok((record, d)),
            _ => Err(Error::invalid(
                "data reference",
                format!("{id} is not a decision"),
            )),
        }
    }

    fn task_text(&self, instance: &ContentId) -> Result<String> {
        let Some(instance_record) = self.entity(instance)? else {
            return Err(Error::NotFound {
                what: format!("task instance {instance}"),
            });
        };
        let Body::TaskInstance(task) = self.record(instance_record)?.body else {
            return Err(Error::corrupt(
                format!("record {instance_record}"),
                "not a task instance",
            ));
        };
        let description = match self.entity(&task.definition)? {
            Some(definition) => match self.record(definition)?.body {
                Body::TaskDefinition(d) => d.description,
                _ => String::new(),
            },
            None => String::new(),
        };
        Ok(format!("[task] {description} {}", task.params))
    }

    fn action_text(decision: &Decision) -> String {
        if decision.action.arguments.is_null() {
            decision.action.name.clone()
        } else {
            format!("{} {}", decision.action.name, decision.action.arguments)
        }
    }

    /// The line a record contributes to a history, if it contributes one.
    fn history_line(&self, record: &Record) -> Result<Option<String>> {
        Ok(match &record.body {
            Body::Observation(o) => {
                Some(format!("[observation] {}", self.content_text(&o.content)?))
            }
            Body::Decision(d) => Some(format!("[action] {}", Self::action_text(d))),
            _ => None,
        })
    }

    /// The records from the start of a path to `from`, inclusive.
    fn path_to(&self, from: Option<RecordId>) -> Result<Vec<Record>> {
        let index = self.index()?;
        let mut path = Vec::new();
        let mut at = from;
        while let Some(id) = at {
            path.push(self.record(id)?);
            at = index.parent(id);
        }
        path.reverse();
        Ok(path)
    }

    /// The text a reference stands for.
    pub fn render(&self, data: &DataRef) -> Result<String> {
        match data {
            DataRef::Task { instance } => self.task_text(instance),
            DataRef::Action { decision } => Ok(Self::action_text(&self.decision(*decision)?.1)),
            DataRef::Observation { record } => match self.record(*record)?.body {
                Body::Observation(o) => self.content_text(&o.content),
                _ => Err(Error::invalid(
                    "data reference",
                    format!("{record} is not an observation"),
                )),
            },
            DataRef::Blob { blob } => {
                Ok(String::from_utf8_lossy(&self.blob_store()?.get(blob)?).into_owned())
            }
            DataRef::Window { .. }
            | DataRef::ActionSegment { .. }
            | DataRef::EventPayload { .. } => Err(Error::invalid(
                "data reference",
                "it is not text; read it as a span",
            )),
            DataRef::Context { decision } => {
                let (record, d) = self.decision(*decision)?;
                let mut lines = Vec::new();
                if let Some(instance) = record.task_instance {
                    lines.push(self.task_text(&instance)?);
                }
                if let Some(context) = &d.context {
                    lines.push(format!(
                        "[context] {}",
                        String::from_utf8_lossy(&self.blob_store()?.get(context)?)
                    ));
                }
                let history = self.path_to(record.parent)?;
                for earlier in &history {
                    lines.extend(self.history_line(earlier)?);
                }
                if let Some(seen) = d
                    .observation
                    .filter(|o| !history.iter().any(|r| r.id == *o))
                {
                    lines.extend(self.history_line(&self.record(seen)?)?);
                }
                Ok(lines.join("\n"))
            }
            DataRef::Episode { attempt } => {
                let index = self.index()?;
                let members = index.by_attempt(*attempt);
                // The end of the attempt, or failing that the last step of its
                // path. An evaluation can be the newest record of an attempt
                // that never finished, but it is not part of the path.
                let on_path = |id: &&RecordId| {
                    matches!(
                        index.kind_of(**id),
                        Some(
                            RecordKind::Observation | RecordKind::Decision | RecordKind::Transition
                        )
                    )
                };
                let end = members
                    .iter()
                    .find(|id| index.kind_of(**id) == Some(RecordKind::AttemptEnd))
                    .or_else(|| members.iter().rfind(on_path))
                    .copied();
                let mut lines = Vec::new();
                for record in self.path_to(end)? {
                    lines.extend(self.history_line(&record)?);
                }
                Ok(lines.join("\n"))
            }
        }
    }

    /// Reads the blocks holding `ids` in storage order, so a window of
    /// shuffled samples costs each block once.
    pub(crate) fn prefetch(&self, ids: &[RecordId]) -> Result<()> {
        let index = self.index()?;
        let mut places: Vec<_> = ids
            .iter()
            .filter_map(|id| index.loc(*id))
            .map(|l| (l.segment, l.block))
            .collect();
        places.sort_unstable();
        places.dedup();
        for (segment, block) in places {
            self.block(segment, block)?;
        }
        Ok(())
    }
}
