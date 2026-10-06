// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Where a model came from, kept in the experience database: the experience a
//! dataset was projected from, the dataset a training run read, the run a
//! release was made by.
//!
//! A dataset, a candidate and a release are named by the ids the campaign
//! already gives them. Each is recorded once, as a node of the graph that
//! points back at what it was made from, and found again by that id. A dataset
//! also pins the snapshot of the database it was built from, so the
//! experience it read stays readable however much is collected afterwards.

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_expdb::model::{DatasetNode, Entity, ModelNode, TrainingRun};
use splinter_expdb::{ContentId, RecordId};

use crate::documents::encode;
use crate::error::StoreError;
use crate::projection::projection_of;
use crate::workspace::{content_id, put_spilling, Workspace};
use splinter_core::digest::Digest;
use splinter_core::experience::ExperienceId;

/// The class of the entity that names a lineage node.
const NODE: &str = "lineage_node";
const CANDIDATE: &str = "candidate";
const RELEASE: &str = "release";

/// How a dataset came to be, as its builder describes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatasetLineage {
    /// What built it: the view, with the settings that decided what it kept.
    pub recipe: serde_json::Value,
    /// How many records it holds.
    pub records: u64,
}

/// What a release was made from, as far back as the database knows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReleaseTrace {
    /// The candidates whose training led here, newest first.
    pub candidates: Vec<String>,
    /// The datasets they read.
    pub datasets: Vec<Digest>,
    /// The earlier releases this one continues.
    pub earlier_releases: Vec<Digest>,
    /// How many attempts the release traces back to: the datasets' own, one
    /// per experience, and those they were retried, critiqued or revised
    /// from.
    pub attempts: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct NodeRef {
    kind: String,
    id: String,
    record: RecordId,
    /// The other records of a node that is several, such as a candidate trained
    /// on several datasets, which has a run for each.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    also: Vec<RecordId>,
}

fn node_key(kind: &str, id: &str) -> Result<ContentId, StoreError> {
    let mut bytes = NODE.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend_from_slice(kind.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(id.as_bytes());
    content_id(&Digest::of(&bytes))
}

impl Workspace {
    fn node_ref(&self, kind: &str, id: &str) -> Result<Option<NodeRef>, StoreError> {
        let key = node_key(kind, id)?;
        let found = self.read_or_default(|s| s.entity(&key))?;
        found
            .map(|entity| {
                serde_json::from_value::<NodeRef>(entity.value).map_err(|e| {
                    StoreError::UndecodableObject {
                        what: format!("the {kind} node {id}"),
                        reason: e.to_string(),
                    }
                })
            })
            .transpose()
    }

    fn node(&self, kind: &str, id: &str) -> Result<Option<RecordId>, StoreError> {
        Ok(self.node_ref(kind, id)?.map(|n| n.record))
    }

    /// Every record of the node, the first one first.
    fn node_all(&self, kind: &str, id: &str) -> Result<Option<Vec<RecordId>>, StoreError> {
        Ok(self.node_ref(kind, id)?.map(|n| {
            let mut all = vec![n.record];
            all.extend(n.also);
            all
        }))
    }

    fn name_node(
        s: &mut splinter_expdb::Session,
        key: ContentId,
        kind: &str,
        id: &str,
        records: Vec<RecordId>,
    ) -> splinter_expdb::Result<()> {
        let mut records = records.into_iter();
        let record = records
            .next()
            .ok_or_else(|| splinter_expdb::Error::invalid("lineage node", "it has no record"))?;
        let value = serde_json::to_value(NodeRef {
            kind: kind.to_owned(),
            id: id.to_owned(),
            record,
            also: records.collect(),
        })
        .map_err(|source| splinter_expdb::Error::Encode {
            what: "lineage node",
            source,
        })?;
        s.put_entity(&splinter_expdb::model::Entity::keyed(NODE, key, value))
            .map(|_| ())
    }

    /// Records the dataset `dataset`, built from `from`, and pins the database
    /// as it is now under the dataset's name so what it read stays readable.
    /// Recording the same dataset again changes nothing.
    pub fn record_dataset(
        &self,
        dataset: &Digest,
        lineage: &DatasetLineage,
        from: &[ExperienceId],
    ) -> Result<(), StoreError> {
        if self.node("dataset", dataset.as_str())?.is_some() {
            return Ok(());
        }
        let mut attempts = Vec::with_capacity(from.len());
        for id in from {
            attempts.push(projection_of(self, id)?.attempt);
        }
        let key = node_key("dataset", dataset.as_str())?;
        let holder = format!("dataset-{}", dataset.hex());
        self.write(|s| {
            let snapshot = s.snapshot()?;
            snapshot.pin(&holder)?;
            let snapshot = snapshot.id();
            let node = DatasetNode {
                recipe: json!({ "built_by": lineage.recipe, "dataset": dataset }),
                snapshot,
                samples: lineage.records,
            };
            let record = s.with_collector(|c| c.record_dataset(node, &attempts))?;
            Self::name_node(s, key, "dataset", dataset.as_str(), vec![record])
        })
    }

    /// Records the candidate `candidate`: its `document`, and the training that
    /// made it from `datasets`, continuing `parent` when it had one, in one
    /// commit. Every dataset must have been recorded. Returns the document's
    /// address. Recording the same candidate again changes nothing.
    pub fn record_candidate<T: Serialize>(
        &self,
        document: &T,
        candidate: &str,
        datasets: &[Digest],
        objective: &str,
        parent: Option<&Digest>,
    ) -> Result<Digest, StoreError> {
        let (id, entity) = encode(CANDIDATE, document)?;
        self.record_candidate_node(Some(entity), candidate, datasets, objective, parent)?;
        Ok(id)
    }

    /// Records the candidate `candidate` that was trained outside Splinter's
    /// own trainers (a full checkpoint brain trained): the training that made
    /// it from `datasets`, continuing `parent` when it had one, in one commit,
    /// with no candidate document - the release's manifest is its record.
    /// Every dataset must have been recorded. Recording the same candidate
    /// again changes nothing.
    pub fn record_external_candidate(
        &self,
        candidate: &str,
        datasets: &[Digest],
        objective: &str,
        parent: Option<&Digest>,
    ) -> Result<(), StoreError> {
        self.record_candidate_node(None, candidate, datasets, objective, parent)
    }

    fn record_candidate_node(
        &self,
        entity: Option<Entity>,
        candidate: &str,
        datasets: &[Digest],
        objective: &str,
        parent: Option<&Digest>,
    ) -> Result<(), StoreError> {
        if self.node("candidate", candidate)?.is_some() {
            return Ok(());
        }
        let mut sources = Vec::with_capacity(datasets.len());
        for dataset in datasets {
            sources.push(self.node("dataset", dataset.as_str())?.ok_or_else(|| {
                StoreError::Rejected {
                    what: "training run",
                    reason: format!("{dataset} was never recorded as a dataset"),
                }
            })?);
        }
        if sources.is_empty() {
            return Err(StoreError::Rejected {
                what: "training run",
                reason: "it read no dataset".into(),
            });
        }
        let base = match parent {
            Some(release) => Some(self.node("release", release.as_str())?.ok_or_else(|| {
                StoreError::Rejected {
                    what: "training run",
                    reason: format!("{release} was never recorded as a release"),
                }
            })?),
            None => None,
        };
        let key = node_key("candidate", candidate)?;
        let objective = objective.to_owned();
        self.write(|s| {
            if let Some(entity) = &entity {
                put_spilling(s, entity.clone())?;
            }
            let runs = s.with_collector(|c| {
                sources
                    .iter()
                    .map(|source| {
                        c.record_training_run(TrainingRun {
                            dataset: *source,
                            objective: objective.clone(),
                            base_model: base,
                        })
                    })
                    .collect::<splinter_expdb::Result<Vec<_>>>()
            })?;
            Self::name_node(s, key, "candidate", candidate, runs)
        })
    }

    /// Records the release whose manifest is `document`, named `name`, made by
    /// the training of `candidate` and continuing `parent` when it had one, in
    /// one commit. The candidate and the parent must have been recorded.
    /// Returns the document's address, which is the release's id. Recording the
    /// same release again changes nothing.
    pub fn record_release<T: Serialize>(
        &self,
        document: &T,
        name: &str,
        candidate: &str,
        parent: Option<&Digest>,
    ) -> Result<Digest, StoreError> {
        let (id, entity) = encode(RELEASE, document)?;
        if self.node("release", id.as_str())?.is_some() {
            return Ok(id);
        }
        let runs = self
            .node_all("candidate", candidate)?
            .ok_or_else(|| StoreError::Rejected {
                what: "release",
                reason: format!("candidate {candidate} was never recorded"),
            })?;
        let parent = match parent {
            Some(parent) => Some(self.node("release", parent.as_str())?.ok_or_else(|| {
                StoreError::Rejected {
                    what: "release",
                    reason: format!("the release {parent} it continues was never recorded"),
                }
            })?),
            None => None,
        };
        let key = node_key("release", id.as_str())?;
        let name = name.to_owned();
        let node_id = id.clone();
        self.write(|s| {
            put_spilling(s, entity.clone())?;
            let model = s.with_collector(|c| {
                c.record_model(ModelNode {
                    name: name.clone(),
                    run: runs.first().copied(),
                    parent,
                })
            })?;
            // A candidate trained on several datasets has one run per
            // dataset, and the model came from all of them.
            for run in runs.iter().skip(1) {
                s.link_records(model, splinter_expdb::model::Rel::ProducedBy, *run)?;
            }
            Self::name_node(s, key, "release", node_id.as_str(), vec![model])
        })?;
        Ok(id)
    }

    /// What `release` was made from: the candidates that trained it, the
    /// datasets they read, the releases it continues and the number of
    /// attempts it traces back to. `None` for a release that was
    /// never recorded.
    pub fn trace_release(&self, release: &Digest) -> Result<Option<ReleaseTrace>, StoreError> {
        let Some(start) = self.node("release", release.as_str())? else {
            return Ok(None);
        };
        self.refresh()?;
        let names: std::collections::HashMap<RecordId, NodeRef> = self
            .read(|s| s.entities(NODE))?
            .into_iter()
            .filter_map(|stored| serde_json::from_value::<NodeRef>(stored.entity.value).ok())
            .map(|node| (node.record, node))
            .collect();
        let reached = self.read(|s| {
            let index = s.snapshot()?.index()?;
            let reached = index.reachable_from(start, None);
            let attempts = reached
                .iter()
                .filter(|id| {
                    index.kind_of(**id) == Some(splinter_expdb::model::RecordKind::Attempt)
                })
                .count();
            // Newest first by the time on the records, which every reader
            // agrees on, and not by the random ids of the writers.
            let mut stamped: Vec<(u64, RecordId)> = reached
                .into_iter()
                .map(|id| (index.timestamp_ns(id).unwrap_or(0), id))
                .collect();
            stamped.sort_unstable_by(|a, b| b.cmp(a));
            Ok((
                stamped.into_iter().map(|(_, id)| id).collect::<Vec<_>>(),
                attempts,
            ))
        })?;
        let mut trace = ReleaseTrace {
            attempts: reached.1,
            ..ReleaseTrace::default()
        };
        let reached = reached.0;
        for id in reached {
            let Some(node) = names.get(&id) else { continue };
            match node.kind.as_str() {
                "candidate" => trace.candidates.push(node.id.clone()),
                "dataset" => {
                    if let Ok(digest) = Digest::parse(&node.id) {
                        trace.datasets.push(digest);
                    }
                }
                "release" if id != start => {
                    if let Ok(digest) = Digest::parse(&node.id) {
                        trace.earlier_releases.push(digest);
                    }
                }
                _ => {}
            }
        }
        trace.datasets.sort();
        trace.datasets.dedup();
        Ok(Some(trace))
    }
}
