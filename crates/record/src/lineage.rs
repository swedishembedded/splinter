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
use splinter_expdb::model::{DatasetNode, ModelNode, TrainingRun};
use splinter_expdb::{ContentId, RecordId};

use crate::digest::Digest;
use crate::error::StoreError;
use crate::experience::ExperienceId;
use crate::projection::projection_of;
use crate::workspace::{content_id, Workspace};

/// The class of the entity that names a lineage node.
const NODE: &str = "lineage_node";

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
    /// How many attempts, one per experience, the datasets were built from.
    pub attempts: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct NodeRef {
    kind: String,
    id: String,
    record: RecordId,
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
    fn node(&self, kind: &str, id: &str) -> Result<Option<RecordId>, StoreError> {
        let key = node_key(kind, id)?;
        let found = self.read(|s| s.entity(&key))?;
        Ok(match found {
            Some(entity) => Some(
                serde_json::from_value::<NodeRef>(entity.value)
                    .map_err(|e| StoreError::UndecodableObject {
                        what: format!("the {kind} node {id}"),
                        reason: e.to_string(),
                    })?
                    .record,
            ),
            None => None,
        })
    }

    fn name_node(
        s: &mut splinter_expdb::Session,
        key: ContentId,
        kind: &str,
        id: &str,
        record: RecordId,
    ) -> splinter_expdb::Result<()> {
        let value = serde_json::to_value(NodeRef {
            kind: kind.to_owned(),
            id: id.to_owned(),
            record,
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
            Self::name_node(s, key, "dataset", dataset.as_str(), record)
        })
    }

    /// Records the training run that made `candidate` from `datasets`,
    /// continuing `base` when it had one.
    pub fn record_training_run(
        &self,
        candidate: &str,
        datasets: &[Digest],
        objective: &str,
        base: Option<&Digest>,
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
        let base = match base {
            Some(release) => self.node("release", release.as_str())?,
            None => None,
        };
        let key = node_key("candidate", candidate)?;
        let objective = objective.to_owned();
        self.write(|s| {
            let record = s.with_collector(|c| {
                let mut first = None;
                for source in &sources {
                    let run = c.record_training_run(TrainingRun {
                        dataset: *source,
                        objective: objective.clone(),
                        base_model: base,
                    })?;
                    first.get_or_insert(run);
                }
                first.ok_or_else(|| {
                    splinter_expdb::Error::invalid("training run", "it read no dataset")
                })
            })?;
            Self::name_node(s, key, "candidate", candidate, record)
        })
    }

    /// Records the release `release`, named `name`, made by the training of
    /// `candidate` and continuing `parent` when it had one.
    pub fn record_model(
        &self,
        release: &Digest,
        name: &str,
        candidate: &str,
        parent: Option<&Digest>,
    ) -> Result<(), StoreError> {
        if self.node("release", release.as_str())?.is_some() {
            return Ok(());
        }
        let run = self.node("candidate", candidate)?;
        let parent = match parent {
            Some(parent) => self.node("release", parent.as_str())?,
            None => None,
        };
        let key = node_key("release", release.as_str())?;
        let name = name.to_owned();
        self.write(|s| {
            let record = s.with_collector(|c| c.record_model(ModelNode { name, run, parent }))?;
            Self::name_node(s, key, "release", release.as_str(), record)
        })
    }

    /// What `release` was made from: the candidates that trained it, the
    /// datasets they read, the releases it continues and the number of
    /// attempts the datasets were built from. `None` for a release that was
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
            Ok((reached, attempts))
        })?;
        let mut trace = ReleaseTrace {
            attempts: reached.1,
            ..ReleaseTrace::default()
        };
        let mut reached = reached.0;
        reached.sort();
        reached.reverse();
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
