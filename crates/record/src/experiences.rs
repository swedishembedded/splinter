// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The experience store: immutable experiences under their content
//! address, the annotations written about them, and named sets of ids, all
//! kept in the experience database.
//!
//! An experience is an entity keyed by its content address, so the id a
//! caller holds is the id the database finds it by, and [`ExperienceStore::get`]
//! checks on every read that what comes back still hashes to it. An
//! annotation is an evaluation of that entity (a verdict also carries its
//! ranked ruling, a relation also an edge to the other experience), with the
//! whole annotation kept as its evidence so it reads back exactly as written.
//! Nothing is ever rewritten: re-grading is a new evaluation.

use serde::{Deserialize, Serialize};
use splinter_expdb::analyze::{EvalFilter, TASK_COMPLETION};
use splinter_expdb::model::{Entity, Evaluation, EvaluatorRef, Target};

use crate::annotation::{Annotation, AnnotationBody, Label, Outcome};
use crate::digest::{canonical_json, Digest};
pub use crate::error::StoreError;
use crate::experience::{Experience, ExperienceId};
use crate::projection::{project, projection_address, projection_of, Projected, PROJECTION};
use crate::workspace::{content_id, Workspace};

const EXPERIENCE: &str = "experience";
const EXPERIENCE_SET: &str = "experience_set";

/// The content address of an [`ExperienceSet`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SetId(pub Digest);

impl std::fmt::Display for SetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A named, ordered list of experiences: how a pipeline stage names its
/// input. Content-addressed like an experience, name included.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperienceSet {
    /// What the set is, for a reader.
    pub name: String,
    /// The experiences, in order; each at most once.
    pub members: Vec<ExperienceId>,
}

/// An experience's annotations.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnnotationLog {
    /// The annotations, in the order they were written.
    pub annotations: Vec<Annotation>,
}

/// The experience store: immutable experiences under their content address,
/// the annotations written about them, and named sets of experience ids.
#[derive(Clone, Debug)]
pub struct ExperienceStore {
    workspace: Workspace,
}

fn undecodable(what: String, reason: impl ToString) -> StoreError {
    StoreError::UndecodableObject {
        what,
        reason: reason.to_string(),
    }
}

impl ExperienceStore {
    /// The store over `workspace`.
    #[must_use]
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            workspace: workspace.clone(),
        }
    }

    /// Stores `experience` and returns its id. Storing the same content
    /// again writes nothing and returns the same id; one that no longer
    /// matches its address is reported as altered rather than replaced.
    pub fn put(&self, experience: &Experience) -> Result<ExperienceId, StoreError> {
        experience.validate()?;
        let bytes = experience.canonical()?;
        let id = ExperienceId(Digest::of(&bytes));
        if self.workspace.has(EXPERIENCE, &id.0)? {
            self.get(&id)?;
            return Ok(id);
        }
        let value = serde_json::to_value(experience).map_err(|source| StoreError::Serialize {
            what: "experience",
            source,
        })?;
        let entity = Entity::keyed(EXPERIENCE, content_id(&id.0)?, value);
        let projection_key = content_id(&projection_address(&id))?;
        self.workspace.write(|s| {
            s.put_entity(&entity)?;
            let projected = s.with_collector(|c| project(c, &id, experience))?;
            let value = serde_json::to_value(&projected).map_err(|source| {
                splinter_expdb::Error::Encode {
                    what: "projection",
                    source,
                }
            })?;
            s.put_entity(&Entity::keyed(PROJECTION, projection_key, value))
                .map(|_| ())
        })?;
        Ok(id)
    }

    /// Whether the store holds `id` (without verifying its content).
    pub fn contains(&self, id: &ExperienceId) -> Result<bool, StoreError> {
        self.workspace.has(EXPERIENCE, &id.0)
    }

    /// The experience stored under `id`, verified against its address.
    pub fn get(&self, id: &ExperienceId) -> Result<Experience, StoreError> {
        let entity = self
            .workspace
            .find(EXPERIENCE, &id.0)?
            .ok_or_else(|| StoreError::UnknownExperience(id.clone()))?;
        let experience: Experience = serde_json::from_value(entity.value)
            .map_err(|e| undecodable(format!("experience {id}"), e))?;
        // The address must be the address of the value handed back: a field
        // this build does not know would be dropped on decode, and the caller
        // would hold a different record.
        let found = Digest::of(&experience.canonical()?);
        if found != id.0 {
            return Err(StoreError::Altered {
                what: format!("experience {id}"),
                expected: id.0.clone(),
                found,
            });
        }
        Ok(experience)
    }

    /// Records `note` about its experience. Refused for an experience the
    /// store does not hold (or a relation to one). Durable once this
    /// returns, unless the workspace is in a batch.
    pub fn annotate(&self, note: &Annotation) -> Result<(), StoreError> {
        let rejected = |reason: &str| StoreError::Rejected {
            what: "annotation",
            reason: reason.to_string(),
        };
        if note.producer.name.trim().is_empty() || note.producer.version.trim().is_empty() {
            return Err(rejected("the producer's name and version are required"));
        }
        if !self.contains(&note.experience)? {
            return Err(StoreError::UnknownExperience(note.experience.clone()));
        }
        if let AnnotationBody::Relation { other, .. } = &note.body {
            if *other == note.experience {
                return Err(rejected("an experience cannot be related to itself"));
            }
            if !self.contains(other)? {
                return Err(StoreError::UnknownExperience(other.clone()));
            }
        }
        let json = serde_json::to_vec(note).map_err(|source| StoreError::Serialize {
            what: "annotation",
            source,
        })?;
        let projected = self.projected(&note.experience)?;
        let (criterion, score, confidence, verdict) = describe(&note.body);
        let (target, criterion) = match &note.body {
            AnnotationBody::StepLabel { step, .. } => match projected.decisions.get(step) {
                Some(decision) => (*decision, criterion),
                None => (projected.attempt, format!("{criterion}:{step}")),
            },
            _ => (projected.attempt, criterion),
        };
        let related = match &note.body {
            AnnotationBody::Relation { kind, other } => {
                Some((kind.edge(), self.projected(other)?.attempt))
            }
            _ => None,
        };
        let producer = EvaluatorRef::new(&note.producer.name, &note.producer.version);
        self.workspace.write(|s| {
            let mut evaluation = Evaluation::new(
                Target::Record(target),
                producer,
                &criterion,
                score,
                confidence,
            );
            evaluation.evidence = Some(s.put_blob(&json)?);
            evaluation.verdict = verdict;
            s.evaluate(evaluation)?;
            if let Some((rel, other)) = related {
                s.link_records(projected.attempt, rel, other)?;
            }
            Ok(())
        })
    }

    fn projected(&self, id: &ExperienceId) -> Result<Projected, StoreError> {
        projection_of(&self.workspace, id)
    }

    /// `id`'s annotations in the order they were written.
    pub fn annotations(&self, id: &ExperienceId) -> Result<AnnotationLog, StoreError> {
        if !self.contains(id)? {
            return Err(StoreError::UnknownExperience(id.clone()));
        }
        let projected = self.projected(id)?;
        let mut targets = vec![projected.attempt];
        targets.extend(projected.decisions.values().copied());
        let raw = self.workspace.read(|s| {
            let mut views = Vec::new();
            for target in &targets {
                views.extend(s.evaluations(&EvalFilter::new().target(Target::Record(*target)))?);
            }
            views.sort_by_key(|view| view.id);
            let mut raw = Vec::new();
            for view in views {
                if let Some(blob) = view.evaluation.evidence {
                    raw.push(s.read_blob(&blob.id)?);
                }
            }
            Ok(raw)
        })?;
        let mut log = AnnotationLog::default();
        for bytes in raw {
            let note: Annotation = serde_json::from_slice(&bytes)
                .map_err(|e| undecodable(format!("an annotation of {id}"), e))?;
            if note.experience != *id {
                return Err(undecodable(
                    format!("an annotation of {id}"),
                    format!("it is about {}", note.experience),
                ));
            }
            log.annotations.push(note);
        }
        Ok(log)
    }

    /// Stores `set` and returns its id; write-once like [`Self::put`].
    /// Refused when a member is unknown or listed twice.
    pub fn put_set(&self, set: &ExperienceSet) -> Result<SetId, StoreError> {
        let rejected = |reason: String| StoreError::Rejected {
            what: "experience set",
            reason,
        };
        if set.name.trim().is_empty() {
            return Err(rejected("the name is empty".into()));
        }
        let mut seen = std::collections::HashSet::new();
        for member in &set.members {
            if !seen.insert(member) {
                return Err(rejected(format!("{member} is listed twice")));
            }
            if !self.contains(member)? {
                return Err(StoreError::UnknownExperience(member.clone()));
            }
        }
        let bytes = canonical_json(set).map_err(|source| StoreError::Serialize {
            what: "experience set",
            source,
        })?;
        let id = SetId(Digest::of(&bytes));
        if !self.workspace.has(EXPERIENCE_SET, &id.0)? {
            let value = serde_json::to_value(set).map_err(|source| StoreError::Serialize {
                what: "experience set",
                source,
            })?;
            let entity = Entity::keyed(EXPERIENCE_SET, content_id(&id.0)?, value);
            self.workspace.write(|s| s.put_entity(&entity))?;
        }
        Ok(id)
    }

    /// The set stored under `id`, verified against its address.
    pub fn get_set(&self, id: &SetId) -> Result<ExperienceSet, StoreError> {
        let entity = self
            .workspace
            .find(EXPERIENCE_SET, &id.0)?
            .ok_or_else(|| StoreError::UnknownSet(id.clone()))?;
        let set: ExperienceSet = serde_json::from_value(entity.value)
            .map_err(|e| undecodable(format!("experience set {id}"), e))?;
        let bytes = canonical_json(&set).map_err(|source| StoreError::Serialize {
            what: "experience set",
            source,
        })?;
        let found = Digest::of(&bytes);
        if found != id.0 {
            return Err(StoreError::Altered {
                what: format!("experience set {id}"),
                expected: id.0.clone(),
                found,
            });
        }
        Ok(set)
    }

    /// Every stored experience's id, in id order (without verifying them).
    pub fn list(&self) -> Result<Vec<ExperienceId>, StoreError> {
        Ok(self
            .workspace
            .ids_of(EXPERIENCE)?
            .into_iter()
            .map(ExperienceId)
            .collect())
    }

    /// Every stored set's id, in id order (without verifying them).
    pub fn list_sets(&self) -> Result<Vec<SetId>, StoreError> {
        Ok(self
            .workspace
            .ids_of(EXPERIENCE_SET)?
            .into_iter()
            .map(SetId)
            .collect())
    }
}

/// How an annotation is recorded as an evaluation: its criterion, a score,
/// how much the score is worth, and the ranked verdict when it is one.
fn describe(body: &AnnotationBody) -> (String, f64, f64, Option<splinter_expdb::model::Verdict>) {
    use splinter_expdb::model::{Ruling, Verdict};
    match body {
        AnnotationBody::Verdict {
            outcome, strength, ..
        } => {
            let (ruling, score, confidence) = match outcome {
                Outcome::Pass => (Ruling::Pass, 1.0, 1.0),
                Outcome::Fail => (Ruling::Fail, 0.0, 1.0),
                Outcome::Abstain => (Ruling::Abstain, 0.0, 0.0),
            };
            (
                TASK_COMPLETION.to_owned(),
                score,
                confidence,
                Some(Verdict::new(ruling, strength.rank())),
            )
        }
        AnnotationBody::StepLabel { label, .. } => {
            let score = match label {
                Label::Good => 1.0,
                Label::Bad => 0.0,
                Label::Neutral => 0.5,
            };
            ("step_label".to_owned(), score, 1.0, None)
        }
        AnnotationBody::Relation { kind, .. } => {
            let name = serde_json::to_value(kind)
                .ok()
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_default();
            (format!("relation:{name}"), 1.0, 1.0, None)
        }
    }
}
