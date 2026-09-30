// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that projects
// verified agent experience into supervised datasets, for its clients. If
// your team needs expertise in dataset curation for fine-tuning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Views: how a training set is projected from the experience store.
//!
//! A [`View`] reads one experience and its annotations and returns the
//! training records it yields for one [`Objective`] - none when the
//! annotations do not qualify it. Views never write to the store: re-grading
//! an experience (a new annotation) changes what a view yields, while the
//! experience stays as it was.
//!
//! * [`SftFinal`] - a passed experience as one chat record: the instruction
//!   as the user turn, the final output as the only supervised turn.
//! * [`write_dataset`] - records written as the `generic-messages-v2` chat
//!   dataset brain trains on, checked by brain's own parser before the
//!   file is reported written.
//!
//! No view may put privileged information (what only the teacher saw) into
//! a record: the student is trained on exactly what it will be shown.

#![warn(missing_docs)]

mod dataset;
mod sft;

use serde::{Deserialize, Serialize};
use splinter_lab::WireMessage;
use splinter_store::annotation::Annotation;
use splinter_store::experience::{Experience, ExperienceError, ExperienceId};

pub use dataset::{write_dataset, Dataset};
pub use sft::SftFinal;

/// The training objective a view's records serve. Only supervised
/// fine-tuning is implemented; an objective that needs a different record
/// shape (a preference pair, a rewarded rollout) is added as a variant here
/// together with the view that produces it and the writer for its format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Supervised fine-tuning on chat records.
    Sft,
}

/// One training record: a conversation in `generic-messages-v2` messages,
/// with the metadata that traces it back to its experience.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    /// The conversation; `train` marks the supervised turns.
    pub messages: Vec<WireMessage>,
    /// Where the record came from; carried in the dataset, never rendered
    /// into the prompt.
    pub metadata: RecordMetadata,
}

/// Where a record came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordMetadata {
    /// The experience it was projected from.
    pub experience: ExperienceId,
    /// The view that projected it.
    pub view: String,
    /// The objective it serves.
    pub objective: Objective,
}

/// A projection of experiences into training records.
pub trait View {
    /// The view's name, recorded in every record it yields.
    fn name(&self) -> &str;
    /// The objective its records serve.
    fn objective(&self) -> Objective;
    /// The records `experience` yields given `notes`, its annotations.
    fn project(
        &self,
        experience: &Experience,
        notes: &[Annotation],
    ) -> Result<Vec<Record>, ViewError>;
}

/// Why a view or the dataset writer failed.
#[derive(Debug, thiserror::Error)]
pub enum ViewError {
    /// An annotation passed with an experience is about another one.
    #[error("an annotation of {annotation_of} was passed with experience {experience}")]
    ForeignAnnotation {
        /// The experience being projected.
        experience: ExperienceId,
        /// The experience the annotation is about.
        annotation_of: ExperienceId,
    },
    /// The experience's id cannot be computed.
    #[error(transparent)]
    Experience(#[from] ExperienceError),
    /// There are no records to write; a dataset that trains on nothing is
    /// a producer bug, not a dataset.
    #[error("no records to write")]
    Empty,
    /// A record cannot be serialized.
    #[error("cannot serialize a record: {0}")]
    Serialize(#[from] serde_json::Error),
    /// Brain's parser refused the dataset.
    #[error("{path} is not a valid chat dataset: {reason}")]
    Invalid {
        /// The dataset file.
        path: std::path::PathBuf,
        /// The parser's error.
        reason: String,
    },
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: std::path::PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
}

/// Refuses `notes` unless every one is about `experience`; returns its id.
fn own_notes(experience: &Experience, notes: &[Annotation]) -> Result<ExperienceId, ViewError> {
    let id = experience.id()?;
    if let Some(foreign) = notes.iter().find(|n| n.experience != id) {
        return Err(ViewError::ForeignAnnotation {
            experience: id,
            annotation_of: foreign.experience.clone(),
        });
    }
    Ok(id)
}
