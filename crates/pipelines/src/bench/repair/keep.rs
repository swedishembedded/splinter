// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! What a repair run keeps: each graded repair's record, an accepted one's
//! experience with the verifier's pass and its relation to the student's
//! attempt, and the diagnosed class on the attempt.

use std::collections::BTreeMap;

use splinter_core::annotation::{
    Annotation, AnnotationBody, Outcome, Producer, RelationKind, Strength,
};
use splinter_core::experience::{Experience, ExperienceId};
use splinter_core::outcome::AttemptReport;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::super::VERIFIER_PRODUCER;
use super::{RepairOutcome, RepairRecord, PRODUCER, RECORD};

/// Every repair record the state root holds, by key.
pub(super) fn held_records(
    ctx: &Context,
) -> Result<BTreeMap<String, RepairRecord>, OrchestratorError> {
    let mut held = BTreeMap::new();
    for id in ctx.workspace().document_ids(RECORD)? {
        if let Some(record) = ctx.workspace().get_document::<RepairRecord>(RECORD, &id)? {
            // A graded record decides; a refused one only stands in for none.
            let graded = held
                .get(&record.key)
                .is_some_and(|r: &RepairRecord| r.outcome != RepairOutcome::Refused);
            if !graded {
                held.insert(record.key.clone(), record);
            }
        }
    }
    Ok(held)
}

/// Keeps one graded repair: the accepted experience with its annotations,
/// the class on the student's attempt, and the record; the record and the
/// experience's id.
pub(super) fn keep(
    ctx: &Context,
    mut record: RepairRecord,
    experience: Option<&Experience>,
) -> Result<(RepairRecord, Option<ExperienceId>), OrchestratorError> {
    let id = match experience {
        Some(experience) => {
            let id = keep_accepted(ctx, &record, experience)?;
            record.experience = Some(id.clone());
            Some(id)
        }
        None => None,
    };
    annotate_failure(ctx, &record)?;
    ctx.workspace().put_document(RECORD, &record)?;
    Ok((record, id))
}

/// Stores an accepted repair with the verifier's pass and its relation to
/// the student's attempt.
fn keep_accepted(
    ctx: &Context,
    record: &RepairRecord,
    experience: &Experience,
) -> Result<ExperienceId, OrchestratorError> {
    let store = ctx.experiences();
    let id = store.put(experience)?;
    let verifier = store
        .annotations(&record.student)?
        .annotations
        .into_iter()
        .find(|a| a.producer.name == VERIFIER_PRODUCER)
        .map(|a| a.producer)
        .unwrap_or_else(|| Producer {
            name: VERIFIER_PRODUCER.into(),
            version: record
                .grade
                .as_ref()
                .map(|g| g.digest.clone())
                .unwrap_or_default(),
        });
    store.annotate(&Annotation {
        experience: id.clone(),
        producer: verifier,
        body: AnnotationBody::Verdict {
            outcome: Outcome::Pass,
            strength: Strength::Executable,
            evidence: serde_json::json!({ "grade": record.grade, "repair": record.key }),
        },
    })?;
    let producer = Producer {
        name: PRODUCER.into(),
        version: splinter_eval::diagnosis::RULES_VERSION.into(),
    };
    // `RevisionOf` and not `PreferredOver`: the preference view finds a
    // repair by its copied prefix and pairs it with the failed attempt that
    // prefix is, and reads `PreferredOver` as a pair of final answers, which an
    // agent's trial does not have.
    store.annotate(&Annotation {
        experience: id.clone(),
        producer,
        body: AnnotationBody::Relation {
            kind: RelationKind::RevisionOf,
            other: record.student.clone(),
        },
    })?;
    Ok(id)
}

/// Records the diagnosed class on the student's attempt, once.
fn annotate_failure(ctx: &Context, record: &RepairRecord) -> Result<(), OrchestratorError> {
    let store = ctx.experiences();
    let producer = Producer {
        name: splinter_eval::diagnosis::PRODUCER.into(),
        version: splinter_eval::diagnosis::RULES_VERSION.into(),
    };
    let annotated = store
        .annotations(&record.student)?
        .annotations
        .iter()
        .any(|a| a.producer == producer);
    if !annotated {
        store.annotate(&Annotation {
            experience: record.student.clone(),
            producer,
            body: AnnotationBody::AttemptReport(AttemptReport {
                failure: Some(record.diagnosis.class),
                ..AttemptReport::default()
            }),
        })?;
    }
    Ok(())
}
