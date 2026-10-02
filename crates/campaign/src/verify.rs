// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade agent work by the
// strongest check available, for its clients. If your team needs expertise
// in verifier design or reward modelling, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The verify stage: verdicts appended to an experience set's experiences,
//! without re-running anything.
//!
//! Which verifiers grade an experience is its task kind's decision (the
//! kind's `verifiers` list): a formal check of a task whose kind only a
//! judge can grade would fail right answers at a strength that outranks
//! the judge. The judged verifier runs only when a judge is named, and only
//! gated by that judge's stored calibration: a verdict stands only where the
//! judge was measured precise.

use std::time::Duration;

use serde::Serialize;
use splinter_agent::judge::JudgeVerifier;
use splinter_agent::solve::Model;
use splinter_knowledge::tasks::{Catalogue, VerifierKind};
use splinter_lab::denoise::{FormalVerifier, KIND as DENOISE_KIND};
use splinter_lab::verifiers::calibration::{CalibratedJudge, Calibration, DEFAULT_MIN_PRECISION};
use splinter_lab::verifiers::consistency::AgreementVerifier;
use splinter_lab::verifiers::executable::ExecutableVerifier;
use splinter_lab::verifiers::formal::{ExactMatchVerifier, StatedReferenceVerifier};
use splinter_lab::verifiers::mutation::{MutationPolicy, MutationValidatedVerifier};
use splinter_lab::verifiers::normalise::Normalisation;
use splinter_lab::verifiers::{verify_and_annotate, Strongest, Verifier};
use splinter_record::annotation::{decide, AnnotationBody, Outcome, Producer};
use splinter_record::digest::Digest;
use splinter_record::experience::{Experience, ExperienceId, Task};
use splinter_record::experiences::SetId;
use sven_sdk::CancelToken;

use crate::context::Context;
use crate::error::CampaignError;
use crate::model_ref::ModelRef;

/// The producer of the formal verifier's verdicts on generated tasks.
pub const FORMAL_PRODUCER: &str = "splinter-lab/exact-match:lenient";

/// The formal verifier's version: bumped with its normalisation.
pub const FORMAL_VERSION: &str = "1";

/// The producer of the stated-reference verifier's verdicts.
pub const STATED_PRODUCER: &str = "splinter-lab/reference-stated:lenient";

/// The stated-reference verifier's version: bumped with its normalisation
/// or its bound on the answer's length.
pub const STATED_VERSION: &str = "1";

/// How long a judge may take over one answer.
pub const DEFAULT_JUDGE_DEADLINE: Duration = Duration::from_secs(120);

const CALIBRATION: &str = "calibration";

/// A judge and the calibration its verdicts are gated by.
pub struct Judge {
    model: Model,
    calibration: Calibration,
}

impl Judge {
    /// The judge `reference` names, with its stored calibration; refused
    /// when it was never calibrated.
    pub fn load(ctx: &Context, reference: &ModelRef) -> Result<Self, CampaignError> {
        let model = ctx.model(reference)?;
        let producer = judge_verifier(ctx, &model).producer();
        ctx.workspace().refresh()?;
        let calibration = latest_calibration(ctx, &producer)?.ok_or_else(|| {
            CampaignError::Refused(format!(
                "judge {reference} has no calibration; its verdicts would stand unmeasured. \
                 Run `splinter judge calibrate <LABELLED-FILE> --judge {reference}` first"
            ))
        })?;
        Ok(Self { model, calibration })
    }

    fn verifier(&self, ctx: &Context) -> Result<Box<dyn Verifier>, CampaignError> {
        Ok(Box::new(CalibratedJudge::new(
            judge_verifier(ctx, &self.model),
            self.calibration.clone(),
            DEFAULT_MIN_PRECISION,
        )?))
    }
}

/// The judged verifier on `model`.
pub(crate) fn judge_verifier(ctx: &Context, model: &Model) -> JudgeVerifier {
    JudgeVerifier::new(model.clone(), ctx.handle(), DEFAULT_JUDGE_DEADLINE)
}

/// The pointer that names the latest calibration of the judge `producer`.
fn calibration_pointer(producer: &Producer) -> String {
    let name: String = format!("{}@{}", producer.name, producer.version)
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("calibration-{name}")
}

/// The latest calibration of the judge `producer`, if it has one.
fn latest_calibration(
    ctx: &Context,
    producer: &Producer,
) -> Result<Option<Calibration>, CampaignError> {
    let Some((_, value)) = ctx.workspace().pointer(&calibration_pointer(producer))? else {
        return Ok(None);
    };
    let digest = Digest::parse(&value)
        .map_err(|e| CampaignError::Refused(format!("a calibration pointer is corrupt: {e}")))?;
    Ok(ctx.workspace().get_document(CALIBRATION, &digest)?)
}

/// Records `calibration` as its judge's latest, keeping the ones before it,
/// and returns its address.
pub(crate) fn store_calibration(
    ctx: &Context,
    calibration: &Calibration,
) -> Result<Digest, CampaignError> {
    let digest = ctx.workspace().put_document(CALIBRATION, calibration)?;
    let pointer = calibration_pointer(&calibration.producer);
    let current = ctx.workspace().pointer(&pointer)?.map(|(_, value)| value);
    if current.as_deref() != Some(digest.as_str()) {
        ctx.workspace().move_pointer(
            &pointer,
            current.as_deref(),
            digest.as_str(),
            &ctx.clock().utc_now(),
        )?;
    }
    Ok(digest)
}

/// The verifiers that grade answers to `task`: its kind's, in the
/// environments it runs code in, with `pool` as the other answers
/// agreement is counted over. Refused for a kind with no known verifiers.
pub(crate) fn verifiers_for(
    ctx: &Context,
    task: &Task,
    pool: &[Experience],
    judge: Option<&Judge>,
) -> Result<Strongest, CampaignError> {
    let kind = &task.task.kind;
    if kind == DENOISE_KIND {
        return Ok(Strongest::new(vec![Box::new(FormalVerifier::new())]));
    }
    let catalogue = Catalogue::builtin();
    let Some(spec) = catalogue.get(kind) else {
        return Err(CampaignError::Refused(format!(
            "no verifiers are known for task kind {kind:?}"
        )));
    };
    let runtimes = || {
        let names = spec.runtime.iter().map(String::as_str).chain(
            task.environment
                .kind
                .strip_prefix(splinter_sandbox::environment::RUNTIME_KIND_PREFIX),
        );
        ctx.environments().available(names)
    };
    let mut verifiers: Vec<Box<dyn Verifier>> = Vec::new();
    for verifier in &spec.verifiers {
        match verifier {
            VerifierKind::Formal => verifiers.push(Box::new(ExactMatchVerifier::new(
                Producer {
                    name: FORMAL_PRODUCER.into(),
                    version: FORMAL_VERSION.into(),
                },
                Normalisation::LENIENT,
            ))),
            VerifierKind::Stated => verifiers.push(Box::new(StatedReferenceVerifier::new(
                Producer {
                    name: STATED_PRODUCER.into(),
                    version: STATED_VERSION.into(),
                },
                Normalisation::LENIENT,
            ))),
            VerifierKind::Executable => {
                verifiers.push(Box::new(ExecutableVerifier::new(runtimes())));
            }
            VerifierKind::MutationValidated => verifiers.push(Box::new(
                MutationValidatedVerifier::new(runtimes(), MutationPolicy::default()),
            )),
            VerifierKind::Consistency => verifiers.push(Box::new(AgreementVerifier::new(
                pool.to_vec(),
                Normalisation::LENIENT,
            ))),
            VerifierKind::Judged => {
                if let Some(judge) = judge {
                    verifiers.push(judge.verifier(ctx)?);
                }
            }
        }
    }
    Ok(Strongest::new(verifiers))
}

/// Verdicts appended, by outcome.
#[derive(Clone, Debug, Default, Serialize)]
pub struct VerdictTally {
    /// Pass verdicts.
    pub pass: usize,
    /// Fail verdicts.
    pub fail: usize,
    /// Abstentions.
    pub abstain: usize,
}

/// A verifier that could not grade an experience, or an experience no
/// verifier could grade.
#[derive(Clone, Debug, Serialize)]
pub struct Unverified {
    /// The experience.
    pub experience: ExperienceId,
    /// What could not run, and why.
    pub reason: String,
}

/// What the verify stage reports.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Verified {
    /// Experiences graded.
    pub experiences: usize,
    /// Verdicts appended, by outcome.
    pub verdicts: VerdictTally,
    /// Experiences now decided pass.
    pub passed: usize,
    /// Experiences now decided fail.
    pub failed: usize,
    /// Experiences with no decision.
    pub undecided: usize,
    /// What could not be graded, with why.
    pub unverified: Vec<Unverified>,
}

/// Appends verdicts to every experience of `set`, graded by its kind's
/// verifiers (the judged one only with `judge`).
pub fn verify_set(
    ctx: &Context,
    set: &SetId,
    judge: Option<&Judge>,
    cancel: &CancelToken,
) -> Result<Verified, CampaignError> {
    let store = ctx.experiences();
    let batch = ctx.workspace().batch();
    let members = store.get_set(set)?.members;
    let experiences = members
        .iter()
        .map(|id| store.get(id))
        .collect::<Result<Vec<_>, _>>()?;
    let mut report = Verified::default();
    for (id, experience) in members.iter().zip(&experiences) {
        if cancel.is_cancelled() {
            return Err(CampaignError::Cancelled);
        }
        let task = experience.to_task();
        let unverified = |reason: String| Unverified {
            experience: id.clone(),
            reason,
        };
        let verifiers = match verifiers_for(ctx, &task, &experiences, judge) {
            Ok(verifiers) => verifiers,
            Err(e) => {
                report.unverified.push(unverified(e.to_string()));
                continue;
            }
        };
        let verification = verify_and_annotate(&store, &verifiers, &task, experience)?;
        for note in &verification.annotations {
            if let AnnotationBody::Verdict { outcome, .. } = note.body {
                match outcome {
                    Outcome::Pass => report.verdicts.pass += 1,
                    Outcome::Fail => report.verdicts.fail += 1,
                    Outcome::Abstain => report.verdicts.abstain += 1,
                }
            }
        }
        for (producer, error) in verification.errors {
            report
                .unverified
                .push(unverified(format!("{}: {error}", producer.name)));
        }
        report.experiences += 1;
        match decide(&store.annotations(id)?.annotations) {
            Some(d) if d.passed => report.passed += 1,
            Some(_) => report.failed += 1,
            None => report.undecided += 1,
        }
    }
    batch.commit()?;
    Ok(report)
}
