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

use crate::judging::{controls, reference as reference_of, spaced};
use serde::Serialize;
use splinter_agent::judge::JudgeVerifier;
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::annotation::{AnnotationBody, Outcome, Producer};
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, ExperienceId, Task};
use splinter_core::kinds::DENOISE;
use splinter_eval::denoise::FormalVerifier;
use splinter_eval::verifiers::calibration::{
    calibrate, CalibratedJudge, Calibration, DEFAULT_MIN_PRECISION,
};
use splinter_eval::verifiers::consistency::AgreementVerifier;
use splinter_eval::verifiers::executable::ExecutableVerifier;
use splinter_eval::verifiers::formal::{ExactMatchVerifier, StatedReferenceVerifier};
use splinter_eval::verifiers::grounding::{GroundingPolicy, GroundingVerifier};
use splinter_eval::verifiers::mutation::{MutationPolicy, MutationValidatedVerifier};
use splinter_eval::verifiers::normalise::Normalisation;
use splinter_eval::verifiers::quotation::{QuotationPolicy, QuotationVerifier, StoredEvidence};
use splinter_eval::verifiers::speech::SpeechVerifier;
use splinter_eval::verifiers::{verify_and_annotate, Strongest, Verifier};
use splinter_knowledge::tasks::{Catalogue, VerifierKind};
use splinter_store::experiences::SetId;

use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The producer of the formal verifier's verdicts on generated tasks.
pub const FORMAL_PRODUCER: &str = "splinter-lab/exact-match:lenient";

/// The formal verifier's version: bumped with its normalisation.
pub const FORMAL_VERSION: &str = "1";

/// The producer of the stated-reference verifier's verdicts.
pub const STATED_PRODUCER: &str = "splinter-lab/reference-stated:lenient";

/// The stated-reference verifier's version: bumped with its normalisation
/// or its bound on the answer's length.
pub const STATED_VERSION: &str = "1";

/// The producer of the quotation verifier's verdicts.
pub const QUOTATION_PRODUCER: &str = "splinter-lab/quotation";

/// The quotation verifier's version: bumped with its policy or the way it
/// reads a quotation.
pub const QUOTATION_VERSION: &str = "1";

/// The fewest words a quoted passage runs to count as a claim to a passage.
pub const QUOTATION_MIN_WORDS: usize = 8;

/// The share of the reference advice an answer must reproduce.
pub const QUOTATION_MIN_RECALL: f64 = 0.5;

/// Who the grounding verdicts name as their producer.
pub const GROUNDING_PRODUCER: &str = "splinter-lab/grounding";

/// The version of the grounding check; a changed check carries a new one.
pub const GROUNDING_VERSION: &str = "1";

/// The proper names a dialogue may state that nothing in its source grounds:
/// the check cannot tell a form of address or a common name from a person,
/// and one stray name is not the invented specifics it exists to refuse.
pub const GROUNDING_MAX_NAMES: usize = 1;

/// Who the speech verdicts name as their producer.
pub const SPEECH_PRODUCER: &str = "splinter-lab/speech";

/// The version of the speech check; a changed check carries a new one.
pub const SPEECH_VERSION: &str = "1";

/// How long a judge may take over one answer.
pub const DEFAULT_JUDGE_DEADLINE: Duration = Duration::from_secs(120);

const CALIBRATION: &str = "calibration";

/// The most tasks a fresh calibration's controls are made from.
const MAX_CALIBRATION_TASKS: usize = 24;

/// A judge and the calibration its verdicts are gated by.
#[derive(Clone)]
pub struct Judge {
    model: Model,
    calibration: Calibration,
    judging: Judging,
}

/// What a judge is asked about an answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Judging {
    /// Whether it gives what the task's reference says.
    Reference,
    /// Whether it is a natural reply to the task's instruction; no
    /// reference is shown ([`JudgeVerifier::fit`]).
    Fit,
}

impl Judge {
    /// The judge `reference` names, with its stored calibration; refused
    /// when it was never calibrated.
    pub fn load(ctx: &Context, reference: &ModelRef) -> Result<Self, OrchestratorError> {
        let model = ctx.model(reference)?;
        let producer = judge_verifier(ctx, &model).producer();
        ctx.workspace().refresh()?;
        let calibration = latest_calibration(ctx, &producer)?.ok_or_else(|| {
            OrchestratorError::Refused(format!(
                "judge {reference} has no calibration; its verdicts would stand unmeasured. \
                 Run `splinter judge calibrate <LABELLED-FILE> --judge {reference}` first"
            ))
        })?;
        Ok(Self {
            model,
            calibration,
            judging: Judging::Reference,
        })
    }

    /// The judge `reference` names, with a calibration it can be trusted on:
    /// the stored one when it was measured on at least
    /// [`splinter_orchestrator::config::Config::min_calibration_controls`] controls, else one measured now on
    /// controls made from the references of `tasks` ([`controls`]) and
    /// stored. Refused when the tasks give too few controls to measure on,
    /// or when the judge is not precise enough on them
    /// ([`DEFAULT_MIN_PRECISION`], on passes and on fails): its verdicts
    /// would all abstain, and a kind graded by a judge would grade nothing.
    pub fn calibrated(
        ctx: &Context,
        reference: &ModelRef,
        tasks: &[Task],
    ) -> Result<Self, OrchestratorError> {
        Self::calibrated_for(ctx, reference, tasks, Judging::Reference)
    }

    /// [`Judge::calibrated`] for what `judging` asks: a fit judge is
    /// measured on the same controls - a task's own reference put forward as
    /// the reply, against another family's - under a calibration of its own.
    pub fn calibrated_for(
        ctx: &Context,
        reference: &ModelRef,
        tasks: &[Task],
        judging: Judging,
    ) -> Result<Self, OrchestratorError> {
        let model = ctx.model(reference)?;
        let verifier = judge_verifier_for(ctx, &model, judging);
        let minimum = ctx.config().min_calibration_controls;
        ctx.workspace().refresh()?;
        let stored = latest_calibration(ctx, &verifier.producer())?.filter(|c| c.n >= minimum);
        let calibration = match stored {
            Some(calibration) => calibration,
            None => {
                let with_reference: Vec<Task> = tasks
                    .iter()
                    .filter(|task| reference_of(task).is_some())
                    .cloned()
                    .collect();
                let labelled = controls(ctx, &spaced(&with_reference, MAX_CALIBRATION_TASKS))?;
                if labelled.len() < minimum {
                    return Err(OrchestratorError::Refused(format!(
                        "judge {reference} has no calibration and the tasks give {} controls to \
                         measure one on, fewer than {minimum}: a judge needs \
                         tasks with a reference from several families of sources, or `splinter \
                         judge calibrate <LABELLED-FILE> --judge {reference}`",
                        labelled.len()
                    )));
                }
                let calibration = calibrate(&verifier, &labelled)?;
                store_calibration(ctx, &calibration)?;
                calibration
            }
        };
        let judge = Self {
            model,
            calibration,
            judging,
        };
        if !judge.trusted() {
            let c = &judge.calibration;
            return Err(OrchestratorError::Refused(format!(
                "judge {reference} is not precise enough to grade with: precision {:?} on passes \
                 and {:?} on fails over {} controls, below {DEFAULT_MIN_PRECISION}; a kind \
                 graded by a judge would be graded by nothing",
                c.precision_pass, c.precision_fail, c.n
            )));
        }
        Ok(judge)
    }

    /// The judge this command grades with: the model its context names
    /// ([`Context::set_judge`]), loaded with its stored calibration; `None`
    /// when it names none. Refused when the judge was never calibrated or is
    /// not precise enough ([`Judge::trusted`]): grading by a judge whose
    /// verdicts would all abstain is not grading.
    pub fn active(ctx: &Context) -> Result<Option<Self>, OrchestratorError> {
        let Some(reference) = ctx.judge() else {
            return Ok(None);
        };
        let judge = Self::load(ctx, &reference)?;
        if !judge.trusted() {
            return Err(OrchestratorError::Refused(format!(
                "judge {reference} is not precise enough to grade with: precision {:?} on \
                 passes and {:?} on fails over {} controls, below {DEFAULT_MIN_PRECISION}",
                judge.calibration.precision_pass,
                judge.calibration.precision_fail,
                judge.calibration.n
            )));
        }
        Ok(Some(judge))
    }

    /// Whether its measured precision, on passes and on fails, reaches
    /// [`DEFAULT_MIN_PRECISION`], so that its verdicts stand.
    #[must_use]
    pub fn trusted(&self) -> bool {
        let precise = |p: Option<f64>| p.is_some_and(|p| p >= DEFAULT_MIN_PRECISION);
        precise(self.calibration.precision_pass) && precise(self.calibration.precision_fail)
    }

    pub(crate) fn verifier(&self, ctx: &Context) -> Result<Box<dyn Verifier>, OrchestratorError> {
        Ok(Box::new(CalibratedJudge::new(
            judge_verifier_for(ctx, &self.model, self.judging),
            self.calibration.clone(),
            DEFAULT_MIN_PRECISION,
        )?))
    }
}

/// Whether only a judge can pass an answer to a task of `kind`
/// ([`splinter_knowledge::tasks::TaskKind::needs_judge`]).
#[must_use]
pub fn kind_needs_judge(kind: &str) -> bool {
    Catalogue::builtin()
        .get(kind)
        .is_some_and(|spec| spec.needs_judge())
}

/// The judged verifier on `model`.
pub(crate) fn judge_verifier(ctx: &Context, model: &Model) -> JudgeVerifier {
    judge_verifier_for(ctx, model, Judging::Reference)
}

/// The verifier of `judging` on `model`.
fn judge_verifier_for(ctx: &Context, model: &Model, judging: Judging) -> JudgeVerifier {
    match judging {
        Judging::Reference => {
            JudgeVerifier::new(model.clone(), ctx.handle(), DEFAULT_JUDGE_DEADLINE)
        }
        Judging::Fit => JudgeVerifier::fit(model.clone(), ctx.handle(), DEFAULT_JUDGE_DEADLINE),
    }
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
) -> Result<Option<Calibration>, OrchestratorError> {
    let Some((_, value)) = ctx.workspace().pointer(&calibration_pointer(producer))? else {
        return Ok(None);
    };
    let digest = Digest::parse(&value).map_err(|e| {
        OrchestratorError::Refused(format!("a calibration pointer is corrupt: {e}"))
    })?;
    Ok(ctx.workspace().get_document(CALIBRATION, &digest)?)
}

/// Records `calibration` as its judge's latest, keeping the ones before it,
/// and returns its address.
pub(crate) fn store_calibration(
    ctx: &Context,
    calibration: &Calibration,
) -> Result<Digest, OrchestratorError> {
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

/// The grounding check as every grade of a dialogue or an exam applies it.
pub(crate) fn grounding_verifier(ctx: &Context) -> GroundingVerifier {
    GroundingVerifier::new(
        Producer {
            name: GROUNDING_PRODUCER.into(),
            version: GROUNDING_VERSION.into(),
        },
        Box::new(StoredEvidence::new(ctx.sources())),
        GroundingPolicy {
            max_ungrounded_names: GROUNDING_MAX_NAMES,
            min_quote_words: QUOTATION_MIN_WORDS,
        },
    )
}

/// The verifiers that grade answers to `task`: its kind's, in the
/// environments it runs code in, with `pool` as the other answers
/// agreement is counted over. Refused for a kind with no known verifiers.
pub(crate) fn verifiers_for(
    ctx: &Context,
    task: &Task,
    pool: &[Experience],
    judge: Option<&Judge>,
) -> Result<Strongest, OrchestratorError> {
    let kind = &task.task.kind;
    if kind == DENOISE {
        return Ok(Strongest::new(vec![Box::new(FormalVerifier::new())]));
    }
    let catalogue = Catalogue::builtin();
    let Some(spec) = catalogue.get(kind) else {
        return Err(OrchestratorError::Refused(format!(
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
            VerifierKind::Quotation => verifiers.push(Box::new(QuotationVerifier::new(
                Producer {
                    name: QUOTATION_PRODUCER.into(),
                    version: QUOTATION_VERSION.into(),
                },
                Box::new(StoredEvidence::new(ctx.sources())),
                QuotationPolicy {
                    min_words: QUOTATION_MIN_WORDS,
                    min_reference_recall: QUOTATION_MIN_RECALL,
                },
            ))),
            VerifierKind::Grounding => verifiers.push(Box::new(grounding_verifier(ctx))),
            VerifierKind::Speech => verifiers.push(Box::new(SpeechVerifier::new(Producer {
                name: SPEECH_PRODUCER.into(),
                version: SPEECH_VERSION.into(),
            }))),
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
) -> Result<Verified, OrchestratorError> {
    let active = match judge {
        Some(_) => None,
        None => Judge::active(ctx)?,
    };
    let judge = judge.or(active.as_ref());
    let store = ctx.experiences();
    let batch = ctx.workspace().batch();
    let members = store.get_set(set)?.members;
    let experiences = members
        .iter()
        .map(|id| store.get(id))
        .collect::<Result<Vec<_>, _>>()?;
    let mut report = Verified::default();
    let mut verified = Vec::new();
    for (id, experience) in members.iter().zip(&experiences) {
        if cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
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
        verified.push(id.clone());
    }
    let decisions = store.decisions(&verified)?;
    for id in &verified {
        match decisions.get(id) {
            Some(d) if d.passed => report.passed += 1,
            Some(_) => report.failed += 1,
            None => report.undecided += 1,
        }
    }
    batch.commit()?;
    Ok(report)
}
