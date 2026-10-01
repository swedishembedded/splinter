// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded and
// unverifiable model-written tasks out of training data, for its clients.
// If your team needs expertise in synthetic data quality or verifier
// design, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Admission: a generator model's proposals checked by code, in the order
//! the module documentation of [`super`] lists, cheapest first.

use std::collections::BTreeSet;

use splinter_lab::verifiers::executable::{
    program_text, ExecutableCheck, ExecutableVerifier, Expectation, ExpectedStdout, CHECK_KIND,
    OUTPUT_CHECK_KIND,
};
use splinter_lab::verifiers::mutation::{validate_oracle, MutationPolicy};
use splinter_lab::verifiers::normalise::Normalisation;
use splinter_sandbox::{ResolvedEnvironment, RuntimeEnvironment};
use splinter_store::annotation::Outcome;
use splinter_store::digest::Digest;
use splinter_store::experience::{Environment, Privileged, PrivilegedKind, Span, Task};
use splinter_store::sources::SourceStore;
use splinter_views::check_self_contained;

use crate::gates::normalize;

use super::dedup::{Repeat, Seen};
use super::generator::{GenerateError, SourceText};
use super::grounding::{content_words, support};
use super::kind::{AnswerForm, Material, SolverEnvironment, TaskKind};
use super::reply::{Candidate, Reply};
use super::{GeneratedTask, GenerationPolicy, GenerationReport, Rejection};

/// One request's outcome, waiting for admission.
pub(crate) struct Proposal {
    /// The kind asked for.
    pub(crate) kind: TaskKind,
    /// The offered runtime the kind's code runs in, when it runs any.
    pub(crate) runtime: Option<RuntimeEnvironment>,
    /// The digest of the prompt sent.
    pub(crate) prompt: Digest,
    /// The tasks the model proposed, or why it proposed none.
    pub(crate) reply: Result<Reply, Refusal>,
}

/// What admission needs besides the proposals.
pub(crate) struct Admission {
    pub(crate) store: SourceStore,
    pub(crate) policy: GenerationPolicy,
    pub(crate) mutation: MutationPolicy,
    pub(crate) generator: String,
}

/// A rejection: why, and what exactly failed.
pub(crate) type Refusal = (Rejection, String);

/// The subject `candidate` names, when its kind must name one; see the
/// module documentation of [`super`].
fn subject(context: &Context<'_>, candidate: &Candidate) -> Result<Option<String>, Refusal> {
    if !context.kind.names_subject() {
        return Ok(None);
    }
    let refuse = |detail: String| Err((Rejection::NoSubject, detail));
    let Some(subject) = candidate
        .subject
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return refuse("no subject is named: say what the question is about".into());
    };
    if content_words(subject).is_empty() {
        return refuse(format!("{subject:?} names nothing in particular"));
    }
    let wanted = normalize(subject);
    if !normalize(&candidate.instruction).contains(&wanted) {
        return refuse(format!(
            "the instruction does not name its subject {subject:?}"
        ));
    }
    let source = context.source;
    let named_by_source = source
        .identity
        .names()
        .any(|name| normalize(name).contains(&wanted));
    let named_by_section = candidate
        .evidence
        .iter()
        .filter_map(|citation| source.section_text(citation.section))
        .any(|text| normalize(text).contains(&wanted));
    if !named_by_source && !named_by_section {
        return refuse(format!(
            "{subject:?} is named neither by the source ({}) nor by a cited section",
            source.identity.names().collect::<Vec<_>>().join(", ")
        ));
    }
    Ok(Some(subject.to_string()))
}

/// Everything a candidate's checks are judged in.
struct Context<'a> {
    source: &'a SourceText,
    kind: &'a TaskKind,
    runtime: Option<&'a RuntimeEnvironment>,
    /// The runtime's record, when the kind has one.
    runtime_record: Option<Environment>,
}

impl Admission {
    /// Every proposal's candidates admitted or rejected, in order; the
    /// duplicate rules span the whole batch. Blocks while checks run.
    pub(crate) fn admit(
        &self,
        source: &SourceText,
        proposals: Vec<Proposal>,
    ) -> Result<GenerationReport, GenerateError> {
        let mut report = GenerationReport::default();
        let mut seen = Seen::new(self.policy.shingle_words, self.policy.max_overlap);
        for proposal in proposals {
            let name = proposal.kind.name.as_str();
            let parsed = match proposal.reply {
                Ok(parsed) => parsed,
                Err((rejection, why)) => {
                    report.reject(name, 0, rejection, why);
                    continue;
                }
            };
            let runtime_record = proposal
                .runtime
                .as_ref()
                .map(|env| ResolvedEnvironment::Runtime(env.clone()).record())
                .transpose()
                .map_err(splinter_lab::verifiers::VerifyError::from)?;
            let context = Context {
                source,
                kind: &proposal.kind,
                runtime: proposal.runtime.as_ref(),
                runtime_record,
            };
            for (index, candidate) in parsed.tasks.into_iter().enumerate() {
                if index >= self.policy.tasks_per_request {
                    let detail = format!("{} tasks were asked for", self.policy.tasks_per_request);
                    report.reject(name, index, Rejection::OverCount, detail);
                    continue;
                }
                match self.candidate(&context, candidate, &seen)? {
                    Ok((task, subject)) => {
                        seen.admit(&task.instruction);
                        let generated = GeneratedTask {
                            subject,
                            task,
                            generator: self.generator.clone(),
                            prompt: proposal.prompt.clone(),
                        };
                        report.admit(name, generated);
                    }
                    Err((reason, detail)) => report.reject(name, index, reason, detail),
                }
            }
        }
        Ok(report)
    }

    /// One candidate admitted as a task with the subject it names, or
    /// refused; an error only when a check could not be run at all.
    fn candidate(
        &self,
        context: &Context<'_>,
        candidate: Candidate,
        seen: &Seen,
    ) -> Result<Result<(Task, Option<String>), Refusal>, GenerateError> {
        let kind = context.kind;
        let evidence = match evidence(context, &candidate) {
            Ok(evidence) => evidence,
            Err(refusal) => return Ok(Err(refusal)),
        };
        let mut texts = Vec::with_capacity(evidence.len());
        for span in &evidence {
            match self.store.read_span(span).map(String::from_utf8) {
                Ok(Ok(text)) => texts.push(text),
                Ok(Err(_)) => {
                    return Ok(Err((
                        Rejection::Unresolved,
                        "a span is not UTF-8 text".into(),
                    )))
                }
                Err(e) => return Ok(Err((Rejection::Unresolved, e.to_string()))),
            }
        }
        let privileged = match privileged(context, &candidate, &evidence) {
            Ok(privileged) => privileged,
            Err(refusal) => return Ok(Err(refusal)),
        };

        // The student sees the instruction alone: for a closed-book kind the
        // evidence is material it does not see, beside the dropped items.
        let hidden: Vec<Privileged> = if kind.shows_material {
            Vec::new()
        } else {
            evidence
                .iter()
                .zip(&texts)
                .map(|(span, text)| Privileged {
                    kind: PrivilegedKind::Passage,
                    content: text.clone(),
                    span: Some(span.clone()),
                })
                .collect()
        };
        let dropped: Vec<&Privileged> = hidden.iter().chain(&privileged).collect();
        if let Err(why) = check_self_contained(&candidate.instruction, &dropped) {
            return Ok(Err((Rejection::NotSelfContained, why.to_string())));
        }

        let subject = match subject(context, &candidate) {
            Ok(subject) => subject,
            Err(refusal) => return Ok(Err(refusal)),
        };

        match seen.repeats(&candidate.instruction) {
            Some(Repeat::Exact) => {
                return Ok(Err((Rejection::Duplicate, candidate.instruction)));
            }
            Some(Repeat::Near) => {
                return Ok(Err((Rejection::NearDuplicate, candidate.instruction)));
            }
            None => {}
        }

        let environment = match kind.environment {
            SolverEnvironment::ClosedBook => Environment::closed_book(),
            SolverEnvironment::Runtime => match &context.runtime_record {
                Some(record) => record.clone(),
                // `generate` resolves the runtime of every kind that has one.
                None => unreachable!("a runtime kind's proposal carries its runtime"),
            },
        };
        let task = match Task::new(
            kind.name.as_str(),
            evidence,
            environment,
            candidate.instruction.as_str(),
            privileged,
        ) {
            Ok(task) => task,
            Err(e) => return Ok(Err((Rejection::Invalid, e.to_string()))),
        };

        let grounded = match kind.answer {
            AnswerForm::Text => {
                let found = support(&candidate.reference, &texts.join("\n"));
                if found.holds(self.policy.min_support) {
                    Ok(())
                } else {
                    Err((
                        Rejection::Ungrounded,
                        format!(
                            "content words supported: {:?} (at least {} needed); numbers \
                             traceable: {}",
                            found.share, self.policy.min_support, found.numbers_traceable
                        ),
                    ))
                }
            }
            AnswerForm::Program => self.program_passes(context, &task, &candidate)?,
            AnswerForm::Output => run_checks(context, &task, OUTPUT_CHECK_KIND, "")?,
        };
        Ok(grounded.map(|()| (task, subject)))
    }

    /// Whether a program reference passes its checks, and every generated
    /// test is admitted against it by mutation validation.
    fn program_passes(
        &self,
        context: &Context<'_>,
        task: &Task,
        candidate: &Candidate,
    ) -> Result<Result<(), Refusal>, GenerateError> {
        if !candidate.checks.is_empty() {
            if let Err(refusal) = run_checks(context, task, CHECK_KIND, &candidate.reference)? {
                return Ok(Err(refusal));
            }
        }
        let Some(runtime) = context.runtime else {
            unreachable!("a program kind's proposal carries its runtime");
        };
        let reference = program_text(&candidate.reference);
        for (position, test) in candidate.tests.iter().enumerate() {
            let Some(test) = test.to_check(check_environment(context)) else {
                unreachable!("expectation-free tests are refused before grounding");
            };
            let validation = validate_oracle(reference, &test, runtime, &self.mutation)?;
            if !validation.admitted {
                return Ok(Err((
                    Rejection::TestNotAdmitted,
                    format!(
                        "test {position}: passes the reference: {}, mutants killed {}, survived \
                         {}, errored {}",
                        validation.passes_reference,
                        validation.killed,
                        validation.survived,
                        validation.errored
                    ),
                )));
            }
        }
        Ok(Ok(()))
    }
}

/// The task's checks of privileged kind `kind` run against `answer` in the
/// kind's runtime, through the lab's executable verifier.
fn run_checks(
    context: &Context<'_>,
    task: &Task,
    kind: &str,
    answer: &str,
) -> Result<Result<(), Refusal>, GenerateError> {
    let Some(runtime) = context.runtime else {
        unreachable!("a computed kind's proposal carries its runtime");
    };
    let verifier = ExecutableVerifier::new(vec![runtime.clone()]);
    let finding = verifier.check(task, kind, Some(answer))?;
    Ok(match finding.outcome {
        Outcome::Pass => Ok(()),
        _ => Err((Rejection::ChecksFailed, finding.evidence.to_string())),
    })
}

/// Where a kind's checks run: the task's own environment when the student
/// works in the runtime, the runtime otherwise.
fn check_environment(context: &Context<'_>) -> Option<Environment> {
    match context.kind.environment {
        SolverEnvironment::Runtime => None,
        SolverEnvironment::ClosedBook => context.runtime_record.clone(),
    }
}

/// The candidate's evidence as spans of the source part.
fn evidence(context: &Context<'_>, candidate: &Candidate) -> Result<Vec<Span>, Refusal> {
    let source = context.source;
    if candidate.evidence.is_empty() {
        return Err((Rejection::UnknownSection, "no evidence is cited".into()));
    }
    let mut spans = Vec::with_capacity(candidate.evidence.len());
    let mut sections = BTreeSet::new();
    for citation in &candidate.evidence {
        let position = citation.section;
        let (Some(section), Some(text)) =
            (source.sections.get(position), source.section_text(position))
        else {
            return Err((
                Rejection::UnknownSection,
                format!("section {position} of {}", source.sections.len()),
            ));
        };
        let (start, end) = match citation.quote.as_deref().filter(|q| !q.trim().is_empty()) {
            None => (section.range.start, section.range.end),
            Some(quote) => match text.find(quote) {
                Some(at) => (
                    section.range.start + at,
                    section.range.start + at + quote.len(),
                ),
                None => {
                    return Err((
                        Rejection::QuoteNotFound,
                        format!("{quote:?} is not in section {position}"),
                    ))
                }
            },
        };
        let span = Span::in_part(
            source.part.clone(),
            source.content.clone(),
            start as u64,
            end as u64,
        )
        .map_err(|e| (Rejection::Invalid, e.to_string()))?;
        sections.insert(position);
        spans.push(span);
    }
    if sections.len() < context.kind.min_sections {
        return Err((
            Rejection::TooFewSections,
            format!(
                "{} distinct section(s) cited; the kind needs {}",
                sections.len(),
                context.kind.min_sections
            ),
        ));
    }
    Ok(spans)
}

/// The candidate's privileged items: its reference, hints, checks and
/// tests, and for an output answer the check that establishes it; refused
/// when material the kind requires is missing or shown wrongly.
fn privileged(
    context: &Context<'_>,
    candidate: &Candidate,
    evidence: &[Span],
) -> Result<Vec<Privileged>, Refusal> {
    let kind = context.kind;
    let missing = |what: &str| Err((Rejection::MissingMaterial, what.to_string()));
    if candidate.reference.trim().is_empty() {
        return missing("the reference is empty");
    }
    let material = candidate
        .material
        .as_deref()
        .filter(|m| !m.trim().is_empty());
    match (kind.shows_material, material) {
        (true, None) => {
            return Err((Rejection::MaterialNotShown, "no material is given".into()));
        }
        (true, Some(m)) if !candidate.instruction.contains(m) => {
            return Err((
                Rejection::MaterialNotShown,
                "the material is not in the instruction verbatim".into(),
            ));
        }
        (false, Some(_)) => {
            return Err((
                Rejection::MaterialNotShown,
                "a closed-book kind shows no material".into(),
            ));
        }
        _ => {}
    }
    let (checks, tests) = (!candidate.checks.is_empty(), !candidate.tests.is_empty());
    for required in &kind.requires {
        let present = match required {
            Material::Hints => !candidate.hints.is_empty(),
            Material::Checks => checks,
            Material::Tests => tests,
            Material::ChecksOrTests => checks || tests,
        };
        if !present {
            return missing(&format!("the kind requires {required:?}"));
        }
    }
    if (checks || tests) && kind.answer != AnswerForm::Program {
        return Err((
            Rejection::Invalid,
            "checks and tests grade program answers only".into(),
        ));
    }

    let reference_span = match evidence {
        [only] => Some(only.clone()),
        _ => None,
    };
    let mut items = vec![Privileged {
        kind: PrivilegedKind::Reference,
        content: candidate.reference.clone(),
        span: reference_span,
    }];
    items.extend(candidate.hints.iter().map(|hint| Privileged {
        kind: PrivilegedKind::Hint,
        content: hint.clone(),
        span: None,
    }));
    let environment = check_environment(context);
    let as_item = |check: ExecutableCheck, kind: &str| -> Result<Privileged, Refusal> {
        Ok(Privileged {
            kind: PrivilegedKind::Other(kind.to_string()),
            content: serde_json::to_string(&check)
                .map_err(|e| (Rejection::Invalid, e.to_string()))?,
            span: None,
        })
    };
    for (list, item_kind) in [
        (&candidate.checks, CHECK_KIND),
        (
            &candidate.tests,
            splinter_lab::verifiers::mutation::TEST_KIND,
        ),
    ] {
        for check in list {
            let Some(check) = check.to_check(environment.clone()) else {
                return missing("a check or test expects neither an exit code nor output");
            };
            items.push(as_item(check, item_kind)?);
        }
    }
    if kind.answer == AnswerForm::Output {
        let Some(code) = material else {
            unreachable!("an output kind shows its material, checked above");
        };
        let check = ExecutableCheck {
            code: code.to_string(),
            stdin: None,
            expect: Expectation {
                exit_code: Some(0),
                stdout: Some(ExpectedStdout {
                    text: candidate.reference.clone(),
                    normalisation: Normalisation::WHITESPACE,
                }),
            },
            environment,
        };
        items.push(as_item(check, OUTPUT_CHECK_KIND)?);
    }
    Ok(items)
}
