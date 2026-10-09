// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in
// continual learning from user feedback or admission of model-written
// training data, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The extract and gate stages: what a person taught, proposed by a model
//! and ruled on by code.
//!
//! [`extract`] shows each stored session to the extractor model (secrets were
//! removed at intake) and stores the proposals, whole, as a claim set. A
//! session whose reply stays unusable after correction is reported with the
//! reason and has no proposals; nothing is guessed in its place.
//!
//! [`gate`] rules on every proposal of a claim set by code
//! ([`splinter_knowledge::claims`]) against the ledger so far and appends
//! each ruling to it, refusals included; a judge, when one is named, decides
//! whether a later claim supersedes, restates or is apart from an earlier
//! one about the same thing. Ruling on a set again rules on nothing, so the
//! stage is safe to repeat after a crash. [`ledger`] reads the ledger back:
//! the claims that are live and how often each was said again, the ones
//! superseded and by which, and the proposals refused with their reasons.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use serde::Serialize;
use splinter_agent::claim_judge::SvenClaimJudge;
use splinter_agent::claims::{ClaimExtractor, Extraction};
use splinter_agent::CancelToken;
use splinter_core::claim::{Claim, ClaimId, ClaimSet, LedgerEntry, Ruling, SessionClaims};
use splinter_core::digest::Digest;
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_core::source::SourceId;
use splinter_knowledge::claims::agreement::{agree, Agreed};
use splinter_knowledge::claims::extract::ExtractionPolicy;
use splinter_knowledge::claims::{GateError, Ledger, RuleRequest};
use splinter_knowledge::session::{SessionError, SessionView};
use splinter_knowledge::tasks::{DEFAULT_REPAIRS, DEFAULT_REQUEST_DEADLINE};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::ids;

use crate::claim_quality::{assess, ExtractionQuality};
use crate::verify::judge_model;
use splinter_knowledge::claims::coverage::Subject;

fn session_error(source: &SourceId, e: SessionError) -> OrchestratorError {
    match e {
        SessionError::Store(e) => e.into(),
        other => OrchestratorError::Refused(format!("session {source}: {other}")),
    }
}

fn gate_error(e: GateError) -> OrchestratorError {
    OrchestratorError::Refused(e.to_string())
}

/// One extraction request.
pub struct ExtractRequest<'a> {
    /// The stored sessions to read, in the order they happened.
    pub sessions: &'a [SourceId],
    /// The extractor model.
    pub extractor: &'a ModelRef,
    /// How each session's extraction is bounded.
    pub policy: ExtractionPolicy,
    /// How many times each session is read, in different orders; a claim is
    /// proposed only when every pass proposed it (at least one).
    pub passes: u32,
    /// No session is started after this.
    pub deadline: Option<Instant>,
    /// Stops extraction.
    pub cancel: CancelToken,
}

/// The extraction passes made over each session unless asked otherwise: two
/// readings in different orders, and only what both produced is proposed.
pub const DEFAULT_EXTRACTION_PASSES: u32 = 2;

/// A proposal one pass made and another did not.
#[derive(Clone, Debug, Serialize)]
pub struct Unconfirmed {
    /// The session.
    pub session: SourceId,
    /// The statement proposed.
    pub statement: String,
}

/// A session the extractor gave nothing usable for.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionFailed {
    /// The session.
    pub session: SourceId,
    /// Why.
    pub reason: String,
}

/// What the extract stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct ClaimsExtracted {
    /// The stored claim set (`claims gate <claim_set>`).
    pub claim_set: Digest,
    /// Sessions read.
    pub sessions: usize,
    /// Extraction passes made over each.
    pub passes: u32,
    /// Claims proposed across them: those every pass produced.
    pub proposals: usize,
    /// Proposals some pass made and another did not, so left unruled.
    pub disagreements: usize,
    /// Each of them.
    pub unconfirmed: Vec<Unconfirmed>,
    /// Proposals by kind.
    pub by_kind: BTreeMap<String, usize>,
    /// Sessions the extractor gave nothing usable for.
    pub failed: Vec<ExtractionFailed>,
    /// Why extraction stopped before every session was read, if it did.
    pub stopped: Option<String>,
}

/// Has the extractor model propose claims from each of `request.sessions`
/// and stores them as a claim set.
pub fn extract(
    ctx: &Context,
    request: &ExtractRequest<'_>,
) -> Result<ClaimsExtracted, OrchestratorError> {
    let model = ctx.model(request.extractor)?;
    let extractor = ClaimExtractor::new(model)
        .with_policy(request.policy)
        .with_cancel(request.cancel.clone());
    let store = ctx.sources();
    let mut report = ClaimsExtracted {
        claim_set: Digest::of(b""),
        sessions: 0,
        passes: request.passes,
        proposals: 0,
        disagreements: 0,
        unconfirmed: Vec::new(),
        by_kind: BTreeMap::new(),
        failed: Vec::new(),
        stopped: None,
    };
    let mut sessions = Vec::new();
    for id in request.sessions {
        if request.cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
        }
        if request.deadline.is_some_and(|d| Instant::now() >= d) {
            report.stopped = Some("the budget was spent before every session was read".into());
            break;
        }
        let view = SessionView::load(&store, id).map_err(|e| session_error(id, e))?;
        let extraction = extract_session(ctx, &extractor, &view, request)?;
        report.sessions += 1;
        sessions.push(match extraction {
            Passed::Proposals(agreed) => {
                report.proposals += agreed.kept.len();
                report.disagreements += agreed.unconfirmed.len();
                report
                    .unconfirmed
                    .extend(agreed.unconfirmed.iter().map(|p| Unconfirmed {
                        session: id.clone(),
                        statement: p.statement.clone(),
                    }));
                for p in &agreed.kept {
                    *report.by_kind.entry(p.kind.as_str().into()).or_default() += 1;
                }
                SessionClaims {
                    session: id.clone(),
                    proposals: agreed.kept,
                    failure: None,
                    unconfirmed: agreed.unconfirmed,
                }
            }
            Passed::Declined(reason) => {
                report.failed.push(ExtractionFailed {
                    session: id.clone(),
                    reason: reason.clone(),
                });
                SessionClaims {
                    session: id.clone(),
                    proposals: Vec::new(),
                    failure: Some(reason),
                    unconfirmed: Vec::new(),
                }
            }
        });
    }
    report.claim_set = ctx.claims().put_set(&ClaimSet {
        extractor: extractor.identity().to_string(),
        passes: request.passes,
        sessions,
    })?;
    Ok(report)
}

/// What the passes over one session agreed on, or why one gave nothing.
enum Passed {
    Proposals(Agreed),
    Declined(String),
}

/// Reads `view` `request.passes` times and keeps what every pass proposed.
fn extract_session(
    ctx: &Context,
    extractor: &ClaimExtractor,
    view: &SessionView,
    request: &ExtractRequest<'_>,
) -> Result<Passed, OrchestratorError> {
    let mut passes = Vec::new();
    for pass in 0..request.passes {
        let extraction = ctx
            .block_on(extractor.extract_pass(view, pass))
            .map_err(|e| OrchestratorError::Model {
                model: request.extractor.to_string(),
                detail: e.to_string(),
            })?;
        match extraction {
            Extraction::Proposals(proposals) => passes.push(proposals),
            Extraction::Declined(reason) => {
                return Ok(Passed::Declined(format!("pass {}: {reason}", pass + 1)))
            }
        }
    }
    Ok(Passed::Proposals(agree(&passes)))
}

/// One proposal's ruling, as the gate stage reports it.
#[derive(Clone, Debug, Serialize)]
pub struct Ruled {
    /// The proposal's position in the claim set.
    pub index: usize,
    /// The session it was proposed from.
    pub session: SourceId,
    /// `correction`, `fact` or `procedure`.
    pub kind: String,
    /// The statement proposed.
    pub statement: String,
    /// `admitted`, `reinforced` or `refused`.
    pub outcome: &'static str,
    /// The claim, when admitted or reinforcing.
    pub claim: Option<ClaimId>,
    /// The live claim a reinforcing one restates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reinforces: Option<ClaimId>,
    /// The earlier claims an admitted one replaces as the answer to its
    /// question.
    pub supersedes: Vec<ClaimId>,
    /// Why it was refused: the reason's stable name.
    pub reason_code: Option<&'static str>,
    /// Why it was refused, in words.
    pub reason: Option<String>,
}

/// One gate request.
#[derive(Clone, Copy)]
pub struct GateRequest<'a> {
    /// The claim set to rule on.
    pub claim_set: &'a Digest,
    /// The model that decides what a later claim does to an earlier one
    /// about the same thing; without one the statements decide.
    pub judge: Option<&'a ModelRef>,
    /// Whether the judge, when there is one, is also shown each admitted
    /// claim's cited words and statement and must find that the words
    /// assert the statement.
    pub entail: bool,
    /// What the facts the sessions are known to be about look like, for the
    /// report of how well the extraction covered them.
    pub subjects: &'a [Subject],
}

impl<'a> GateRequest<'a> {
    /// A request to rule on `claim_set` by code alone.
    #[must_use]
    pub fn new(claim_set: &'a Digest) -> Self {
        Self {
            claim_set,
            judge: None,
            entail: true,
            subjects: &[],
        }
    }
}

/// What the gate stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct ClaimsGated {
    /// The claim set ruled on.
    pub claim_set: Digest,
    /// Proposals in it.
    pub proposals: usize,
    /// Proposals ruled on now.
    pub ruled: usize,
    /// Proposals an earlier ruling already decided.
    pub already_ruled: usize,
    /// Claims admitted now.
    pub admitted: usize,
    /// Earlier claims the admitted ones replaced.
    pub superseded: usize,
    /// Proposals that said a live claim's fact again: recorded, not new.
    pub reinforced: usize,
    /// Proposals refused now, by reason.
    pub refused: BTreeMap<String, usize>,
    /// Every ruling made now, in the order made.
    pub rulings: Vec<Ruled>,
    /// Claims live after the stage.
    pub live: usize,
    /// Sessions of the set the extractor gave nothing usable for.
    pub extraction_failed: usize,
    /// How well the extraction covered the sessions that name the fact they
    /// are about; absent when none does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extraction_quality: Option<ExtractionQuality>,
}

/// Rules on the proposals of the claim set `set_id` and appends the rulings
/// to the ledger.
pub fn gate(ctx: &Context, request: &GateRequest<'_>) -> Result<ClaimsGated, OrchestratorError> {
    let set_id = request.claim_set;
    let claims = ctx.claims();
    let set = claims.get_set(set_id)?;
    let mut views = BTreeMap::new();
    for session in &set.sessions {
        let view = SessionView::load(&ctx.sources(), &session.session)
            .map_err(|e| session_error(&session.session, e))?;
        views.insert(session.session.clone(), view);
    }
    let ledger = Ledger::new(claims.entries()?);
    let judge = request
        .judge
        .map(|reference| {
            Ok::<_, OrchestratorError>(SvenClaimJudge::new(
                judge_model(ctx, reference)?,
                ctx.handle(),
                DEFAULT_REQUEST_DEADLINE,
                DEFAULT_REPAIRS,
            ))
        })
        .transpose()?;
    let mut rules = RuleRequest::new(set_id, &set, &views);
    if let Some(judge) = &judge {
        rules = rules.judged_by(judge);
        if !request.entail {
            rules = rules.without_entailment();
        }
    }
    let entries = ledger.rule_set(&rules).map_err(gate_error)?;
    claims.append(&entries)?;

    let proposals: usize = set.sessions.iter().map(|s| s.proposals.len()).sum();
    let mut report = ClaimsGated {
        claim_set: set_id.clone(),
        proposals,
        ruled: entries.len(),
        already_ruled: proposals - entries.len(),
        admitted: 0,
        superseded: 0,
        reinforced: 0,
        refused: BTreeMap::new(),
        rulings: Vec::new(),
        live: 0,
        extraction_failed: set.sessions.iter().filter(|s| s.failure.is_some()).count(),
        extraction_quality: None,
    };
    for entry in &entries {
        let ruled = ruled(entry).map_err(|e| gate_error(e.into()))?;
        match &entry.ruling {
            Ruling::Admitted { supersedes, .. } => {
                report.admitted += 1;
                report.superseded += supersedes.len();
            }
            Ruling::Reinforced { supersedes, .. } => {
                report.reinforced += 1;
                report.superseded += supersedes.len();
            }
            Ruling::Refused { reason } => {
                *report.refused.entry(reason.code().into()).or_default() += 1;
            }
            Ruling::Forgotten { .. } => {}
        }
        report.rulings.push(ruled);
    }
    let ruled_on_set: Vec<LedgerEntry> = claims
        .entries()?
        .into_iter()
        .filter(|e| e.claim_set == *set_id)
        .collect();
    report.extraction_quality = assess(&set, &ruled_on_set, &views, request.subjects);
    report.live = Ledger::new(claims.entries()?)
        .live()
        .map_err(|e| gate_error(e.into()))?
        .len();
    Ok(report)
}

fn ruled(entry: &LedgerEntry) -> Result<Ruled, serde_json::Error> {
    let (outcome, claim, reinforces, supersedes, reason) = match &entry.ruling {
        Ruling::Admitted { claim, supersedes } => (
            "admitted",
            Some(claim.id()?),
            None,
            supersedes.clone(),
            None,
        ),
        Ruling::Reinforced {
            claim,
            of,
            supersedes,
        } => (
            "reinforced",
            Some(claim.id()?),
            Some(of.clone()),
            supersedes.clone(),
            None,
        ),
        Ruling::Forgotten { claim } => ("forgotten", Some(claim.clone()), None, Vec::new(), None),
        Ruling::Refused { reason } => ("refused", None, None, Vec::new(), Some(reason)),
    };
    Ok(Ruled {
        index: entry.index,
        session: entry.session.clone(),
        kind: entry.proposal.kind.as_str().into(),
        statement: entry.proposal.statement.clone(),
        outcome,
        claim,
        reinforces,
        supersedes,
        reason_code: reason.map(|r| r.code()),
        reason: reason.map(ToString::to_string),
    })
}

/// A claim as the ledger report shows it.
#[derive(Clone, Debug, Serialize)]
pub struct ClaimLine {
    /// Its address.
    pub claim: ClaimId,
    /// `correction`, `fact` or `procedure`.
    pub kind: String,
    /// The statement.
    pub statement: String,
    /// The question it answers.
    pub question: String,
    /// The session it came from.
    pub session: SourceId,
    /// The person's words that support it.
    pub quotes: Vec<String>,
    /// The release that first absorbed it; absent until one has.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub absorbed_by: Option<ReleaseId>,
    /// How many times the person said it again: a signal that the model
    /// is still wrong.
    pub reinforced: usize,
}

impl ClaimLine {
    fn of(id: ClaimId, claim: &Claim, absorbed_by: Option<ReleaseId>, reinforced: usize) -> Self {
        Self {
            absorbed_by,
            reinforced,
            claim: id,
            kind: claim.kind.as_str().into(),
            statement: claim.statement.clone(),
            question: claim.question.clone(),
            session: claim.session.clone(),
            quotes: claim.quotes.iter().map(|q| q.text.clone()).collect(),
        }
    }
}

/// A claim that was replaced, and by which.
#[derive(Clone, Debug, Serialize)]
pub struct SupersededLine {
    /// The replaced claim.
    pub claim: ClaimId,
    /// Its statement.
    pub statement: String,
    /// The claim that replaced it.
    pub by: ClaimId,
}

/// A proposal that was refused.
#[derive(Clone, Debug, Serialize)]
pub struct RefusedLine {
    /// The claim set it was in.
    pub claim_set: Digest,
    /// Its position there.
    pub index: usize,
    /// The session it was proposed from.
    pub session: SourceId,
    /// The statement proposed.
    pub statement: String,
    /// The reason's stable name.
    pub reason_code: &'static str,
    /// The reason, in words.
    pub reason: String,
}

/// A claim the person asked to forget.
#[derive(Clone, Debug, Serialize)]
pub struct ForgottenLine {
    /// The claim.
    pub claim: ClaimId,
    /// Its statement.
    pub statement: String,
}

/// What `claims ledger` reports.
#[derive(Clone, Debug, Serialize)]
pub struct LedgerReport {
    /// Admitted claims that are still the answer to their question.
    pub live: Vec<ClaimLine>,
    /// Admitted claims that were replaced.
    pub superseded: Vec<SupersededLine>,
    /// Claims the person asked to forget: out of the live set, so out of
    /// the next training.
    pub forgotten: Vec<ForgottenLine>,
    /// Proposals that were refused.
    pub refused: Vec<RefusedLine>,
}

/// The ledger as it stands.
pub fn ledger(ctx: &Context) -> Result<LedgerReport, OrchestratorError> {
    let ledger = Ledger::new(ctx.claims().entries()?);
    let live = ledger.live().map_err(|e| gate_error(e.into()))?;
    let live_ids: Vec<&ClaimId> = live.iter().map(|(id, _)| id).collect();
    let absorbed: HashMap<ClaimId, ReleaseId> = ctx
        .claims()
        .absorptions()?
        .into_iter()
        .map(|a| (a.claim, a.release))
        .collect();
    let mut report = LedgerReport {
        live: live
            .iter()
            .map(|(id, claim)| {
                ClaimLine::of(
                    id.clone(),
                    claim,
                    absorbed.get(id).cloned(),
                    ledger.reinforcements(id),
                )
            })
            .collect(),
        superseded: Vec::new(),
        forgotten: Vec::new(),
        refused: Vec::new(),
    };
    for entry in ledger.entries() {
        match &entry.ruling {
            Ruling::Reinforced { .. } => {}
            Ruling::Forgotten { claim } => {
                if let Some(line) = ledger.entries().iter().find_map(|e| match &e.ruling {
                    Ruling::Admitted { claim: c, .. } if c.id().ok().as_ref() == Some(claim) => {
                        Some(c.statement.clone())
                    }
                    _ => None,
                }) {
                    report.forgotten.push(ForgottenLine {
                        claim: claim.clone(),
                        statement: line,
                    });
                }
            }
            Ruling::Admitted { claim, .. } => {
                let id = claim.id().map_err(|e| gate_error(e.into()))?;
                if live_ids.contains(&&id) {
                    continue;
                }
                if let Some(by) = ledger.superseded_by(&id) {
                    report.superseded.push(SupersededLine {
                        claim: id,
                        statement: claim.statement.clone(),
                        by,
                    });
                }
            }
            Ruling::Refused { reason } => report.refused.push(RefusedLine {
                claim_set: entry.claim_set.clone(),
                index: entry.index,
                session: entry.session.clone(),
                statement: entry.proposal.statement.clone(),
                reason_code: reason.code(),
                reason: reason.to_string(),
            }),
        }
    }
    Ok(report)
}

/// Asks for the live claim `id` (or a unique prefix of it) to be forgotten:
/// it leaves the live set, and so every later training. The ruling is
/// appended to the ledger; the claim and its evidence stay there.
pub fn forget(ctx: &Context, id: &str) -> Result<ForgottenLine, OrchestratorError> {
    let ledger = Ledger::new(ctx.claims().entries()?);
    let live = ledger.live().map_err(|e| gate_error(e.into()))?;
    let claim = ClaimId(ids::resolve(
        "live claim",
        id,
        live.iter().map(|(claim, _)| claim.0.clone()),
    )?);
    let statement = live
        .iter()
        .find(|(live_id, _)| *live_id == claim)
        .map(|(_, c)| c.statement.clone())
        .unwrap_or_default();
    let entry = ledger
        .forget(&claim)
        .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
    ctx.claims().append(&[entry])?;
    Ok(ForgottenLine { claim, statement })
}

/// One claim set, as `claims list` shows it.
#[derive(Clone, Debug, Serialize)]
pub struct ClaimSetSummary {
    /// Its address.
    pub id: Digest,
    /// The model that proposed the claims.
    pub extractor: String,
    /// Sessions read.
    pub sessions: usize,
    /// Proposals in it.
    pub proposals: usize,
    /// Sessions the extractor gave nothing usable for.
    pub failed: usize,
}

/// What `claims list` reports.
#[derive(Clone, Debug, Serialize)]
pub struct ClaimSetList {
    /// Every stored claim set, in id order.
    pub claim_sets: Vec<ClaimSetSummary>,
}

/// Every stored claim set.
pub fn list(ctx: &Context) -> Result<ClaimSetList, OrchestratorError> {
    let claims = ctx.claims();
    let claim_sets = claims
        .list_sets()?
        .into_iter()
        .map(|id| {
            let set = claims.get_set(&id)?;
            Ok(ClaimSetSummary {
                extractor: set.extractor,
                sessions: set.sessions.len(),
                proposals: set.sessions.iter().map(|s| s.proposals.len()).sum(),
                failed: set.sessions.iter().filter(|s| s.failure.is_some()).count(),
                id,
            })
        })
        .collect::<Result<_, OrchestratorError>>()?;
    Ok(ClaimSetList { claim_sets })
}

/// The stored claim set `id` (or a unique prefix of it) names.
pub fn resolve_set(ctx: &Context, id: &str) -> Result<Digest, OrchestratorError> {
    ids::resolve("claim set", id, ctx.claims().list_sets()?)
}

/// The claim set `id` names, whole.
pub fn show(ctx: &Context, id: &str) -> Result<ClaimSet, OrchestratorError> {
    Ok(ctx.claims().get_set(&resolve_set(ctx, id)?)?)
}
