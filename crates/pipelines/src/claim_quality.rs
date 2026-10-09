// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in continual
// learning from user feedback or admission of model-written training data,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! How well a night's extraction covered what its sessions were known to be
//! about.
//!
//! A recording names the fact a session was made to teach
//! (`extra.absorb_subject` of its trajectory) and a subjects file says what
//! that fact looks like in a statement
//! ([`splinter_knowledge::claims::coverage::Subject`]). For each such session
//! the report says whether the claims the extractor proposed, and the claims
//! the gates admitted, cover it. Recall is the share of those sessions whose
//! admitted claims cover the subject; precision is the share of the admitted
//! claims of those sessions that cover their own subject. A session whose
//! subject has no description is listed without a verdict; a ratio over nothing
//! is absent, never zero.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use splinter_core::claim::{ClaimSet, LedgerEntry, Ruling};
use splinter_core::source::SourceId;
use splinter_knowledge::claims::coverage::Subject;
use splinter_knowledge::session::SessionView;
use splinter_orchestrator::error::{io, OrchestratorError};

/// Reads subjects from JSON Lines files: `{"id": "...", "terms": ["..."]}`.
pub fn load_subjects(files: &[impl AsRef<Path>]) -> Result<Vec<Subject>, OrchestratorError> {
    let mut subjects = Vec::new();
    for file in files {
        let file = file.as_ref();
        let text = std::fs::read_to_string(file).map_err(io(file))?;
        for (index, line) in text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
        {
            subjects.push(serde_json::from_str(line).map_err(|source| {
                OrchestratorError::Json {
                    what: format!("{} line {}", file.display(), index + 1),
                    source,
                }
            })?);
        }
    }
    Ok(subjects)
}

/// One annotated session's coverage.
#[derive(Clone, Debug, Serialize)]
pub struct SessionCoverage {
    /// The session.
    pub session: SourceId,
    /// The fact it is known to be about.
    pub subject: String,
    /// Whether a claim the extractor proposed covers it; absent when the
    /// subject has no description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposed: Option<bool>,
    /// Whether a claim the gates admitted covers it; absent when the
    /// subject has no description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub admitted: Option<bool>,
}

/// What the extraction of a claim set covered.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionQuality {
    /// Every session that names a subject.
    pub sessions: Vec<SessionCoverage>,
    /// Of the sessions with a described subject, the share whose admitted
    /// claims cover it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall: Option<f64>,
    /// Of the claims admitted from those sessions, the share that cover the
    /// session's subject; absent when none was admitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision: Option<f64>,
    /// Claims admitted from the sessions with a described subject.
    pub admitted_claims: usize,
}

/// The coverage of `set`'s sessions, from the rulings made on it; `None`
/// when no session names a subject.
#[must_use]
pub fn assess(
    set: &ClaimSet,
    entries: &[LedgerEntry],
    views: &BTreeMap<SourceId, SessionView>,
    subjects: &[Subject],
) -> Option<ExtractionQuality> {
    let mut sessions = Vec::new();
    let (mut measured, mut covered, mut admitted_claims, mut covering) = (0, 0, 0, 0);
    for claims in &set.sessions {
        let Some(named) = views.get(&claims.session).and_then(SessionView::subject) else {
            continue;
        };
        let described = subjects.iter().find(|s| s.id == named);
        let proposed =
            described.map(|s| claims.proposals.iter().any(|p| s.covered_by(&p.statement)));
        let statements: Vec<&str> = entries
            .iter()
            .filter(|e| e.session == claims.session)
            .filter_map(|e| match &e.ruling {
                Ruling::Admitted { claim, .. } | Ruling::Reinforced { claim, .. } => {
                    Some(claim.statement.as_str())
                }
                _ => None,
            })
            .collect();
        let admitted = described.map(|s| statements.iter().any(|t| s.covered_by(t)));
        if let (Some(subject), Some(admitted)) = (described, admitted) {
            measured += 1;
            covered += usize::from(admitted);
            admitted_claims += statements.len();
            covering += statements.iter().filter(|t| subject.covered_by(t)).count();
        }
        sessions.push(SessionCoverage {
            session: claims.session.clone(),
            subject: named.to_string(),
            proposed,
            admitted,
        });
    }
    if sessions.is_empty() {
        return None;
    }
    let ratio = |part: usize, whole: usize| (whole > 0).then(|| part as f64 / whole as f64);
    Some(ExtractionQuality {
        sessions,
        recall: ratio(covered, measured),
        precision: ratio(covering, admitted_claims),
        admitted_claims,
    })
}
