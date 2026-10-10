// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in continual
// learning from user feedback, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The stages of `absorb` that take the sessions in and have a model propose
//! the claims they teach.

use std::collections::BTreeMap;

use splinter_agent::claims::ClaimExtractor;
use splinter_knowledge::claims::extract::ExtractionPolicy;
use splinter_orchestrator::pipeline::StageEnd;
use splinter_orchestrator::runs::Recorder;
use splinter_orchestrator::Context;

use super::{value, Done, State};
use crate::claims::{extract, ClaimsExtracted, ExtractRequest, Unconfirmed};
use crate::sessions::{intake, IntakeRequest};

pub(super) fn intake_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let taken = intake(
        ctx,
        &IntakeRequest {
            paths: &st.request.sessions,
            max_bytes: st.request.max_session_bytes,
        },
    )?;
    st.sessions = taken.sessions.iter().map(|s| s.source.clone()).collect();
    let summary = value(&taken)?;
    let none = st.sessions.is_empty();
    st.report.intake = Some(taken);
    Ok(if none {
        StageEnd::stop(summary, "no session was taken in")
    } else {
        StageEnd::done(summary)
    })
}

pub(super) fn extract_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let identity = ClaimExtractor::new(ctx.model(&st.generator)?)
        .identity()
        .to_string();
    let claims = ctx.claims();
    let mut reused = None;
    for id in claims.list_sets()? {
        let set = claims.get_set(&id)?;
        let same_sessions = set
            .sessions
            .iter()
            .map(|s| &s.session)
            .eq(st.sessions.iter());
        if same_sessions
            && set.passes == st.request.extraction_passes
            && set.extractor == identity
            && set.sessions.iter().all(|s| s.failure.is_none())
        {
            reused = Some((id, set));
        }
    }
    let extracted = match reused {
        Some((claim_set, set)) => {
            let mut by_kind = BTreeMap::new();
            for p in set.sessions.iter().flat_map(|s| &s.proposals) {
                *by_kind.entry(p.kind.as_str().to_string()).or_default() += 1;
            }
            ClaimsExtracted {
                claim_set,
                sessions: set.sessions.len(),
                passes: set.passes,
                proposals: set.sessions.iter().map(|s| s.proposals.len()).sum(),
                disagreements: set.sessions.iter().map(|s| s.unconfirmed.len()).sum(),
                unconfirmed: set
                    .sessions
                    .iter()
                    .flat_map(|s| {
                        s.unconfirmed.iter().map(|p| Unconfirmed {
                            session: s.session.clone(),
                            statement: p.statement.clone(),
                        })
                    })
                    .collect(),
                by_kind,
                failed: Vec::new(),
                stopped: None,
            }
        }
        None => extract(
            ctx,
            &ExtractRequest {
                sessions: &st.sessions,
                extractor: &st.generator,
                policy: ExtractionPolicy::default(),
                passes: st.request.extraction_passes,
                deadline: None,
                cancel: run.cancel_token(),
            },
        )?,
    };
    let summary = value(&extracted)?;
    st.report.extract = Some(extracted);
    Ok(StageEnd::done(summary))
}
