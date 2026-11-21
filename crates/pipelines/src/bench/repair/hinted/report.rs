// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// measuring how much self-generated training data survives verification you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a rationalisation run reports: the attempts through the filters in
//! order and each alone, the controls the student's own passes give, and
//! the experience set of everything kept.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;
use splinter_core::experience::ExperienceId;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::{ExperienceSet, SetId};

use super::filters::{self, Verdict};
use super::RATIONALISE_FORMAT;
use super::{refused, Filters, Origin, Outcome, Picked, RationaliseRecord, RationaliseRequest};

/// How many attempts reached each filter, in order: each count is of the
/// ones before it.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Funnel {
    /// Attempts graded (not stopped by the infrastructure).
    pub graded: usize,
    /// Verified.
    pub verified: usize,
    /// Verified, with real observations.
    pub real_observations: usize,
    /// ... and no leak.
    pub no_leak: usize,
    /// ... and sufficient: kept.
    pub sufficient: usize,
}

/// How many graded attempts passed each of the first three filters, each
/// on its own.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Alone {
    /// Verified.
    pub verified: usize,
    /// Real observations.
    pub real_observations: usize,
    /// No leak.
    pub no_leak: usize,
}

/// What the student's jobs held.
#[derive(Clone, Debug, Default, Serialize)]
pub struct StudentSummary {
    /// Trials read.
    pub trials: usize,
    /// Passed.
    pub passed: usize,
    /// Graded and not passed.
    pub failed: usize,
    /// Tasks failed at least once.
    pub failed_tasks: usize,
}

/// What a rationalisation run did.
#[derive(Clone, Debug, Serialize)]
pub struct RationaliseReport {
    /// Always [`RATIONALISE_FORMAT`].
    pub format: &'static str,
    /// The rationaliser's identity, and the id its endpoint serves it under.
    pub rationaliser: String,
    /// The id the endpoint lists it under.
    pub served_as: String,
    /// The dataset, its compile record's sha256 and the examiner commit
    /// that compiled it.
    pub dataset: serde_json::Value,
    /// The profile the rationaliser's was derived from, and its sha256.
    pub profile: PathBuf,
    /// Its sha256.
    pub profile_sha256: String,
    /// The seed of the pick, the count asked for and the samples per task.
    pub seed: u64,
    /// Failed tasks asked for.
    pub count: usize,
    /// Attempts per task.
    pub samples: usize,
    /// The leak terms.
    pub leak_terms: Vec<String>,
    /// How the hint was framed.
    pub hint_framing: filters::HintFraming,
    /// What the student's jobs held.
    pub students: StudentSummary,
    /// The tasks picked.
    pub picked: Vec<Picked>,
    /// The attempts through the filters in order.
    pub funnel: Funnel,
    /// The graded attempts through each of the first three filters alone.
    pub alone: Alone,
    /// Attempts the infrastructure stopped, or still waiting for their
    /// sufficiency run.
    pub undecided: usize,
    /// Kept experiences of the student's own passes.
    pub own: usize,
    /// How many of those mention each leak term in the model's own text
    /// ([`filters::mentioned_terms`]): passes that never saw a hint, so the
    /// rate at which the term check rejects unhinted work. A term most of
    /// them mention (one the instruction itself uses) rejects nearly every
    /// attempt whatever it learned from the hint.
    pub own_term_mentions: BTreeMap<String, usize>,
    /// How many of the student's own passes would pass the real-observations
    /// filter ([`filters::read_before_write`]): the same control for the
    /// second filter.
    pub own_real_observations: usize,
    /// Kept rationalisations.
    pub rationalised: usize,
    /// Every record of the run: the student's own passes and every
    /// attempt, kept or not.
    pub records: Vec<RationaliseRecord>,
    /// The kept experiences, own and rationalised: the set a dataset is
    /// built from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub set: Option<SetId>,
    /// The run's directory.
    pub dir: PathBuf,
}

/// What the report states besides its records.
pub(super) struct ReportParts {
    pub(super) rationaliser: String,
    pub(super) served_as: String,
    pub(super) dataset: serde_json::Value,
    pub(super) profile_sha256: String,
    pub(super) students: StudentSummary,
    pub(super) picked: Vec<Picked>,
    /// Each task's files and instruction, by folder, for the controls over own passes.
    pub(super) files: BTreeMap<String, (filters::TaskFiles, String)>,
    pub(super) dir: PathBuf,
}

/// The run's report over `records`, with the set of its kept experiences,
/// also written as `report.json` in its directory.
pub(super) fn report(
    ctx: &Context,
    request: &RationaliseRequest<'_>,
    records: Vec<RationaliseRecord>,
    parts: ReportParts,
) -> Result<RationaliseReport, OrchestratorError> {
    let attempts = || records.iter().filter(|r| r.origin == Origin::Rationalised);
    let graded: Vec<&RationaliseRecord> = attempts()
        .filter(|r| r.filters.verified.is_some())
        .collect();
    let passes = |r: &&RationaliseRecord, f: fn(&Filters) -> &Option<Verdict>| {
        f(&r.filters).as_ref().is_some_and(|v| v.passed)
    };
    let mut funnel = Funnel {
        graded: graded.len(),
        ..Funnel::default()
    };
    let mut alone = Alone::default();
    for r in &graded {
        let v = passes(r, |f| &f.verified);
        let o = passes(r, |f| &f.real_observations);
        let l = passes(r, |f| &f.no_leak);
        let s = passes(r, |f| &f.sufficient);
        alone.verified += usize::from(v);
        alone.real_observations += usize::from(o);
        alone.no_leak += usize::from(l);
        funnel.verified += usize::from(v);
        funnel.real_observations += usize::from(v && o);
        funnel.no_leak += usize::from(v && o && l);
        funnel.sufficient += usize::from(v && o && l && s);
    }
    let mut own_term_mentions: BTreeMap<String, usize> =
        request.leak_terms.iter().map(|t| (t.clone(), 0)).collect();
    let mut own_real_observations = 0;
    let store = ctx.experiences();
    for record in records
        .iter()
        .filter(|r| r.origin == Origin::Own && r.outcome == Outcome::Kept)
    {
        let Some(id) = &record.experience else {
            continue;
        };
        let experience = store.get(id)?;
        for term in filters::mentioned_terms(&experience.trajectory, request.leak_terms) {
            *own_term_mentions.entry(term.to_string()).or_default() += 1;
        }
        own_real_observations += usize::from(parts.files.get(&record.folder).is_some_and(
            |(files, instruction)| {
                filters::read_before_write(&experience.trajectory, files, instruction).passed
            },
        ));
    }
    let kept: Vec<ExperienceId> = records
        .iter()
        .filter(|r| r.outcome == Outcome::Kept)
        .filter_map(|r| r.experience.clone())
        .collect();
    let set = if kept.is_empty() {
        None
    } else {
        Some(store.put_set(&ExperienceSet {
            name: format!("rationalised and own passes by {}", parts.rationaliser),
            members: kept,
        })?)
    };
    let report = RationaliseReport {
        format: RATIONALISE_FORMAT,
        rationaliser: parts.rationaliser,
        served_as: parts.served_as,
        dataset: parts.dataset,
        profile: request.profile.to_path_buf(),
        profile_sha256: parts.profile_sha256,
        seed: request.seed,
        count: request.count,
        samples: request.samples,
        leak_terms: request.leak_terms.to_vec(),
        hint_framing: request.framing,
        students: parts.students,
        picked: parts.picked,
        funnel,
        alone,
        undecided: attempts().filter(|r| !r.outcome.decided()).count(),
        own: records
            .iter()
            .filter(|r| r.origin == Origin::Own && r.outcome == Outcome::Kept)
            .count(),
        own_term_mentions,
        own_real_observations,
        rationalised: attempts().filter(|r| r.outcome == Outcome::Kept).count(),
        records,
        set,
        dir: parts.dir,
    };
    let path = report.dir.join("report.json");
    let text =
        serde_json::to_string_pretty(&report).map_err(|e| refused(format!("the report: {e}")))?;
    std::fs::write(&path, text).map_err(|e| refused(format!("{}: {e}", path.display())))?;
    Ok(report)
}
