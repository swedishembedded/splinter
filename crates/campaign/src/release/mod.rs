// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! `release`, `release list` and `rollback`: a trained candidate becomes
//! the policy only through the gate, and an alias moves back along the
//! lineage it came by.
//!
//! A candidate is decided against the release its alias points at - the
//! champion - and only if it was trained from that release (or, before any
//! release, from the base): a candidate trained from the base once a
//! champion exists, or from a champion since replaced, is refused before
//! anything is measured. The gate ([`gate`]) grades the candidate and the
//! champion (the base, before any release) closed-book on the same suites
//! ([`probe`]): the new data's held-out tasks, every earlier release's, and
//! the anchor suite ([`anchor`]), each without the tasks the candidate was
//! trained on ([`leakage`]); then plain brain serves the candidate
//! ([`serve`]). Only when every check passes is the release written
//! ([`store`]) - the adapter copied, the manifest recording every number -
//! and the alias moved to it, from the champion it was measured against.
//! Whatever the decision, the concepts of the tasks a failed retention
//! suite shows forgotten are queued for new tasks
//! ([`crate::curriculum::queue`]).
//!
//! `rollback` points an alias at the release its current one was trained
//! from, and refuses when there is none.

pub mod anchor;
pub mod gate;
pub mod leakage;
pub mod probe;
pub mod serve;
pub mod store;

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;
use splinter_knowledge::concepts::Concept;
use splinter_policy::local::resolve_base;
use splinter_policy::selection::local_model_name;
use splinter_store::digest::Digest;
use sven_sdk::CancelToken;

pub use store::{ReleaseId, ReleaseManifest, ReleaseStore, StoredRelease, RELEASE_FORMAT};

use crate::config::Config;
use crate::context::Context;
use crate::curriculum::queue::enqueue_retention;
use crate::error::{io, CampaignError};
use crate::model_ref::{is_alias_name, ModelRef, POLICY_DEFAULT};
use crate::train::{load_candidate, Candidate, TrainingSummary};
use gate::{Check, GateConfig, GateReport};
use probe::{pair, Suite};

/// One `release`.
#[derive(Clone, Debug, Serialize)]
pub struct ReleaseRequest {
    /// The candidate, by id or unique prefix.
    pub candidate: String,
    /// The alias it is released under.
    pub alias: String,
    /// The gate's thresholds.
    pub gate: GateConfig,
}

impl ReleaseRequest {
    /// `candidate` released under `default` through the default gate.
    #[must_use]
    pub fn new(candidate: impl Into<String>) -> Self {
        Self {
            candidate: candidate.into(),
            alias: POLICY_DEFAULT.into(),
            gate: GateConfig::default(),
        }
    }
}

/// What `release` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Released {
    /// The candidate.
    pub candidate: String,
    /// The alias.
    pub alias: String,
    /// The release it was measured against; `None` before any release,
    /// when the base is the champion.
    pub champion: Option<ReleaseId>,
    /// The gate, with every number.
    pub gate: GateReport,
    /// The concepts the candidate forgot on a retention suite that failed,
    /// queued for new tasks.
    pub requeued: Vec<Concept>,
    /// The release written; `None` when the gate blocked it.
    pub release: Option<ReleaseId>,
    /// The release's directory.
    pub dir: Option<PathBuf>,
}

/// The reference the gate grades the base with `adapter` through: an
/// absolute `local:` checkpoint, so a caller holding a model for it (a
/// test's scripted one) can hand it in under the same reference.
#[must_use]
pub fn arm(config: &Config, adapter: Option<&Path>) -> ModelRef {
    let base =
        std::path::absolute(&config.policy_base).unwrap_or_else(|_| config.policy_base.clone());
    ModelRef::Local {
        checkpoint: base.display().to_string(),
        adapter: adapter.map(|a| a.display().to_string()),
    }
}

/// Runs the gate on `request.candidate` and releases it if it passes.
pub fn release(
    ctx: &Context,
    request: &ReleaseRequest,
    cancel: &CancelToken,
) -> Result<Released, CampaignError> {
    if !is_alias_name(&request.alias) {
        return Err(CampaignError::Refused(format!(
            "{:?} is not an alias name",
            request.alias
        )));
    }
    let candidate = load_candidate(ctx, &request.candidate)?;
    let store = ReleaseStore::open(ctx.root());
    let champion_id = store.alias(&request.alias)?;
    if candidate.parent != champion_id {
        let named = |id: &Option<ReleaseId>| {
            id.as_ref()
                .map_or("the base alone".to_string(), |id| format!("release {id}"))
        };
        return Err(CampaignError::Refused(format!(
            "candidate {} was trained from {}, but {} points at {}; only a candidate trained \
             from the champion can replace it: train again from policy:{}",
            candidate.candidate,
            named(&candidate.parent),
            request.alias,
            named(&champion_id),
            request.alias
        )));
    }
    let config = ctx.config();
    if candidate.base != config.policy_base {
        return Err(CampaignError::Refused(format!(
            "candidate {} sits on {}, not the policy base {}",
            candidate.candidate,
            candidate.base.display(),
            config.policy_base.display()
        )));
    }
    let champion = champion_id.as_ref().map(|id| store.get(id)).transpose()?;
    let base_file = resolve_base(&config.policy_base)
        .map_err(|e| CampaignError::Refused(format!("the policy base: {e}")))?;
    let gate = run_gate(
        ctx,
        &candidate,
        champion.as_ref(),
        &base_file,
        &request.gate,
        cancel,
    )?;
    let requeued = enqueue_retention(ctx, &gate)?;
    let mut released = Released {
        candidate: candidate.candidate.clone(),
        alias: request.alias.clone(),
        champion: champion_id.clone(),
        gate,
        requeued,
        release: None,
        dir: None,
    };
    if !released.gate.passed {
        return Ok(released);
    }
    let manifest = manifest(ctx, &candidate, &released.gate)?;
    let stored = store.put(&manifest, &candidate.adapter)?;
    store.move_alias(&request.alias, champion_id.as_ref(), &stored.id)?;
    ctx.repin_policy(&request.alias);
    released.release = Some(stored.id);
    released.dir = Some(stored.dir);
    Ok(released)
}

/// The manifest of `candidate` released after `gate`. The base digest is
/// the one training recorded on the adapter's card: the serve check has
/// just had brain bind the adapter to the base on disk, which it refuses
/// for any other base, so it is the base the release runs on.
fn manifest(
    ctx: &Context,
    candidate: &Candidate,
    gate: &GateReport,
) -> Result<ReleaseManifest, CampaignError> {
    let base_digest = Digest::parse(&candidate.base_digest)
        .map_err(|e| CampaignError::Refused(format!("candidate {}: {e}", candidate.candidate)))?;
    let adapter_digest = Digest::parse(&candidate.adapter_digest)
        .map_err(|e| CampaignError::Refused(format!("candidate {}: {e}", candidate.candidate)))?;
    let record_path = &candidate.training_record;
    let record_text = std::fs::read_to_string(record_path).map_err(io(record_path))?;
    let record = serde_json::from_str(&record_text).map_err(|source| CampaignError::Json {
        what: record_path.display().to_string(),
        source,
    })?;
    Ok(ReleaseManifest {
        format: RELEASE_FORMAT.into(),
        base_model: local_model_name(&ctx.config().policy_base),
        base_digest,
        adapter_digest,
        parent: candidate.parent.clone(),
        candidate: candidate.candidate.clone(),
        datasets: candidate.datasets.clone(),
        replay: candidate.replay.clone(),
        training: TrainingSummary {
            from: candidate.from.clone(),
            steps: candidate.steps,
            rank: candidate.rank,
            records: candidate.records,
            regime: candidate.regime,
            base_score: candidate.base_score,
            tuned_score: candidate.tuned_score,
            preference: candidate.preference.clone(),
            record,
        },
        gate: gate.clone(),
        created_at: ctx.clock().utc_now(),
    })
}

/// One model's outcome on each task of a suite.
type Outcomes = Vec<Option<bool>>;

/// A suite's outcomes for one model, or why there are none.
type Graded = Result<Outcomes, String>;

/// `result`, with every failure but a cancel turned into its reason: a
/// check that cannot be measured fails the gate, it does not abort it.
fn soft<T>(result: Result<T, CampaignError>) -> Result<Result<T, String>, CampaignError> {
    match result {
        Ok(value) => Ok(Ok(value)),
        Err(CampaignError::Cancelled) => Err(CampaignError::Cancelled),
        Err(e) => Ok(Err(e.to_string())),
    }
}

/// The model `reference` names graded on each of `suites`, decoding
/// greedily ([`probe::greedy`]) where its sampling can be set here - a
/// verdict is then the weights', not a draw's, and the served candidate is
/// asked the same way. The arms differ only by adapter, so the second one
/// graded runs on the base the first one loaded; the serve check releases
/// it before the served candidate starts.
fn grade_arm(
    ctx: &Context,
    reference: &ModelRef,
    suites: &[&Suite],
    cancel: &CancelToken,
) -> Result<Vec<Graded>, CampaignError> {
    let model = match soft(probe::greedy(ctx, reference))? {
        Ok(model) => model,
        Err(why) => return Ok(suites.iter().map(|_| Err(why.clone())).collect()),
    };
    let mut graded = Vec::with_capacity(suites.len());
    for suite in suites {
        graded.push(soft(probe::grade(ctx, &model, suite, cancel))?);
    }
    Ok(graded)
}

/// The suites the gate grades on, each built or with why it could not be.
struct Suites {
    held_out: Result<Suite, String>,
    retention: Result<Vec<(ReleaseId, Suite)>, String>,
    anchor: Result<Option<anchor::FrozenAnchor>, String>,
    anchor_suite: Option<Suite>,
}

impl Suites {
    fn build(
        ctx: &Context,
        candidate: &Candidate,
        champion: Option<&StoredRelease>,
    ) -> Result<Self, CampaignError> {
        let trained = match soft(leakage::trained_prompts(ctx, candidate))? {
            Ok(trained) => trained,
            Err(why) => {
                let why = format!("the candidate's training records: {why}");
                return Ok(Self {
                    held_out: Err(why.clone()),
                    retention: Err(why.clone()),
                    anchor: Err(why),
                    anchor_suite: None,
                });
            }
        };
        let clean = |mut suite: Suite| {
            leakage::exclude_leaked(&mut suite, &trained);
            suite
        };
        let held_out = soft(probe::held_out(ctx, "held-out", &candidate.datasets))?.map(clean);
        let retention = soft(champion.map_or(Ok(Vec::new()), |champion| {
            ReleaseStore::open(ctx.root())
                .lineage(&champion.id)?
                .into_iter()
                .map(|release| {
                    let name = format!("retention {}", release.id);
                    probe::held_out(ctx, name, &release.manifest.datasets)
                        .map(|suite| (release.id, clean(suite)))
                })
                .collect()
        }))?;
        let anchor = soft(anchor::current(ctx))?;
        let anchor_suite = anchor
            .as_ref()
            .ok()
            .and_then(|a| a.as_ref())
            .map(|frozen| clean(frozen.probe_suite()));
        Ok(Self {
            held_out,
            retention,
            anchor,
            anchor_suite,
        })
    }

    /// Every suite built, in a fixed order: held-out, retention, anchor.
    fn all(&self) -> Vec<&Suite> {
        let mut all: Vec<&Suite> = self.held_out.iter().collect();
        if let Ok(retention) = &self.retention {
            all.extend(retention.iter().map(|(_, suite)| suite));
        }
        all.extend(self.anchor_suite.iter());
        all
    }
}

/// The four checks on `candidate` against `champion`.
fn run_gate(
    ctx: &Context,
    candidate: &Candidate,
    champion: Option<&StoredRelease>,
    base_file: &Path,
    config: &GateConfig,
    cancel: &CancelToken,
) -> Result<GateReport, CampaignError> {
    let suites = Suites::build(ctx, candidate, champion)?;
    let all = suites.all();
    let candidate_ref = arm(ctx.config(), Some(&candidate.adapter));
    let champion_ref = arm(ctx.config(), champion.map(|c| c.adapter.as_path()));
    let mut theirs = grade_arm(ctx, &candidate_ref, &all, cancel)?.into_iter();
    let mut ours = grade_arm(ctx, &champion_ref, &all, cancel)?.into_iter();
    let mut next = || -> Result<(Outcomes, Outcomes), String> {
        // Both arms graded the same suites, so both iterators hold one
        // entry per suite.
        let (Some(c), Some(b)) = (theirs.next(), ours.next()) else {
            unreachable!("both arms are graded on every suite");
        };
        Ok((c?, b?))
    };

    let mut in_process: Option<Outcomes> = None;
    let improvement = match &suites.held_out {
        Err(why) => Check::unmeasured(why.clone()),
        Ok(suite) => match next() {
            Err(why) => Check::unmeasured(why),
            Ok((c, b)) => {
                let check = gate::improvement(suite.summary(), &pair(suite, &c, &b), config.alpha);
                in_process = Some(c);
                check
            }
        },
    };
    let retention = match &suites.retention {
        Err(why) => Check::unmeasured(why.clone()),
        Ok(releases) => {
            let mut measured = Vec::new();
            let mut failure = None;
            for (release, suite) in releases {
                match next() {
                    Ok((c, b)) => {
                        measured.push((release.clone(), suite.summary(), pair(suite, &c, &b)))
                    }
                    Err(why) => failure = failure.or(Some(format!("{}: {why}", suite.name))),
                }
            }
            match failure {
                Some(why) => Check::unmeasured(why),
                None => gate::retention(measured, config.retention_bound),
            }
        }
    };
    let anchor = match (&suites.anchor, &suites.anchor_suite) {
        (Err(why), _) => Check::unmeasured(why.clone()),
        (Ok(None), _) | (_, None) => Check::unmeasured(
            "no anchor suite is frozen: `splinter eval --suite anchor --freeze FILE` freezes one",
        ),
        (Ok(Some(frozen)), Some(suite)) => match next() {
            Err(why) => Check::unmeasured(why),
            Ok((c, b)) => gate::anchor(
                frozen.suite.version,
                frozen.digest.clone(),
                suite.summary(),
                &pair(suite, &c, &b),
                config.anchor_bound,
            ),
        },
    };
    let serve = match (&suites.held_out, &in_process) {
        (Ok(held_out), Some(verdicts)) => {
            let take = config.serve_sample.min(held_out.tasks.len());
            let sample = Suite::of_tasks(
                format!("{} (served sample)", held_out.name),
                held_out.tasks[..take].to_vec(),
            );
            serve::check(
                ctx,
                ctx.config().brain_binary.as_deref(),
                base_file,
                &candidate.adapter,
                &candidate.adapter_digest,
                &sample,
                &verdicts[..take],
                Duration::from_secs(config.serve_startup_secs),
                cancel,
            )?
        }
        _ => Check::unmeasured("the held-out tasks were not graded in-process to compare with"),
    };
    Ok(GateReport::new(
        *config,
        improvement,
        retention,
        anchor,
        serve,
    ))
}

/// One release, as `release list` shows it.
#[derive(Clone, Debug, Serialize)]
pub struct ReleaseLine {
    /// Its id.
    pub id: ReleaseId,
    /// When it was released.
    pub created_at: String,
    /// The candidate it was.
    pub candidate: String,
    /// The release it was trained from.
    pub parent: Option<ReleaseId>,
    /// Its adapter's digest.
    pub adapter_digest: Digest,
    /// The aliases pointing at it.
    pub aliases: Vec<String>,
}

/// What `release list` reports.
#[derive(Clone, Debug, Serialize)]
pub struct ReleaseList {
    /// Every release, oldest first.
    pub releases: Vec<ReleaseLine>,
}

/// Every release under the state root, verified, oldest first.
pub fn list(ctx: &Context) -> Result<ReleaseList, CampaignError> {
    let store = ReleaseStore::open(ctx.root());
    let aliases = store.aliases()?;
    let mut releases = Vec::new();
    for id in store.list()? {
        let release = store.get(&id)?;
        releases.push(ReleaseLine {
            aliases: aliases
                .iter()
                .filter(|(_, target)| **target == id)
                .map(|(name, _)| name.clone())
                .collect(),
            created_at: release.manifest.created_at,
            candidate: release.manifest.candidate,
            parent: release.manifest.parent,
            adapter_digest: release.manifest.adapter_digest,
            id,
        });
    }
    releases.sort_by(|a, b| (&a.created_at, &a.id).cmp(&(&b.created_at, &b.id)));
    Ok(ReleaseList { releases })
}

/// What `rollback` reports.
#[derive(Clone, Debug, Serialize)]
pub struct RolledBack {
    /// The alias.
    pub alias: String,
    /// The release it pointed at.
    pub from: ReleaseId,
    /// The release it points at now: the one `from` was trained from.
    pub to: ReleaseId,
}

/// Points `alias` at the release its current one was trained from.
pub fn rollback(ctx: &Context, alias: &str) -> Result<RolledBack, CampaignError> {
    let store = ReleaseStore::open(ctx.root());
    let Some(from) = store.alias(alias)? else {
        return Err(CampaignError::Refused(format!(
            "alias {alias} points at no release; there is nothing to roll back"
        )));
    };
    let Some(to) = store.get(&from)?.manifest.parent else {
        return Err(CampaignError::Refused(format!(
            "release {from} is the first {alias} has had; there is no previous release to roll \
             back to"
        )));
    };
    store.move_alias(alias, Some(&from), &to)?;
    ctx.repin_policy(alias);
    Ok(RolledBack {
        alias: alias.into(),
        from,
        to,
    })
}
