// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! `learn "<objective>" --source <url|path>`: an objective and a body of
//! documentation become a measured, trained, assessed set of skills.
//!
//! The controller composes the stages that already exist (the task
//! generator, the solve, verify, teach and frontier stages, the dataset
//! projection, training and the release gate) under one decision loop; it
//! adds no scheduler, no task store and no training code.
//!
//! 1. **Sources** - files, directories and web pages are taken in
//!    ([`intake`]).
//! 2. **Skill graph** - the sources' sections become skills with the
//!    constraints and requirements the text states and the prerequisites
//!    between them ([`graph`]), planned by a model and admitted by code, or
//!    read by rule; versioned.
//! 3. **Probes** - every skill gets tasks written from its own sections,
//!    each checked to exercise it, split into **dev** probes that measure
//!    the policy as rounds go and **final** probes that only the closing
//!    assessment touches; no training task may repeat one ([`probes`]).
//! 4. **Baseline** - the policy as it is is measured on every skill's dev
//!    probes: the independent baseline competence ([`competence`]).
//! 5. **Rounds** - the skill the competence graph says will benefit most
//!    ([`select`]: prerequisites first, then room to improve weighted by
//!    what the skill unlocks) is trained: tasks, attempts, a teacher's
//!    verified answers where the policy fails, a dataset, a candidate; it
//!    is accepted when the skill improved and nothing measured before fell
//!    past the retention bound ([`round`]). The competence graph is updated
//!    and kept, revision by revision.
//! 6. **Assessment** - when every skill is mastered, no skill can be chosen,
//!    or the budget or the round limit is spent, the final probes measure the
//!    policy the run began with and the one it ends with, skill by skill.
//!    The run is accepted only if every assessed skill reaches the target on
//!    probes that never decided anything.
//! 7. **Release** - an accepted run's candidate goes through the release
//!    gate like any other; an incumbent is only ever replaced there.

mod competence;
mod cost;
mod graph;
mod intake;
mod probes;
mod round;
mod select;

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_core::model_ref::{ModelRef, POLICY_DEFAULT};
use splinter_core::role::{Role, RoleOverrides};
use splinter_eval::gate::{GateConfig, GatePlan};
use splinter_knowledge::skills::SkillGraph;
use splinter_knowledge::tasks::dedup::Seen;

pub use competence::{wilson, CompetenceGraph, SkillCompetence};
pub use cost::StageCost;
pub use graph::{plan, Planned, PlannedBy};
pub use intake::{intake, is_url, Fetched, Intaken};
pub use probes::{ProbeScore, SkillProbes};
pub use round::{Change, Outcome, Round};
pub use select::{next_skill, Selection, Targets};

use crate::curriculum::frontier::PassAtK;
use crate::learn::PolicyUsed;
use crate::release::{release_recorded, ReleaseRequest, Released};
use crate::tasks::{check_kinds, DEFAULT_LEARN_KINDS};
use crate::train::{Candidate, FullCandidate, Trainer, Tuning, DEFAULT_REPLAY_FRACTION};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::roles;
use splinter_orchestrator::runs::{record, Recorded, Recorder};

/// The fewest graded probes a skill's competence rests on, when a request
/// names none.
pub const DEFAULT_MIN_PROBES: usize = 3;

/// The pass rate a skill must reach to be mastered, when a request names
/// none.
pub const DEFAULT_MASTERY: f64 = 0.7;

/// Rounds one skill may take, when a request names none.
pub const DEFAULT_MAX_ROUNDS_PER_SKILL: usize = 2;

/// Rounds a run may take in all, when a request names none.
pub const DEFAULT_MAX_ROUNDS: usize = 8;

/// How far a skill measured earlier may fall in a round before the round's
/// candidate is rejected, when a request names none.
pub const DEFAULT_RETENTION_DROP: f64 = 0.25;

/// One acquisition.
#[derive(Clone, Debug, Serialize)]
pub struct AcquireRequest {
    /// What the learner is after.
    pub objective: String,
    /// The sources: files, directories, `cmd:` commands, stored ids, URLs.
    pub sources: Vec<String>,
    /// The task kinds probes and training tasks are of; empty is
    /// [`DEFAULT_LEARN_KINDS`].
    pub kinds: Vec<String>,
    /// Probes per skill in each half (dev and final).
    pub probes_per_skill: usize,
    /// k and the sampling of the policy's attempts at training tasks.
    pub pass_at_k: PassAtK,
    /// The pass rate a skill must reach.
    pub mastery: f64,
    /// The graded probes a skill's competence must rest on.
    pub min_probes: usize,
    /// Rounds one skill may take.
    pub max_rounds_per_skill: usize,
    /// Rounds in all.
    pub max_rounds: usize,
    /// How far a skill may fall in a round.
    pub retention_drop: f64,
    /// Wall-clock time the whole run may take.
    pub budget: Option<Duration>,
    /// Stop with the final candidate: do not run the release gate.
    pub no_release: bool,
    /// The release gate's statistical settings.
    pub gate: GatePlan,
    /// The step budget of each training; sized from the examples when `None`.
    pub steps: Option<u32>,
    /// LoRA rank of the adapter.
    pub rank: Option<u32>,
    /// Train every weight instead of an adapter. Each round starts from the
    /// incumbent checkpoint (the start base in round 1, the last accepted
    /// round's checkpoint after) on the round's records and a replay of the
    /// accepted rounds' records; the candidate is the checkpoint directory,
    /// measured as `local:<directory>`. The release gate takes adapters, so
    /// a request that trains every weight must also stop before it
    /// ([`Self::no_release`]).
    pub full_fine_tune: bool,
    /// brain's training precision of a full fine-tune (`state=int8,sr=on`);
    /// checked by brain's parser. `None` is brain's default.
    pub train_precision: Option<String>,
    /// Keep the checkpoints of rejected and superseded full-fine-tune
    /// rounds; they are deleted otherwise, being GBs each.
    pub keep_candidates: bool,
    /// The fraction of each accepted round's trained-on records replayed
    /// into the next full-fine-tune round (a quarter of its draws).
    pub replay_fraction: f64,
    /// Let the incumbent write every round's training tasks: the policy the
    /// run began with in round 1, the last accepted candidate after. Probes
    /// are written once before any training by the configured generator
    /// either way.
    pub self_generate: bool,
    /// Read the skill graph from the sources' structure instead of asking
    /// the planner.
    pub structural_graph: bool,
    /// Teach a task the policy never solves with its own reference answer
    /// (a computed output, a reference program), graded like any answer,
    /// instead of asking the teacher model.
    pub reference_teacher: bool,
    /// How training runs.
    pub tuning: Tuning,
    /// The models named for roles.
    pub roles: RoleOverrides,
}

impl AcquireRequest {
    /// A request for `objective` from `sources`, every other setting at its
    /// default.
    #[must_use]
    pub fn new(objective: impl Into<String>, sources: Vec<String>) -> Self {
        Self {
            objective: objective.into(),
            sources,
            kinds: Vec::new(),
            probes_per_skill: 4,
            pass_at_k: PassAtK::default(),
            mastery: DEFAULT_MASTERY,
            min_probes: DEFAULT_MIN_PROBES,
            max_rounds_per_skill: DEFAULT_MAX_ROUNDS_PER_SKILL,
            max_rounds: DEFAULT_MAX_ROUNDS,
            retention_drop: DEFAULT_RETENTION_DROP,
            budget: None,
            no_release: false,
            gate: GatePlan::default(),
            steps: None,
            rank: None,
            full_fine_tune: false,
            train_precision: None,
            keep_candidates: false,
            replay_fraction: DEFAULT_REPLAY_FRACTION,
            self_generate: false,
            structural_graph: false,
            reference_teacher: false,
            tuning: Tuning::default(),
            roles: RoleOverrides::default(),
        }
    }

    fn validate(&self) -> Result<(), OrchestratorError> {
        if self.objective.trim().is_empty() {
            return Err(OrchestratorError::Refused(
                "state the objective: what is to be learned".into(),
            ));
        }
        if self.sources.is_empty() {
            return Err(OrchestratorError::Refused(
                "name at least one source: a file, a directory or a URL".into(),
            ));
        }
        let share = |name: &str, v: f64| {
            if v.is_finite() && (0.0..=1.0).contains(&v) {
                Ok(())
            } else {
                Err(OrchestratorError::Refused(format!(
                    "{name} is {v}, not a share in [0, 1]"
                )))
            }
        };
        share("the mastery target", self.mastery)?;
        share("the retention bound", self.retention_drop)?;
        share("the replay fraction", self.replay_fraction)?;
        self.validate_training()?;
        if self.probes_per_skill < self.min_probes || self.min_probes == 0 {
            return Err(OrchestratorError::Refused(format!(
                "{} probe(s) per half cannot rest a competence on {} graded probe(s)",
                self.probes_per_skill, self.min_probes
            )));
        }
        if self.max_rounds == 0 || self.max_rounds_per_skill == 0 {
            return Err(OrchestratorError::Refused(
                "a run takes at least one round, and a skill at least one".into(),
            ));
        }
        self.pass_at_k.validate()
    }

    /// Refuses a combination of training options that cannot be carried out
    /// or would be silently ignored.
    fn validate_training(&self) -> Result<(), OrchestratorError> {
        let refuse = |why: &str| Err(OrchestratorError::Refused(why.into()));
        if !self.full_fine_tune {
            if self.train_precision.is_some() {
                return refuse("a training precision applies to a full fine-tune only");
            }
            if self.keep_candidates {
                return refuse("kept candidates are checkpoints of a full fine-tune");
            }
            return Ok(());
        }
        if !self.no_release {
            return refuse(
                "a full fine-tune yields a checkpoint, and the release gate takes adapters: \
                 stop before it with no_release",
            );
        }
        if self.rank.is_some() {
            return refuse("a rank is the size of an adapter; a full fine-tune has none");
        }
        if let Some(spec) = &self.train_precision {
            splinter_model::consolidate::check_train_precision(spec).map_err(|e| {
                OrchestratorError::Refused(format!("training precision '{spec}': {e}"))
            })?;
        }
        Ok(())
    }

    fn targets(&self) -> Targets {
        Targets {
            mastery: self.mastery,
            min_probes: self.min_probes,
            max_rounds: self.max_rounds_per_skill,
        }
    }
}

/// One skill's closing assessment.
#[derive(Clone, Debug, Serialize)]
pub struct SkillAssessment {
    /// The skill.
    pub skill: String,
    /// The policy the run began with, on the final probes.
    pub baseline: ProbeScore,
    /// The policy the run ends with.
    pub final_score: ProbeScore,
    /// The final pass rate's 95 percent Wilson interval.
    pub interval: Option<(f64, f64)>,
    /// Whether the final pass rate reaches the target on enough probes.
    pub met: bool,
}

/// The closing assessment on probes that decided nothing.
#[derive(Clone, Debug, Serialize)]
pub struct Assessment {
    /// Each assessed skill.
    pub skills: Vec<SkillAssessment>,
    /// Skills that met the target.
    pub met: usize,
    /// Skills assessed.
    pub assessed: usize,
    /// Skills whose final pass rate is above the baseline's.
    pub improved: usize,
    /// Whether every assessed skill met the target.
    pub accepted: bool,
}

/// What an acquisition reports.
#[derive(Clone, Debug, Serialize)]
pub struct AcquireReport {
    /// The objective.
    pub objective: String,
    /// The policy the run began with.
    pub policy: PolicyUsed,
    /// The sources taken in.
    pub sources: Vec<Intaken>,
    /// The skill graph and how it was made.
    pub planned: Planned,
    /// Each skill's probes; a skill without enough of both halves is not
    /// assessed.
    pub probes: Vec<SkillProbes>,
    /// The competence graph before any training, and its address.
    pub baseline: (CompetenceGraph, Digest),
    /// The rounds, in order.
    pub rounds: Vec<Round>,
    /// The competence graph at the end, and its address.
    pub competence: (CompetenceGraph, Digest),
    /// The closing assessment; `None` when no candidate was ever accepted
    /// and the baseline stands.
    pub assessment: Option<Assessment>,
    /// The release gate on the final candidate, when it was asked.
    pub release: Option<Released>,
    /// Why the loop ended.
    pub ended: String,
    /// Whether the objective was reached on the final probes.
    pub accepted: bool,
}

/// The run so far.
pub(super) struct Acquisition<'a> {
    pub(super) ctx: &'a Context,
    pub(super) request: &'a AcquireRequest,
    pub(super) trainer: &'a dyn Trainer,
    pub(super) deadline: Option<Instant>,
    pub(super) kinds: Vec<String>,
    pub(super) generator: ModelRef,
    pub(super) teacher: ModelRef,
    pub(super) graph: SkillGraph,
    pub(super) probes: BTreeMap<String, SkillProbes>,
    pub(super) probe_ids: BTreeSet<Digest>,
    pub(super) seen: Seen,
    pub(super) competence: CompetenceGraph,
    pub(super) competence_address: Digest,
    /// Who attempts the round's tasks and is measured next: the policy the
    /// run began with, then the last accepted candidate.
    pub(super) incumbent: ModelRef,
    /// The datasets of the rounds accepted so far.
    pub(super) datasets: Vec<String>,
    pub(super) last_candidate: Option<Candidate>,
    /// The checkpoint the incumbent is, when every weight is trained.
    pub(super) checkpoint: Option<FullCandidate>,
}

impl Acquisition<'_> {
    /// The model that writes a round's training tasks: the configured
    /// generator, or with `self_generate` the incumbent (the policy the run
    /// began with until a candidate is accepted). Probes are always the
    /// configured generator's.
    pub(super) fn author(&self) -> &ModelRef {
        if self.request.self_generate {
            &self.incumbent
        } else {
            &self.generator
        }
    }

    /// Replaces `update`'s skill in the competence graph and keeps the new
    /// revision.
    pub(super) fn update_competence(
        &mut self,
        update: SkillCompetence,
        round: usize,
    ) -> Result<(), OrchestratorError> {
        let next = self.competence.updated(
            self.competence_address.clone(),
            &self.incumbent.to_string(),
            round,
            update,
        );
        self.competence_address = competence::keep(self.ctx, &next)?;
        self.competence = next;
        Ok(())
    }
}

/// Runs `request`, training with `trainer`; see the module documentation.
pub fn acquire(
    ctx: &Context,
    request: &AcquireRequest,
    trainer: &dyn Trainer,
) -> Result<Recorded<AcquireReport>, OrchestratorError> {
    request.validate()?;
    let kinds = if request.kinds.is_empty() {
        DEFAULT_LEARN_KINDS
            .iter()
            .map(|k| (*k).to_string())
            .collect()
    } else {
        check_kinds(&request.kinds)?
    };
    let assignments = roles::assignments(ctx.config(), &request.roles)?;
    let (generator, teacher, planner) = (
        assignments.get(Role::Generator).clone(),
        assignments.get(Role::Teacher).clone(),
        assignments.get(Role::Planner).clone(),
    );
    let deadline = request.budget.map(|b| Instant::now() + b);
    record(ctx, "acquire", request, |run| {
        let policy = PolicyUsed {
            alias: POLICY_DEFAULT.into(),
            release: ctx.policy_pin(POLICY_DEFAULT)?.map(|pin| pin.release),
        };
        run.stage("policy", &policy)?;
        let cancel = run.cancel_token();
        let sources = intake(ctx, &request.sources)?;
        run.stage("sources", &sources)?;
        let ids: Vec<_> = sources.iter().map(|s| s.source.clone()).collect();
        let planned = plan(
            ctx,
            &request.objective,
            &ids,
            (!request.structural_graph).then_some(&planner),
            deadline,
            &cancel,
        )?;
        run.stage("graph", &planned)?;
        let mut state = Acquisition {
            ctx,
            request,
            trainer,
            deadline,
            kinds,
            generator,
            teacher,
            graph: planned.graph.clone(),
            probes: BTreeMap::new(),
            probe_ids: BTreeSet::new(),
            seen: probes::fresh_seen(),
            competence: CompetenceGraph {
                graph: planned.address.clone(),
                parent: None,
                revision: 0,
                model: String::new(),
                round: 0,
                skills: Vec::new(),
            },
            competence_address: planned.address.clone(),
            incumbent: ModelRef::policy_default(),
            datasets: Vec::new(),
            last_candidate: None,
            checkpoint: None,
        };
        let report = state.run(run, policy, sources, planned)?;
        Ok(report)
    })
}

impl Acquisition<'_> {
    fn run(
        &mut self,
        run: &mut Recorder<'_>,
        policy: PolicyUsed,
        sources: Vec<Intaken>,
        planned: Planned,
    ) -> Result<AcquireReport, OrchestratorError> {
        let ctx = self.ctx;
        let cancel = run.cancel_token();

        // Probes, then the baseline measured on the dev half of each.
        let mut probe_list = Vec::new();
        for skill in self.graph.order().into_iter().cloned().collect::<Vec<_>>() {
            let probes = probes::probes_for(
                ctx,
                &skill,
                &probes::Ask {
                    kinds: &self.kinds,
                    generator: &self.generator,
                    goal: format!(
                        "{} (assessment). Skill: {}. It must be shown to: {}",
                        self.request.objective,
                        skill.title,
                        skill
                            .requirements
                            .iter()
                            .map(|r| r.statement.as_str())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ),
                    deadline: self.deadline,
                    cancel: cancel.clone(),
                },
                self.request.probes_per_skill,
                &mut self.seen,
            )?;
            self.probe_ids
                .extend(probes.dev.iter().chain(&probes.finals).cloned());
            probe_list.push(probes);
        }
        let assessable: Vec<SkillProbes> = probe_list
            .iter()
            .filter(|p| p.assessable(self.request.min_probes))
            .cloned()
            .collect();
        run.stage("probes", &probe_list)?;
        if assessable.is_empty() {
            let mut refused: BTreeMap<&str, usize> = BTreeMap::new();
            for (why, n) in probe_list.iter().flat_map(|p| &p.refused) {
                *refused.entry(why.as_str()).or_default() += n;
            }
            let mut reasons: Vec<(&str, usize)> = refused.into_iter().collect();
            reasons.sort_by_key(|r| std::cmp::Reverse(r.1));
            let reasons: Vec<String> = reasons
                .iter()
                .take(5)
                .map(|(why, n)| format!("{why} {n}"))
                .collect();
            return Err(OrchestratorError::Refused(format!(
                "no skill has enough probes to be assessed: the generator wrote too few tasks that \
                 exercise a skill and repeat no other (refused: {})",
                if reasons.is_empty() { "none proposed".into() } else { reasons.join(", ") }
            )));
        }
        self.probes = assessable
            .iter()
            .map(|p| (p.skill.clone(), p.clone()))
            .collect();

        let mut skills = Vec::new();
        for probes in &assessable {
            let (scored, _) = probes::score(
                ctx,
                &self.incumbent,
                &probes.dev,
                &format!("baseline: dev probes of {}", probes.skill),
                &cancel,
            )?;
            if let Some(c) =
                SkillCompetence::measured(&probes.skill, scored.passes, scored.graded, 0)
            {
                skills.push(c);
            }
        }
        self.competence = CompetenceGraph {
            graph: self.competence.graph.clone(),
            parent: None,
            revision: 1,
            model: self.incumbent.to_string(),
            round: 0,
            skills,
        };
        self.competence_address = competence::keep(ctx, &self.competence)?;
        let baseline = (self.competence.clone(), self.competence_address.clone());
        run.stage("baseline", &baseline)?;

        let targets = self.request.targets();
        let mut rounds: Vec<Round> = Vec::new();
        let ended = loop {
            if rounds.len() >= self.request.max_rounds {
                break format!("the round limit ({}) was reached", self.request.max_rounds);
            }
            if self.deadline.is_some_and(|d| Instant::now() >= d) {
                break "the budget was spent".to_string();
            }
            run.check_cancelled()?;
            let Some(chosen) = next_skill(&self.graph, &self.competence, &targets) else {
                break "every skill is mastered or has had its rounds".to_string();
            };
            let Some(skill) = self.graph.get(&chosen.skill).cloned() else {
                unreachable!("a skill is selected from the graph");
            };
            rounds.push(self.round(run, rounds.len() + 1, &skill, chosen.benefit)?);
        };

        let assessment = if self.last_candidate.is_some() || !rounds.is_empty() {
            Some(self.assess(&assessable, &cancel)?)
        } else {
            None
        };
        if let Some(assessment) = &assessment {
            run.stage("assessment", assessment)?;
        }
        let accepted = assessment.as_ref().is_some_and(|a| a.accepted);
        let release = match (&self.last_candidate, accepted, self.request.no_release) {
            (Some(candidate), true, false) => {
                let released = release_recorded(
                    ctx,
                    run,
                    &ReleaseRequest {
                        gate: self.request.gate.applied_to(GateConfig::default()),
                        ..ReleaseRequest::new(candidate.candidate.clone())
                    },
                );
                match released {
                    Ok(released) => {
                        run.stage("release", &released)?;
                        Some(released)
                    }
                    Err(e) if e.is_refusal() => {
                        run.stage("release", &serde_json::json!({ "refused": e.to_string() }))?;
                        None
                    }
                    Err(e) => return Err(e),
                }
            }
            _ => None,
        };
        Ok(AcquireReport {
            objective: self.request.objective.clone(),
            policy,
            sources,
            planned,
            probes: probe_list,
            baseline,
            rounds,
            competence: (self.competence.clone(), self.competence_address.clone()),
            assessment,
            release,
            ended,
            accepted,
        })
    }

    /// The closing assessment: the final probes of every assessable skill,
    /// attempted by the policy the run began with and by the one it ends
    /// with.
    fn assess(
        &self,
        assessable: &[SkillProbes],
        cancel: &splinter_agent::CancelToken,
    ) -> Result<Assessment, OrchestratorError> {
        let start = ModelRef::policy_default();
        let end = self.incumbent.clone();
        let mut skills = Vec::new();
        for probes in assessable {
            let (baseline, _) = probes::score(
                self.ctx,
                &start,
                &probes.finals,
                &format!("final: baseline on the probes of {}", probes.skill),
                cancel,
            )?;
            let (final_score, _) = probes::score(
                self.ctx,
                &end,
                &probes.finals,
                &format!("final: probes of {}", probes.skill),
                cancel,
            )?;
            let rate = |s: ProbeScore| (s.graded > 0).then(|| s.passes as f64 / s.graded as f64);
            skills.push(SkillAssessment {
                skill: probes.skill.clone(),
                baseline,
                final_score,
                interval: wilson(final_score.passes, final_score.graded),
                met: final_score.graded >= self.request.min_probes
                    && rate(final_score).is_some_and(|r| r >= self.request.mastery),
            });
        }
        let improved = skills
            .iter()
            .filter(|s| {
                let rate =
                    |s: ProbeScore| (s.graded > 0).then(|| s.passes as f64 / s.graded as f64);
                matches!((rate(s.final_score), rate(s.baseline)), (Some(f), Some(b)) if f > b)
            })
            .count();
        let met = skills.iter().filter(|s| s.met).count();
        Ok(Assessment {
            assessed: skills.len(),
            accepted: !skills.is_empty() && met == skills.len(),
            met,
            improved,
            skills,
        })
    }
}
