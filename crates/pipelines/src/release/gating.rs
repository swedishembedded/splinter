// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The release gate run on a candidate: the suites it is graded on, both
//! arms graded on each, and the four checks decided over the outcomes.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_eval::gate::{self, Check, GateConfig, GateReport, SuitePrompts, SuiteSummary};
use splinter_eval::paired::PairedOutcome;
use splinter_model::stats::BrainSignificance;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::releases::StoredRelease;

use super::probe::{self, clustered, pair, Probe, Suite};
use super::statistics::{drop_interval, items_to_see_two_points, task_level, with_task_level};
use super::{anchor, arm, leakage, serve};
use crate::train::Candidate;

/// One model's answer and outcome on each task of a suite.
pub(crate) type Outcomes = Vec<Probe>;

/// A suite's outcomes for one model, or why there are none.
pub(crate) type Graded = Result<Outcomes, String>;

/// `result`, with every failure but a cancel turned into its reason: a
/// check that cannot be measured fails the gate, it does not abort it.
pub(super) fn soft<T>(
    result: Result<T, OrchestratorError>,
) -> Result<Result<T, String>, OrchestratorError> {
    match result {
        Ok(value) => Ok(Ok(value)),
        Err(OrchestratorError::Cancelled) => Err(OrchestratorError::Cancelled),
        Err(e) => Ok(Err(e.to_string())),
    }
}

/// Which system prompt an arm is asked a suite under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Asked {
    /// The one the arm was trained under, as it is deployed.
    Deployed,
    /// The default one, whichever arm: for a suite that measures the weights
    /// and not the persona.
    Default,
}

/// How a report names the prompt `system` is: `default` or the start of the
/// persona prompt.
fn prompt_name(system: Option<&str>) -> String {
    const SHOWN: usize = 60;
    match system {
        None => "default".into(),
        Some(system) => {
            let start: String = system.chars().take(SHOWN).collect();
            let cut = if system.chars().count() > SHOWN {
                "..."
            } else {
                ""
            };
            format!("persona: {start}{cut}")
        }
    }
}

/// The model `reference` names graded on each of `suites`, decoding
/// greedily ([`probe::greedy`]) where its sampling can be set here - a
/// verdict is then the weights', not a draw's, and the served candidate is
/// asked the same way. A suite is asked under the arm's `deployed` prompt
/// or the default one, as its [`Asked`] says. The arms differ only by
/// adapter, so the second one graded runs on the base the first one loaded;
/// the serve check releases it before the served candidate starts.
pub(crate) fn grade_arm(
    ctx: &Context,
    reference: &ModelRef,
    deployed: Option<&str>,
    suites: &[(&Suite, Asked)],
    cancel: &CancelToken,
) -> Result<Vec<Graded>, OrchestratorError> {
    let model = match soft(probe::greedy(ctx, reference))? {
        Ok(model) => model,
        Err(why) => return Ok(suites.iter().map(|_| Err(why.clone())).collect()),
    };
    let mut graded = Vec::with_capacity(suites.len());
    for (suite, asked) in suites {
        let asked_model = match (asked, deployed) {
            (Asked::Deployed, Some(system)) => model.clone().with_system(system),
            _ => model.clone(),
        };
        graded.push(soft(probe::grade(ctx, &asked_model, suite, cancel))?);
    }
    Ok(graded)
}

/// The suites the gate grades on, each built or with why it could not be.
struct Suites {
    /// The improvement suite: the held-out records and the variants of
    /// trained tasks together, what the decision rests on.
    held_out: Result<Suite, String>,
    /// How the improvement suite divides, when it was built.
    parts: Option<Parts>,
    /// The variants among the held-out suite, when any were written.
    variants: Option<SuiteSummary>,
    retention: Result<Vec<(ReleaseId, Suite)>, String>,
    anchor: Result<Option<anchor::FrozenAnchor>, String>,
    anchor_suite: Option<Suite>,
}

/// The two kinds of task the improvement suite holds, told apart so each is
/// reported on its own: generalisation to held-out records and recall of
/// trained facts asked in other words.
struct Parts {
    generalisation: SuiteSummary,
    recall: Option<SuiteSummary>,
    /// The addresses of the recall tasks: those of the variants suite that no
    /// held-out record already is.
    recall_items: BTreeSet<String>,
}

impl Suites {
    fn build(
        ctx: &Context,
        candidate: &Candidate,
        champion: Option<&StoredRelease>,
    ) -> Result<Self, OrchestratorError> {
        let trained = match soft(leakage::trained_prompts(ctx, candidate))? {
            Ok(trained) => trained,
            Err(why) => {
                let why = format!("the candidate's training records: {why}");
                return Ok(Self {
                    held_out: Err(why.clone()),
                    parts: None,
                    variants: None,
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
        let improvement = soft((|| {
            let held_out = clean(probe::held_out(ctx, "held-out", &candidate.datasets)?);
            let variants = clean(probe::trained_variants(
                ctx,
                "variants",
                &candidate.datasets,
            )?);
            Ok::<_, OrchestratorError>((held_out, variants))
        })())?;
        let (held_out, parts, variants) = match improvement {
            Ok((mut held_out, variants)) => {
                let summary = (!variants.tasks.is_empty() || !variants.excluded.is_empty())
                    .then(|| variants.summary());
                let held_out_items: BTreeSet<String> = held_out
                    .tasks
                    .iter()
                    .map(|t| t.task.id.to_string())
                    .collect();
                let parts = Parts {
                    generalisation: held_out.summary(),
                    recall: summary.clone(),
                    recall_items: variants
                        .tasks
                        .iter()
                        .map(|t| t.task.id.to_string())
                        .filter(|id| !held_out_items.contains(id))
                        .collect(),
                };
                held_out.absorb(variants);
                (Ok(held_out), Some(parts), summary)
            }
            Err(why) => (Err(why), None, None),
        };
        let retention = soft(champion.map_or(Ok(Vec::new()), |champion| {
            ctx.releases()
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
            parts,
            variants,
            retention,
            anchor,
            anchor_suite,
        })
    }

    /// Every suite built with the prompt it is asked under, in a fixed order:
    /// held-out, retention, anchor. The anchor measures what the weights
    /// kept of general capability, so both arms are asked it under the
    /// default prompt; the others are asked as deployed.
    fn all(&self) -> Vec<(&Suite, Asked)> {
        let mut all: Vec<(&Suite, Asked)> =
            self.held_out.iter().map(|s| (s, Asked::Deployed)).collect();
        if let Ok(retention) = &self.retention {
            all.extend(retention.iter().map(|(_, suite)| (suite, Asked::Deployed)));
        }
        all.extend(self.anchor_suite.iter().map(|s| (s, Asked::Default)));
        all
    }
}

/// The four checks on `candidate` against `champion`.
pub(super) fn run_gate(
    ctx: &Context,
    candidate: &Candidate,
    champion: Option<&StoredRelease>,
    base_source: &Path,
    config: &GateConfig,
    cancel: &CancelToken,
) -> Result<GateReport, OrchestratorError> {
    let suites = Suites::build(ctx, candidate, champion)?;
    let all = suites.all();
    let candidate_ref = arm(ctx.config(), Some(&candidate.adapter));
    let champion_adapter = champion.map(StoredRelease::adapter).transpose()?;
    let champion_ref = arm(ctx.config(), champion_adapter);
    let candidate_prompt = ctx.system_prompt_of(&candidate.datasets)?;
    let champion_prompt = match champion {
        Some(champion) => ctx.system_prompt_of(&champion.manifest.datasets)?,
        None => None,
    };
    let prompts = all
        .iter()
        .map(|(suite, asked)| {
            let name = |deployed: &Option<String>| match asked {
                Asked::Deployed => prompt_name(deployed.as_deref()),
                Asked::Default => prompt_name(None),
            };
            SuitePrompts {
                suite: suite.name.clone(),
                candidate: name(&candidate_prompt),
                champion: name(&champion_prompt),
            }
        })
        .collect();
    let mut theirs = grade_arm(
        ctx,
        &candidate_ref,
        candidate_prompt.as_deref(),
        &all,
        cancel,
    )?
    .into_iter();
    let mut ours =
        grade_arm(ctx, &champion_ref, champion_prompt.as_deref(), &all, cancel)?.into_iter();
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
                let outcomes = clustered(ctx, suite, pair(suite, &c, &b))?;
                let mut check = gate::improvement(
                    suite.summary(),
                    suites.variants.clone(),
                    &outcomes,
                    config.alpha,
                    &BrainSignificance,
                );
                if let Some(measured) = check.measured.as_mut() {
                    measured.task_level = task_level(&outcomes);
                }
                if let (Some(measured), Some(parts)) = (check.measured.as_mut(), &suites.parts) {
                    let (recall, generalisation): (Vec<PairedOutcome>, Vec<PairedOutcome>) =
                        outcomes
                            .into_iter()
                            .partition(|o| parts.recall_items.contains(&o.item));
                    measured.generalisation = gate::part_result(
                        parts.generalisation.clone(),
                        &generalisation,
                        &BrainSignificance,
                    )
                    .map(|part| with_task_level(part, &generalisation));
                    measured.recall = parts
                        .recall
                        .clone()
                        .and_then(|summary| gate::part_result(summary, &recall, &BrainSignificance))
                        .map(|part| with_task_level(part, &recall));
                }
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
            Ok((c, b)) => {
                let outcomes = pair(suite, &c, &b);
                let mut check = gate::anchor(
                    frozen.suite.version,
                    frozen.digest.clone(),
                    suite.summary(),
                    &outcomes,
                    config.anchor_bound,
                );
                if let Some(measured) = check.measured.as_mut() {
                    measured.drop_interval = drop_interval(&outcomes);
                    measured.items_to_see_two_points =
                        items_to_see_two_points(&measured.comparison);
                }
                check
            }
        },
    };
    let serve = match (&suites.held_out, &in_process) {
        (Ok(held_out), Some(probes)) => {
            let take = config.serve_sample.min(held_out.tasks.len());
            let sample = Suite::of_tasks(
                format!("{} (served sample)", held_out.name),
                held_out.tasks[..take].to_vec(),
            );
            serve::check(
                ctx,
                &serve::ServeCheck {
                    binary: ctx.config().brain_binary.as_deref(),
                    base: base_source,
                    adapter: Some(&candidate.adapter),
                    adapter_digest: &candidate.adapter_digest,
                    system: candidate_prompt.as_deref(),
                    sample: &sample,
                    in_process: &probes[..take],
                    startup: Duration::from_secs(config.serve_startup_secs),
                },
                cancel,
            )?
        }
        _ => Check::unmeasured("the held-out tasks were not graded in-process to compare with"),
    };
    Ok(GateReport::new(*config, improvement, retention, anchor, serve).with_prompts(prompts))
}
