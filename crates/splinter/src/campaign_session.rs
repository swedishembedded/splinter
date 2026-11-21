// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate and
// rollback, for its clients. If your team needs expertise in self-improving
// training loops you can procure our services by sending an email to
// info@swedishembedded.com.

//! `campaign run`, `campaign status`, `campaign rollback`, `campaign
//! problems` and `campaign next`.

use std::fmt::Write as _;
use std::path::Path;

use serde::Serialize;
use serde_json::json;
use splinter_sdk::campaign::examiner::BenchExaminer;
use splinter_sdk::campaign::practice::PracticeRollouts;
use splinter_sdk::campaign::problems::{
    arm_text, campaign_next, check_arm, engineering_paths, parse_document, problems_add,
    problems_revise, problems_show, problems_update, ArmCheck, NextReport, ProblemsReport,
    VerdictRequest, VerdictSource,
};
use splinter_sdk::campaign::{
    campaign_rollback, campaign_status, CampaignController, CampaignError, CampaignReport,
    CampaignSpec, Move, MoveKind, Parts, Stopped, WallClock,
};
use splinter_sdk::measure::campaign::problems::{
    priority, CandidateStatus, Event, Evidence, Phase, ProblemStatus, Verdict,
};
use splinter_sdk::runs::record;
use splinter_sdk::{Context, Error};

use crate::cli::{CampaignCommand, ProblemsCommand, StatusArg, VerdictArg};
use crate::output::{emit, Report};
use crate::session::Exit;

impl Report for CampaignReport {
    fn human(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "campaign {} (seed {})", self.campaign, self.seed);
        for generation in &self.generations {
            let decided = match generation.moved.kind {
                MoveKind::Promote => "promoted".to_string(),
                MoveKind::Keep | MoveKind::Rollback => {
                    format!("kept: {}", generation.moved.reason)
                }
            };
            let _ = writeln!(
                out,
                "  generation {}: {} {decided}",
                generation.generation,
                generation
                    .candidate
                    .as_ref()
                    .and_then(|c| c.candidate.as_deref())
                    .unwrap_or("no candidate"),
            );
        }
        for moved in self.history.iter().filter(|m| m.kind == MoveKind::Rollback) {
            let _ = writeln!(
                out,
                "  rolled back to generation {}: {}",
                moved.to.generation, moved.reason
            );
        }
        let _ = writeln!(
            out,
            "champion: generation {} ({})",
            self.champion.generation, self.champion.model
        );
        let _ = writeln!(
            out,
            "spent: {} rollouts, {} tokens, {:.3} GPU-hours",
            self.spent.rollouts, self.spent.tokens, self.spent.gpu_hours
        );
        if let Some(Stopped::Budget {
            limit,
            spent,
            allowed,
        }) = &self.stopped
        {
            let _ = writeln!(out, "stopped by the {limit} budget: {spent} of {allowed}");
        }
        out
    }
}

impl Report for Move {
    fn human(&self) -> String {
        format!(
            "champion moved from generation {} to generation {}: {}\n",
            self.from.generation, self.to.generation, self.reason
        )
    }
}

impl Report for ProblemsReport {
    fn human(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "problem state of campaign {}: {} ({} events{})",
            self.campaign,
            self.digest,
            self.events,
            if self.recorded {
                ", one recorded now"
            } else {
                ""
            }
        );
        let p = &self.state.parameters;
        let targets: Vec<String> = p.targets.iter().map(|(m, d)| format!("{m} {d}")).collect();
        let _ = writeln!(
            out,
            "parameters: belief +{} / -{}, alpha {}, targets {}; engineering paths {}",
            p.belief_up,
            p.belief_down,
            p.alpha,
            targets.join(", "),
            p.engineering.join(", ")
        );
        for problem in &self.state.problems {
            let status = match problem.status {
                ProblemStatus::Open => "open",
                ProblemStatus::Resolved => "resolved",
            };
            let _ = writeln!(
                out,
                "\nproblem {} ({status}, {} attempts, {}): {}",
                problem.id, problem.attempts, problem.metric, problem.statement
            );
            for evidence in &problem.evidence {
                let _ = writeln!(out, "  evidence {} {}", evidence.reference, evidence.note);
            }
            for c in &problem.candidates {
                let status = match c.status {
                    CandidateStatus::Pending => "pending",
                    CandidateStatus::Adopted => "adopted",
                };
                let phase = match c.phase {
                    Phase::Hypothesis => "hypothesis",
                    Phase::Engineering => "engineering",
                };
                let _ = writeln!(
                    out,
                    "  candidate {} [{status}, {phase}] belief {:.3} priority {:.3}: changes {}",
                    c.id,
                    c.belief,
                    priority(c.belief, problem.attempts),
                    c.variable
                );
                for o in &c.history {
                    let _ = writeln!(
                        out,
                        "    {}: {:?}, belief {:.3} -> {:.3}: {}",
                        o.experiment, o.verdict, o.belief_before, o.belief_after, o.reason
                    );
                }
                for r in &c.revisions {
                    let _ = writeln!(out, "    revised {} -> {}: {}", r.before, r.after, r.reason);
                }
            }
        }
        match self.ranked.first() {
            Some(next) => {
                let _ = writeln!(
                    out,
                    "\nnext: {} of {} (priority {:.3})",
                    next.candidate, next.problem, next.priority
                );
            }
            None => out.push_str("\nnext: no pending hypothesis with a belief above 0\n"),
        }
        out
    }
}

/// `campaign next`'s report, with where the arm was written.
#[derive(Serialize)]
struct NextOutput {
    #[serde(flatten)]
    report: NextReport,
    written: Option<std::path::PathBuf>,
}

impl Report for NextOutput {
    fn human(&self) -> String {
        let Some(p) = &self.report.proposal else {
            return format!(
                "campaign {}: no pending hypothesis with a belief above 0\n",
                self.report.campaign
            );
        };
        let mut out = format!(
            "next hypothesis: {} of problem {}: priority {:.3} (belief {:.3}, {} attempts on the problem); changes {}\n",
            p.candidate, p.problem, p.priority, p.belief, p.attempts, p.variable
        );
        match (&self.report.arm, &self.written) {
            (Some(arm), Some(path)) => {
                let _ = writeln!(
                    out,
                    "arm {} written to {} (changes {})",
                    arm.name,
                    path.display(),
                    self.report.changed.join(", ")
                );
            }
            (Some(arm), None) => {
                let _ = writeln!(
                    out,
                    "arm {} changes {}; --write ARM writes its campaign file",
                    arm.name,
                    self.report.changed.join(", ")
                );
            }
            (None, _) => out.push_str(
                "the candidate declares no value the campaign file can hold: build its arm by hand and \
                 check it with `campaign problems check`\n",
            ),
        }
        out
    }
}

impl Report for ArmCheck {
    fn human(&self) -> String {
        let held = match (&self.variable, self.engineering) {
            (Some(v), true) => format!("{v}, inside the engineering paths"),
            (Some(v), false) => v.clone(),
            (None, _) => "the engineering paths".to_string(),
        };
        format!("the arm changes only {held}: {}\n", self.changed.join(", "))
    }
}

fn evidence_links(references: Vec<String>) -> Vec<Evidence> {
    references
        .into_iter()
        .map(|reference| Evidence {
            reference,
            note: String::new(),
        })
        .collect()
}

fn problems(ctx: &Context, json: bool, command: ProblemsCommand) -> Result<Exit, Error> {
    match command {
        ProblemsCommand::Show { spec } => {
            let campaign = read(&spec)?;
            emit(json, &problems_show(ctx, &campaign).map_err(refused)?);
        }
        ProblemsCommand::Add { spec, document } => {
            let campaign = read(&spec)?;
            let text = std::fs::read_to_string(&document)
                .map_err(|e| Error::Refused(format!("{}: {e}", document.display())))?;
            let document = parse_document(&text, &document).map_err(refused)?;
            emit(
                json,
                &problems_add(ctx, &campaign, document).map_err(refused)?,
            );
        }
        ProblemsCommand::Update {
            spec,
            candidate,
            experiment,
            verdict,
            compare,
            arm,
            gate,
            generation,
            evidence,
            reason,
        } => {
            let campaign = read(&spec)?;
            let source = match (verdict, compare, arm, gate, generation) {
                (Some(verdict), None, None, None, None) => VerdictSource::Stated(match verdict {
                    VerdictArg::Supported => Verdict::Supported,
                    VerdictArg::Falsified => Verdict::Falsified,
                    VerdictArg::Inconclusive => Verdict::Inconclusive,
                    VerdictArg::Invalid => Verdict::Invalid,
                }),
                (None, Some(report), Some(arm), None, None) => {
                    let text = std::fs::read_to_string(&report)
                        .map_err(|e| Error::Refused(format!("{}: {e}", report.display())))?;
                    let report = serde_json::from_str(&text).map_err(|e| {
                        Error::Refused(format!(
                            "{}: not an experiences compare-arms report: {e}",
                            report.display()
                        ))
                    })?;
                    VerdictSource::Compare { report, arm }
                }
                (None, None, None, Some(gate), Some(generation)) => VerdictSource::Gate {
                    arm: Box::new(read(&gate)?),
                    generation,
                },
                _ => return Err(Error::Refused(
                    "give one source: --verdict, --compare with --arm, or --gate with --generation"
                        .into(),
                )),
            };
            let request = VerdictRequest {
                candidate,
                experiment,
                source,
                evidence: evidence_links(evidence),
                reason,
            };
            emit(
                json,
                &problems_update(ctx, &campaign, request).map_err(refused)?,
            );
        }
        ProblemsCommand::Revise {
            spec,
            candidate,
            belief,
            problem,
            status,
            reason,
        } => {
            let campaign = read(&spec)?;
            let revision =
                match (candidate, belief, problem, status) {
                    (Some(candidate), Some(belief), None, None) => Event::Belief {
                        candidate,
                        belief,
                        reason,
                    },
                    (None, None, Some(problem), Some(status)) => Event::Status {
                        problem,
                        status: match status {
                            StatusArg::Open => ProblemStatus::Open,
                            StatusArg::Resolved => ProblemStatus::Resolved,
                        },
                        reason,
                    },
                    _ => return Err(Error::Refused(
                        "revise a belief (--candidate, --belief) or a status (--problem, --status)"
                            .into(),
                    )),
                };
            emit(
                json,
                &problems_revise(ctx, &campaign, revision).map_err(refused)?,
            );
        }
        ProblemsCommand::Check {
            spec,
            arm,
            candidate,
            variable,
            engineering,
        } => {
            let campaign = read(&spec)?;
            let arm = read(&arm)?;
            let (variable, engineering, allowed) = match candidate {
                Some(id) => {
                    let state = problems_show(ctx, &campaign).map_err(refused)?.state;
                    let (_, c) = state
                        .candidate(&id)
                        .ok_or_else(|| Error::Refused(format!("no candidate {id:?}")))?;
                    let engineering = engineering || c.phase == Phase::Engineering;
                    (
                        Some(c.variable.clone()),
                        engineering,
                        state.parameters.engineering.clone(),
                    )
                }
                None => (
                    variable,
                    engineering,
                    engineering_paths(ctx, &campaign).map_err(refused)?,
                ),
            };
            emit(
                json,
                &check_arm(&campaign, &arm, variable.as_deref(), engineering, &allowed)
                    .map_err(refused)?,
            );
        }
    }
    Ok(Exit::Ok)
}

fn read(path: &Path) -> Result<CampaignSpec, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Refused(format!("{}: {e}", path.display())))?;
    CampaignSpec::parse(&text).map_err(|e| Error::Refused(format!("{}: {e}", path.display())))
}

fn refused(e: CampaignError) -> Error {
    Error::from(e)
}

/// Runs `command`.
pub fn campaign(
    splinter: &splinter_sdk::Splinter,
    ctx: &Context,
    json: bool,
    command: CampaignCommand,
) -> Result<Exit, Error> {
    match command {
        CampaignCommand::Run { spec, gpus } => {
            let campaign = read(&spec)?;
            let arguments = json!({"spec": spec, "gpus": gpus});
            let ran = record(ctx, "campaign run", &arguments, |run| {
                let missing = |section: &str| {
                    CampaignError::Config(format!("the campaign file has no [{section}] section"))
                };
                let mut rollouts = PracticeRollouts::new(
                    campaign
                        .practice
                        .as_ref()
                        .ok_or_else(|| missing("practice"))?,
                )?;
                let mut examiner = BenchExaminer::new(
                    campaign.examine.clone().ok_or_else(|| missing("examine"))?,
                )?;
                let stopwatch = WallClock::start();
                let progress = |line: &str| eprintln!("{line}");
                let mut controller = CampaignController::open(
                    ctx,
                    campaign.clone(),
                    Parts {
                        rollouts: &mut rollouts,
                        examiner: &mut examiner,
                        trainer: splinter.trainer(),
                        stopwatch: &stopwatch,
                        gpus,
                        cancel: run.cancel_token(),
                        progress: Some(&progress),
                    },
                )?;
                run.seed("seed", Some(controller.seed().into()))?;
                Ok(controller.run()?)
            })?;
            emit(json, &ran.report);
            Ok(Exit::Ok)
        }
        CampaignCommand::Status { spec } => {
            let campaign = read(&spec)?;
            emit(json, &campaign_status(ctx, &campaign).map_err(refused)?);
            Ok(Exit::Ok)
        }
        CampaignCommand::Problems(command) => problems(ctx, json, command),
        CampaignCommand::Next {
            spec,
            engineering,
            write,
        } => {
            let campaign = read(&spec)?;
            let phase = if engineering {
                Phase::Engineering
            } else {
                Phase::Hypothesis
            };
            let report = campaign_next(ctx, &campaign, phase).map_err(refused)?;
            let written = match (&report.arm, write) {
                (Some(arm), Some(path)) => {
                    let text = arm_text(arm).map_err(refused)?;
                    std::fs::write(&path, text)
                        .map_err(|e| Error::Refused(format!("{}: {e}", path.display())))?;
                    Some(path)
                }
                (None, Some(_)) => {
                    return Err(Error::Refused(
                        "the proposed candidate declares no value: there is no arm to write".into(),
                    ))
                }
                (_, None) => None,
            };
            emit(json, &NextOutput { report, written });
            Ok(Exit::Ok)
        }
        CampaignCommand::Rollback { spec, to, reason } => {
            let campaign = read(&spec)?;
            emit(
                json,
                &campaign_rollback(ctx, &campaign, to, &reason).map_err(refused)?,
            );
            Ok(Exit::Ok)
        }
    }
}
