// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate and
// rollback, for its clients. If your team needs expertise in self-improving
// training loops you can procure our services by sending an email to
// info@swedishembedded.com.

//! `campaign run`, `campaign status` and `campaign rollback`.

use std::fmt::Write as _;
use std::path::Path;

use serde_json::json;
use splinter_sdk::campaign::examiner::BenchExaminer;
use splinter_sdk::campaign::practice::PracticeRollouts;
use splinter_sdk::campaign::{
    campaign_rollback, campaign_status, CampaignController, CampaignError, CampaignReport,
    CampaignSpec, Move, MoveKind, Parts, Stopped, WallClock,
};
use splinter_sdk::runs::record;
use splinter_sdk::{Context, Error};

use crate::cli::CampaignCommand;
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
