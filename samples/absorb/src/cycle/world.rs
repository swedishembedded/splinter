// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it without
// forgetting what it knew, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The cycle on the real models, through the SDK: sessions recorded by a
//! simulated user against the release in effect, the nightly `absorb`, and
//! the evaluation's two phases, each loading its model once.
//!
//! One Splinter handle serves the whole cycle, so a base loaded for one phase
//! is reused by the next and the card is handed over only when a phase needs
//! another.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Context as _;
use splinter_sdk::absorb::{absorb, Absorbed};
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::vocabulary::prompt::persona_prompt;
use splinter_sdk::Splinter;

use super::driver::{AbsorbedNight, RecordedNight, World};
use super::eval::{Answered, Judged, Models};
use super::grade::VerdictRequest;
use super::plan::NightPlan;
use super::prereg::AbsorbSettings;
use crate::facts::Manifest;
use crate::grading::{fact_task_set, Judge};
use crate::policy::{Decoding, Policy};
use crate::record;
use crate::runtime;
use crate::seal::SEALED;

/// The alias of the release in effect.
const ALIAS: &str = "default";

/// See the module documentation.
pub struct SdkWorld {
    splinter: Splinter,
    manifest_dir: PathBuf,
    manifest: Manifest,
    settings: AbsorbSettings,
    models: Option<PathBuf>,
    /// The identity of the policy that gave the last answers: the judge must
    /// not be the model it grades.
    solver: Mutex<String>,
}

impl SdkWorld {
    /// Opens the cycle's state under `state`.
    ///
    /// # Errors
    /// The manifest cannot be read or the runtime cannot start.
    pub fn open(
        state: &Path,
        manifest_dir: &Path,
        models: Option<PathBuf>,
        settings: AbsorbSettings,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            splinter: runtime::open_cycle(state, models.as_ref(), &settings.policy)?,
            manifest: Manifest::read(manifest_dir)?,
            manifest_dir: manifest_dir.to_path_buf(),
            settings,
            models,
            solver: Mutex::new("brain/Qwen/Qwen3-8B".into()),
        })
    }

    fn policy_ref() -> ModelRef {
        ModelRef::Policy(ALIAS.to_string())
    }
}

impl Models for SdkWorld {
    fn answer_all(&self, questions: &[String]) -> anyhow::Result<Answered> {
        let started = Instant::now();
        let ctx = self.splinter.context();
        let policy = Policy::load(&ctx, &Self::policy_ref(), &self.manifest.persona)?;
        let mut texts = Vec::with_capacity(questions.len());
        for (n, question) in questions.iter().enumerate() {
            let answer = policy.answer(&ctx, question, Decoding::Greedy)?;
            if n % 50 == 0 {
                eprintln!("evaluation answer {}/{}", n + 1, questions.len());
            }
            texts.push(answer.text);
        }
        let identity = policy.identity().to_string();
        drop(policy);
        *self.solver.lock().unwrap_or_else(|e| e.into_inner()) = identity.clone();
        Ok(Answered {
            texts,
            policy: identity,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    fn judge_all(&self, requests: &[VerdictRequest]) -> anyhow::Result<Judged> {
        let started = Instant::now();
        let ctx = self.splinter.context();
        let calibrated = Judge::calibrate(
            &ctx,
            &runtime::model_ref(&self.settings.judge)?,
            &fact_task_set(&ctx, &self.manifest.facts)?,
        )?;
        let solver = self
            .solver
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut verdicts = Vec::with_capacity(requests.len());
        for (n, r) in requests.iter().enumerate() {
            verdicts.push(calibrated.judge.verdict(
                &ctx,
                &solver,
                &r.question,
                &r.statement,
                &r.answer,
            )?);
            if n % 100 == 0 {
                eprintln!("evaluation verdict {}/{}", n + 1, requests.len());
            }
        }
        Ok(Judged {
            verdicts,
            calibration: calibrated.calibration,
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}

impl World for SdkWorld {
    fn record(&self, night: &NightPlan, dir: &Path) -> anyhow::Result<RecordedNight> {
        let outcome = record::record_with(
            &self.splinter,
            &record::Request {
                run: self.manifest_dir.clone(),
                out: dir.to_path_buf(),
                models: self.models.clone(),
                policy: format!("policy:{ALIAS}"),
                simulator: self.settings.simulator.clone(),
                items: night.sessions.iter().map(|s| s.as_item()).collect(),
            },
        )?;
        anyhow::ensure!(outcome.failed.is_empty(), "{}", outcome.failed.join("; "));
        Ok(RecordedNight {
            sessions: outcome.recorded.len(),
            seconds: outcome.seconds,
        })
    }

    fn absorb(&self, _night: &NightPlan, dir: &Path) -> anyhow::Result<AbsorbedNight> {
        let started = Instant::now();
        let mut request = self.settings.request(
            &dir.join("sessions"),
            &dir.join(crate::subjects::FILE),
            &self.manifest_dir.join(SEALED),
        )?;
        request.system_prompt = Some(persona_prompt(&self.manifest.persona));
        let ctx = self.splinter.context();
        let done = absorb(&ctx, &request, self.splinter.trainer()).context("splinter absorb")?;
        let report: &Absorbed = &done.report;
        let refused = report.intake.as_ref().map_or(0, |i| i.refused.len());
        let failed = report.extract.as_ref().map_or(0, |e| e.failed.len());
        anyhow::ensure!(
            refused == 0 && failed == 0,
            "the update could not read every session: {refused} refused at intake, {failed} \
             without claims after extraction"
        );
        Ok(AbsorbedNight {
            released: report.release.as_ref().map(ToString::to_string),
            live_claims: Some(report.live as u64),
            stopped: report.stopped.clone(),
            seconds: started.elapsed().as_secs_f64(),
            report: serde_json::to_value(&done)?,
        })
    }

    fn release_in_effect(&self) -> anyhow::Result<Option<String>> {
        Ok(self
            .splinter
            .context()
            .policy_pin(ALIAS)?
            .map(|pin| pin.release.to_string()))
    }
}
