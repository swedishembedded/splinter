// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that promote a model only on
// a signed verdict from an examiner the learner cannot reach, for its
// clients. If your team needs expertise in sealed evaluation of learning
// agents, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The sealed examination as ledger units: each checkpoint sits bench's
//! exam once (`<model>/sit/<stage>`), and pairs of sittings are judged
//! (`<model>/<pair>/verdict`).
//!
//! A sitting's output is where bench keeps its jobs, never their content;
//! a verdict's output is bench's signed document of aggregates, and its
//! summary holds the verdict and the pass rate of each holdout tier. When
//! bench no longer holds a committed sitting's jobs, the sitting is run
//! again.

use std::fmt::Write as _;

use anyhow::Context as _;
use serde_json::json;
use splinter_sdk::bench::examine::Sitting;
use splinter_sdk::model::scratch::Checkpoint;
use splinter_sdk::store::stage_ledger::{Commit, StageWork, Standing};

use crate::recipe;
use crate::stages::{reset, write, Run};

impl Run<'_> {
    /// `checkpoint` sits the exam, as unit `<label>/sit/<stage>`; a
    /// `control` sitting is the spurious-reward control arm. `None` when no
    /// examiner is configured.
    ///
    /// # Errors
    /// The checkpoint cannot be served or the exam cannot be run.
    pub fn sit(
        &mut self,
        label: &str,
        stage: &str,
        checkpoint: (&Commit, &Checkpoint),
        control: bool,
    ) -> anyhow::Result<Option<Commit>> {
        let Some(examiner) = self.examiner else {
            return Ok(None);
        };
        let digest = checkpoint
            .0
            .output("checkpoint")
            .context("the stage has no checkpoint")?
            .clone();
        let unit = format!("{label}/sit/{stage}");
        let key = self.key(
            &json!({ "control": control }),
            &unit,
            &[("checkpoint", &digest)],
            recipe::EXAMINE,
            &["bench", "sven", "brain"],
        );
        if let Ok(Standing::Committed(commit)) = self.ledger().standing(&unit, &key) {
            if sitting_of(self, &commit).is_ok_and(|s| !s.present()) {
                self.ledger()
                    .invalidate(&unit, &key, "bench no longer holds its jobs")?;
            }
        }
        let rel = format!("exam/{label}/{stage}/sitting.json");
        // Named by its key, so a rerun of the same sitting replaces its own
        // interrupted jobs and no other.
        let name = key
            .as_str()
            .rsplit(':')
            .next()
            .unwrap_or("")
            .chars()
            .take(16)
            .collect::<String>();
        let ckpt = checkpoint.1.clone();
        let commit = self.unit(&unit, &key, |run, _| {
            let out = run.path(&rel);
            reset(out.parent().context("a sitting has a directory")?)?;
            let sitting = examiner.sit(&run.brain.serve(&ckpt)?, &name, control)?;
            write(&out, &serde_json::to_string(&sitting)?)?;
            Ok(
                StageWork::new(json!({ "model": ckpt.model, "control": control }))
                    .output("sitting", &rel),
            )
        })?;
        Ok(Some(commit))
    }

    /// The verdict on the sitting `candidate` against `base`, with the
    /// `control` arm when there is one, as unit `<label>/<pair>/verdict`;
    /// shown with its tiers.
    ///
    /// # Errors
    /// The verdict cannot be issued.
    pub fn judge(
        &mut self,
        label: &str,
        pair: &str,
        base: Option<&Commit>,
        candidate: Option<&Commit>,
        control: Option<&Commit>,
    ) -> anyhow::Result<Option<Commit>> {
        let (Some(examiner), Some(base), Some(candidate)) = (self.examiner, base, candidate) else {
            writeln!(
                self.out,
                "[ skip ] {label}/{pair}/verdict: no examiner configured; no promotion claim is made"
            )?;
            return Ok(None);
        };
        let digest = |c: &Commit| {
            c.output("sitting")
                .cloned()
                .context("a sitting has no output")
        };
        let (b, c) = (digest(base)?, digest(candidate)?);
        let k = control.map(digest).transpose()?;
        let mut inputs = vec![("base", &b), ("candidate", &c)];
        if let Some(k) = &k {
            inputs.push(("control", k));
        }
        let unit = format!("{label}/{pair}/verdict");
        let key = self.key(&json!({}), &unit, &inputs, recipe::EXAMINE, &["bench"]);
        let rel = format!("exam/{label}/{pair}/verdict.json");
        let sittings = (
            sitting_of(self, base)?,
            sitting_of(self, candidate)?,
            control.map(|c| sitting_of(self, c)).transpose()?,
        );
        let commit = self.unit(&unit, &key, |run, _| {
            let out = run.path(&rel);
            reset(out.parent().context("a verdict has a directory")?)?;
            let examined = examiner.verdict(&sittings.0, &sittings.1, sittings.2.as_ref(), &out)?;
            Ok(StageWork::new(json!({
                "verdict": examined.verdict, "verified": examined.verified, "promotes": examined.promotes(),
                "tiers": examined.tiers, "reasons": examined.reasons,
                "base": base.summary["model"], "candidate": candidate.summary["model"],
            }))
            .output("verdict", &rel))
        })?;
        let shown = show_verdict(&format!("{label}/{pair}"), &commit.summary);
        write!(self.out, "{shown}")?;
        Ok(Some(commit))
    }
}

fn sitting_of(run: &Run<'_>, commit: &Commit) -> anyhow::Result<Sitting> {
    let path = run.path(
        &commit
            .outputs
            .iter()
            .find(|o| o.name == "sitting")
            .context("a sitting has no output")?
            .path,
    );
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// A verdict as the terminal shows it: the verdict and each tier's pass
/// rates, base to candidate.
#[must_use]
pub fn show_verdict(name: &str, summary: &serde_json::Value) -> String {
    let mut s = format!(
        "  exam {name}: {} (signature {}){}\n",
        summary["verdict"].as_str().unwrap_or("?"),
        if summary["verified"].as_bool().unwrap_or(false) {
            "verified"
        } else {
            "NOT verified"
        },
        if summary["promotes"].as_bool().unwrap_or(false) {
            ", promoted"
        } else {
            ", not promoted"
        },
    );
    for (tier, rates) in summary["tiers"].as_object().into_iter().flatten() {
        let _ = writeln!(
            s,
            "    tier {tier:<6} {:>3} items  {:.2} -> {:.2}",
            rates["items"],
            rates["base"].as_f64().unwrap_or(0.0),
            rates["candidate"].as_f64().unwrap_or(0.0)
        );
    }
    s
}
