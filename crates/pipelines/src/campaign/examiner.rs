// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements promotion gates that move a champion only on
// signed, independent evidence, for its clients. If your team needs
// expertise in sealed evaluation of learning agents you can procure our
// services by sending an email to info@swedishembedded.com.

//! The campaign's examiner: bench's `bench examine`, run from its checkout
//! once per verdict. Bench serves each model itself on the build it pins,
//! keeps the item-level grades in its private directory, and writes one
//! signed document of aggregates per comparison (`bench.examination/1`):
//! the candidate against the champion over the development split, over the
//! retention suite, and over the generation's own sealed slice. Only those
//! documents cross back.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_eval::promotion::{VerdictKey, ALPHA};
use splinter_orchestrator::context::Context;

use super::types::CampaignError;
use super::work::{ExamRequest, Examined, Examiner, Model};

/// How often a running examination is asked whether it was cancelled.
const CANCEL_POLL: Duration = Duration::from_millis(250);

/// What a campaign file's `[examine]` names.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExamineSpec {
    /// The examiner's checkout (`./bench` is run from it).
    pub bench: PathBuf,
    /// The verdict signing key (mode 600), the examiner's and the gate's.
    pub signing_key: PathBuf,
    /// The examiner's private directory (item-level grades, burn records).
    pub private: PathBuf,
    /// The development split's collection, as `bench examine --collection`
    /// names it.
    pub dev: String,
    /// The retention suite's collection.
    pub retention: String,
    /// One sealed slice per generation, drawn from a private seed
    /// beforehand, spent in order.
    pub slices: Vec<String>,
    /// The base model as the candidate files name it (`Qwen/Qwen3-4B`).
    pub base: String,
    /// The commit of Splinter that trains the candidates.
    pub proposer_commit: String,
    /// The commit of brain that trains them.
    pub brain_commit: String,
    /// The serving flags of every `bench examine` (`--brain`,
    /// `--models-dir`, `--api-key-file`, `--device`, ...), passed as given.
    #[serde(default)]
    pub serving: Vec<String>,
}

/// Bench as the campaign's examiner.
pub struct BenchExaminer {
    spec: ExamineSpec,
    key: VerdictKey,
}

impl BenchExaminer {
    /// The examiner `spec` names, its key read.
    ///
    /// # Errors
    /// The key cannot be used.
    pub fn new(spec: ExamineSpec) -> Result<Self, CampaignError> {
        let key = VerdictKey::read(&spec.signing_key)
            .map_err(|e| CampaignError::Config(e.to_string()))?;
        Ok(Self { spec, key })
    }

    /// The candidate file of `model` (none for the base served as it is).
    fn candidate_file(&self, model: &Model, campaign: &str) -> Value {
        let mut file = json!({
            "base": self.spec.base,
            "proposer_commit": self.spec.proposer_commit,
            "brain_commit": self.spec.brain_commit,
            "series": campaign,
            "step": model.generation,
        });
        if let (
            Some(digest),
            Ok(splinter_core::model_ref::ModelRef::Local {
                adapter: Some(path),
                ..
            }),
        ) = (&model.adapter, model.reference())
        {
            file["adapter"] = json!(path);
            file["adapter_digest"] = json!(digest);
        }
        file
    }

    /// The argument list of one `bench examine`.
    fn arguments(
        &self,
        collection: &str,
        candidate: &Path,
        predecessor: Option<&Path>,
        training: &Path,
        out: &Path,
    ) -> Vec<String> {
        let mut args: Vec<String> = vec![
            "examine".into(),
            "--collection".into(),
            collection.into(),
            "--candidate".into(),
            candidate.display().to_string(),
        ];
        if let Some(predecessor) = predecessor {
            args.extend(["--predecessor".into(), predecessor.display().to_string()]);
        }
        args.extend([
            "--training-tasks".into(),
            training.display().to_string(),
            "--alpha".into(),
            ALPHA.to_string(),
            "--out".into(),
            out.display().to_string(),
            "--private".into(),
            self.spec.private.display().to_string(),
            "--signing-key".into(),
            self.spec.signing_key.display().to_string(),
        ]);
        args.extend(self.spec.serving.iter().cloned());
        args
    }

    fn run(&self, args: &[String], cancel: &CancelToken) -> Result<(), CampaignError> {
        let mut child = Command::new("./bench")
            .args(args)
            .current_dir(&self.spec.bench)
            .spawn()
            .map_err(|e| CampaignError::Validator(format!("starting bench: {e}")))?;
        loop {
            if let Some(status) = child
                .try_wait()
                .map_err(|e| CampaignError::Validator(format!("waiting for bench: {e}")))?
            {
                return if status.success() {
                    Ok(())
                } else {
                    Err(CampaignError::Validator(format!(
                        "bench {} ended with {status}",
                        args.join(" ")
                    )))
                };
            }
            if cancel.is_cancelled() {
                let _ = child.kill();
                let _ = child.wait();
                return Err(CampaignError::Cancelled);
            }
            std::thread::sleep(CANCEL_POLL);
        }
    }
}

fn write(path: &Path, value: &Value) -> Result<(), CampaignError> {
    std::fs::write(path, value.to_string())
        .map_err(|e| CampaignError::Validator(format!("{}: {e}", path.display())))
}

impl Examiner for BenchExaminer {
    fn identity(&self) -> Value {
        json!({
            "dev": self.spec.dev,
            "retention": self.spec.retention,
            "slices": self.spec.slices,
            "base": self.spec.base,
            "serving": self.spec.serving,
            "alpha": ALPHA,
        })
    }

    fn slices(&self) -> Vec<String> {
        self.spec.slices.clone()
    }

    fn key(&self) -> &VerdictKey {
        &self.key
    }

    fn examine(
        &mut self,
        _ctx: &Context,
        request: &ExamRequest<'_>,
        cancel: &CancelToken,
    ) -> Result<Examined, CampaignError> {
        let campaign = request.campaign;
        let candidate = request.dir.join("candidate.json");
        write(
            &candidate,
            &self.candidate_file(request.candidate, campaign),
        )?;
        let predecessor = request.predecessor.adapter.as_ref().map(|_| {
            let path = request.dir.join("predecessor.json");
            (path, self.candidate_file(request.predecessor, campaign))
        });
        if let Some((path, file)) = &predecessor {
            write(path, file)?;
        }
        let examined = Examined {
            dev: request.dir.join("dev.json"),
            retention: request.dir.join("retention.json"),
            sealed: request.dir.join("sealed.json"),
        };
        for (collection, out) in [
            (self.spec.dev.as_str(), &examined.dev),
            (self.spec.retention.as_str(), &examined.retention),
            (request.slice, &examined.sealed),
        ] {
            let args = self.arguments(
                collection,
                &candidate,
                predecessor.as_ref().map(|(p, _)| p.as_path()),
                request.training,
                out,
            );
            self.run(&args, cancel)?;
        }
        Ok(examined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_examination_names_the_collection_both_models_the_level_and_the_serving_flags() {
        let examiner = BenchExaminer {
            spec: ExamineSpec {
                bench: "/b".into(),
                signing_key: "/k".into(),
                private: "/p".into(),
                dev: "tasks/dev".into(),
                retention: "tasks/anchor".into(),
                slices: vec!["s1".into()],
                base: "Qwen/Qwen3-4B".into(),
                proposer_commit: "p".into(),
                brain_commit: "b".into(),
                serving: vec!["--device".into(), "gpu0".into()],
            },
            key: VerdictKey::from_bytes([0; 32]),
        };
        let args = examiner.arguments(
            "s1",
            Path::new("/c.json"),
            None,
            Path::new("/t.jsonl"),
            Path::new("/o.json"),
        );
        let text = args.join(" ");
        assert!(text.starts_with("examine --collection s1 --candidate /c.json --training-tasks"));
        assert!(text.contains("--alpha 0.05") && text.ends_with("--device gpu0"));
        assert!(!text.contains("--predecessor"), "the base is the baseline");
    }
}
