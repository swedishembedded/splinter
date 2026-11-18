// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements promotion gates that move a champion only on
// signed, independent evidence, for its clients. If your team needs
// expertise in sealed evaluation of learning agents you can procure our
// services by sending an email to info@swedishembedded.com.

//! The campaign's examiner: bench, run from its checkout, on one of its two
//! protocols. Either way bench writes one signed document of aggregates per
//! comparison (the candidate against the champion over the development
//! split, over the retention suite, and over the generation's own sealed
//! slice), signs it with its own key, and only those documents cross back.
//! Splinter holds the examiner's public key and nothing else; it never reads
//! an item, a job or a grade.
//!
//! * **direct completion** - `bench examine` (`bench.examination/2`): bench
//!   serves each model itself on the build it pins and keeps the item-level
//!   grades in its private directory.
//! * **agent** - `bench run` and `bench verdict` (`bench.verdict/3`): bench
//!   runs Sven on every task of each collection against the endpoint brain
//!   serves the model under, its jobs written into bench's private directory,
//!   then pairs the champion's and the candidate's jobs into a verdict that
//!   names both by their candidate files, checks that brain served each the
//!   model it declares, and spends the sealed slice (its burn record).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_eval::promotion::{PublicKey, ALPHA};
use splinter_orchestrator::context::{Context, ServedEndpoint};

use super::types::CampaignError;
use super::work::{ExamRequest, Examined, Examiner, Model};
use crate::bench::practice::KEY_VARIABLE;

/// How often a running examination is asked whether it was cancelled.
const CANCEL_POLL: Duration = Duration::from_millis(250);

/// How bench examines.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Protocol {
    /// One greedy completion per item, bench serving the model
    /// (`bench examine`).
    #[default]
    DirectCompletion,
    /// Sven works each task in its sandbox against the model's endpoint
    /// (`bench run`, judged by `bench verdict`).
    Agent,
}

/// What the agent protocol needs beyond the collections.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentSpec {
    /// The bench profile each model's is derived from: its main role is
    /// pointed at the endpoint brain serves the model under, nothing else
    /// changes.
    pub profile: PathBuf,
    /// The gain bench's own verdict asks for (`bench verdict --min-gain`).
    /// The gate decides on the aggregates, not on bench's outcome; this only
    /// sets what bench calls Promote and its MDE check.
    pub min_gain: f64,
    /// The serving build both sides declare (`[run].serving_build`): bench
    /// refuses to pair runs that do not name the same one.
    pub serving_build: String,
    /// The side of the development collection's split to run (`bench run
    /// --set`); the whole collection when absent.
    #[serde(default)]
    pub dev_set: Option<String>,
    /// The sven build bench runs the agent from (`BENCH_SVEN_BUILD`); bench's
    /// own default when absent.
    #[serde(default)]
    pub sven_build: Option<PathBuf>,
}

/// What a campaign file's `[examine]` names.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExamineSpec {
    /// The examiner's checkout (`./bench` is run from it).
    pub bench: PathBuf,
    /// The examiner's public key (bench's `verdict.key.pub`): what its
    /// verdicts are verified with. Bench signs with its own private key.
    pub public_key: PathBuf,
    /// The protocol.
    #[serde(default)]
    pub protocol: Protocol,
    /// The examiner's private directory: the item-level grades (direct
    /// completion) or the jobs (agent). Splinter names it and never reads it.
    pub private: PathBuf,
    /// The development split's collection, as bench names it.
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
    /// `--models-dir`, `--api-key-file`, `--device`, ...), passed as given
    /// (direct completion only).
    #[serde(default)]
    pub serving: Vec<String>,
    /// The agent protocol's settings (required for it).
    #[serde(default)]
    pub agent: Option<AgentSpec>,
}

/// Bench as the campaign's examiner.
pub struct BenchExaminer {
    spec: ExamineSpec,
    key: PublicKey,
}

impl BenchExaminer {
    /// The examiner `spec` names, its public key read.
    ///
    /// # Errors
    /// The key cannot be used, or the agent protocol has no settings.
    pub fn new(spec: ExamineSpec) -> Result<Self, CampaignError> {
        let key =
            PublicKey::read(&spec.public_key).map_err(|e| CampaignError::Config(e.to_string()))?;
        match (spec.protocol, &spec.agent) {
            (Protocol::Agent, None) => {
                return Err(CampaignError::Config(
                    "[examine] protocol = \"agent\" needs [examine.agent] (profile, min_gain, \
                     serving_build)"
                        .into(),
                ))
            }
            (Protocol::DirectCompletion, Some(_)) => {
                return Err(CampaignError::Config(
                    "[examine.agent] is for protocol = \"agent\"; direct completion is served \
                     by bench itself (serving)"
                        .into(),
                ))
            }
            _ => {}
        }
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
        ]);
        args.extend(self.spec.serving.iter().cloned());
        args
    }

    fn run(
        &self,
        args: &[String],
        env: &[(&str, &str)],
        cancel: &CancelToken,
    ) -> Result<(), CampaignError> {
        let mut command = Command::new("./bench");
        command.args(args).current_dir(&self.spec.bench);
        for (name, value) in env {
            command.env(name, value);
        }
        let mut child = command
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
                        args.first().map_or("", String::as_str)
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

    /// The three `bench examine` verdicts of `request`.
    fn examine_directly(
        &self,
        request: &ExamRequest<'_>,
        files: &CandidateFiles,
        examined: &Examined,
        cancel: &CancelToken,
    ) -> Result<(), CampaignError> {
        for (collection, out) in [
            (self.spec.dev.as_str(), &examined.dev),
            (self.spec.retention.as_str(), &examined.retention),
            (request.slice, &examined.sealed),
        ] {
            let args = self.arguments(
                collection,
                &files.candidate,
                files.predecessor.as_deref(),
                request.training,
                out,
            );
            self.run(&args, &[], cancel)?;
        }
        Ok(())
    }

    /// The three agent-protocol verdicts of `request`, the champion served
    /// at `endpoints.0` and the candidate at `endpoints.1`.
    ///
    /// # Errors
    /// A bench run or verdict failed, or was cancelled.
    pub fn examine_agentic(
        &self,
        request: &ExamRequest<'_>,
        endpoints: (&ServedEndpoint, &ServedEndpoint),
        examined: &Examined,
        cancel: &CancelToken,
    ) -> Result<(), CampaignError> {
        let agent = self.spec.agent.as_ref().ok_or_else(|| {
            CampaignError::Config("the agent protocol has no [examine.agent]".into())
        })?;
        let files = self.candidate_files(request)?;
        let session = format!("{}-g{:04}", request.campaign, request.generation);
        let jobs = self
            .spec
            .private
            .join("jobs")
            .join(request.campaign)
            .join(format!("g{:04}", request.generation));
        let sides = [("champion", endpoints.0), ("candidate", endpoints.1)];
        let mut profiles = Vec::new();
        for (side, endpoint) in sides {
            let path = request.dir.join(format!("{side}.profile.toml"));
            let text = agent_profile(&agent.profile, endpoint, &session, &agent.serving_build)?;
            std::fs::write(&path, text)
                .map_err(|e| CampaignError::Validator(format!("{}: {e}", path.display())))?;
            profiles.push((side, endpoint, path));
        }
        let comparisons = [
            ("dev", self.spec.dev.as_str(), &examined.dev),
            (
                "retention",
                self.spec.retention.as_str(),
                &examined.retention,
            ),
            ("sealed", request.slice, &examined.sealed),
        ];
        for (role, collection, out) in comparisons {
            let mut ran = Vec::new();
            for (side, endpoint, profile) in &profiles {
                // An interrupted run starts over: no partial job is paired.
                let dir = jobs.join(role).join(side);
                if dir.exists() {
                    std::fs::remove_dir_all(&dir).map_err(|e| {
                        CampaignError::Validator(format!("clearing {}: {e}", dir.display()))
                    })?;
                }
                std::fs::create_dir_all(&dir)
                    .map_err(|e| CampaignError::Validator(format!("{}: {e}", dir.display())))?;
                let set = (role == "dev")
                    .then_some(agent.dev_set.as_deref())
                    .flatten();
                let mut env = vec![(KEY_VARIABLE, endpoint.key.as_str())];
                let sven = agent.sven_build.as_ref().map(|p| p.display().to_string());
                if let Some(sven) = &sven {
                    env.push(("BENCH_SVEN_BUILD", sven));
                }
                self.run(&run_arguments(collection, profile, set, &dir), &env, cancel)?;
                ran.push(job_of(&dir)?);
            }
            let anchor = (role == "retention").then(|| collection_name(collection));
            let args = verdict_arguments(&ran[0], &ran[1], &files, anchor, agent.min_gain, out);
            self.run(&args, &[], cancel)?;
        }
        Ok(())
    }

    /// The candidate files of `request`'s two models, written into its
    /// directory (none for a base predecessor).
    fn candidate_files(&self, request: &ExamRequest<'_>) -> Result<CandidateFiles, CampaignError> {
        let candidate = request.dir.join("candidate.json");
        write(
            &candidate,
            &self.candidate_file(request.candidate, request.campaign),
        )?;
        let predecessor = match request.predecessor.adapter {
            Some(_) => {
                let path = request.dir.join("predecessor.json");
                write(
                    &path,
                    &self.candidate_file(request.predecessor, request.campaign),
                )?;
                Some(path)
            }
            None => None,
        };
        Ok(CandidateFiles {
            candidate,
            predecessor,
        })
    }
}

/// The candidate files of one comparison.
struct CandidateFiles {
    candidate: PathBuf,
    predecessor: Option<PathBuf>,
}

/// The collection's name as bench's verdict names it (`--anchor`): the last
/// component of its path.
fn collection_name(collection: &str) -> String {
    Path::new(collection).file_name().map_or_else(
        || collection.to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// The argument list of one `bench run` of `collection` under `profile`.
fn run_arguments(collection: &str, profile: &Path, set: Option<&str>, jobs: &Path) -> Vec<String> {
    let mut args = vec![
        "run".into(),
        collection.into(),
        "--profile".into(),
        profile.display().to_string(),
    ];
    if let Some(set) = set {
        args.extend(["--set".into(), set.into()]);
    }
    args.extend(["--jobs-dir".into(), jobs.display().to_string()]);
    args
}

/// The argument list of one `bench verdict` of the candidate's `candidate`
/// job against the champion's `base` job, at the gate's level.
fn verdict_arguments(
    base: &Path,
    candidate: &Path,
    files: &CandidateFiles,
    anchor: Option<String>,
    min_gain: f64,
    out: &Path,
) -> Vec<String> {
    let mut args = vec![
        "verdict".into(),
        "--base".into(),
        base.display().to_string(),
        "--candidate".into(),
        candidate.display().to_string(),
        "--candidate-file".into(),
        files.candidate.display().to_string(),
    ];
    if let Some(predecessor) = &files.predecessor {
        args.extend([
            "--predecessor-file".into(),
            predecessor.display().to_string(),
        ]);
    }
    args.extend([
        "--anchor".into(),
        anchor.unwrap_or_else(|| "none".into()),
        "--alpha".into(),
        ALPHA.to_string(),
        "--min-gain".into(),
        min_gain.to_string(),
        "--out".into(),
        out.display().to_string(),
    ]);
    args
}

/// The one run directory `bench run --jobs-dir parent` made (a dated
/// directory holding `bench-run.json`). Only its name is read.
fn job_of(parent: &Path) -> Result<PathBuf, CampaignError> {
    let runs: Vec<PathBuf> = std::fs::read_dir(parent)
        .map_err(|e| CampaignError::Validator(format!("{}: {e}", parent.display())))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("bench-run.json").is_file())
        .collect();
    match runs.as_slice() {
        [one] => Ok(one.clone()),
        _ => Err(CampaignError::Validator(format!(
            "{} holds {} bench runs, not one",
            parent.display(),
            runs.len()
        ))),
    }
}

/// The profile one side runs under: `base` with its main role pointed at
/// the endpoint brain serves the side's model under and the comparison's
/// session, as a practice run derives one, and the serving build both sides
/// declare.
fn agent_profile(
    base: &Path,
    endpoint: &ServedEndpoint,
    session: &str,
    serving_build: &str,
) -> Result<String, CampaignError> {
    let refuse = |reason: String| CampaignError::Config(format!("{}: {reason}", base.display()));
    let derived = crate::bench::practice::policy_profile(base, endpoint, session)
        .map_err(|e| refuse(e.to_string()))?;
    let mut profile: toml::Table = toml::from_str(&derived).map_err(|e| refuse(e.to_string()))?;
    profile
        .get_mut("run")
        .and_then(toml::Value::as_table_mut)
        .ok_or_else(|| refuse("[run] is not a table".into()))?
        .insert("serving_build".into(), serving_build.into());
    toml::to_string(&profile).map_err(|e| refuse(e.to_string()))
}

fn write(path: &Path, value: &Value) -> Result<(), CampaignError> {
    std::fs::write(path, value.to_string())
        .map_err(|e| CampaignError::Validator(format!("{}: {e}", path.display())))
}

impl Examiner for BenchExaminer {
    fn identity(&self) -> Value {
        json!({
            "protocol": self.spec.protocol,
            "dev": self.spec.dev,
            "retention": self.spec.retention,
            "slices": self.spec.slices,
            "base": self.spec.base,
            "serving": self.spec.serving,
            "agent": self.spec.agent.as_ref().map(|a| json!({
                "min_gain": a.min_gain,
                "serving_build": a.serving_build,
                "dev_set": a.dev_set,
                "profile_sha256": std::fs::read(&a.profile)
                    .ok()
                    .map(|bytes| crate::bench::artifact::sha256_hex(&bytes)),
            })),
            "alpha": ALPHA,
        })
    }

    fn slices(&self) -> Vec<String> {
        self.spec.slices.clone()
    }

    fn key(&self) -> &PublicKey {
        &self.key
    }

    fn examine(
        &mut self,
        ctx: &Context,
        request: &ExamRequest<'_>,
        cancel: &CancelToken,
    ) -> Result<Examined, CampaignError> {
        let examined = Examined {
            dev: request.dir.join("dev.json"),
            retention: request.dir.join("retention.json"),
            sealed: request.dir.join("sealed.json"),
        };
        match self.spec.protocol {
            Protocol::DirectCompletion => {
                let files = self.candidate_files(request)?;
                self.examine_directly(request, &files, &examined, cancel)?;
            }
            Protocol::Agent => {
                let endpoint = |model: &Model| {
                    ctx.served_endpoint(&model.reference()?).map_err(|e| {
                        CampaignError::Validator(format!(
                            "no endpoint serves {} for the examiner: {e}",
                            model.model
                        ))
                    })
                };
                let (champion, candidate) =
                    (endpoint(request.predecessor)?, endpoint(request.candidate)?);
                self.examine_agentic(request, (&champion, &candidate), &examined, cancel)?;
            }
        }
        Ok(examined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(protocol: Protocol) -> ExamineSpec {
        ExamineSpec {
            bench: "/b".into(),
            public_key: "/k.pub".into(),
            protocol,
            private: "/p".into(),
            dev: "tasks/dev".into(),
            retention: "tasks/anchor".into(),
            slices: vec!["s1".into()],
            base: "Qwen/Qwen3-4B".into(),
            proposer_commit: "p".into(),
            brain_commit: "b".into(),
            serving: vec!["--device".into(), "gpu0".into()],
            agent: None,
        }
    }

    #[test]
    fn an_examination_names_the_collection_both_models_the_level_and_the_serving_flags() {
        let examiner = BenchExaminer {
            spec: spec(Protocol::DirectCompletion),
            key: PublicKey::from_bytes(
                ed25519_dalek::SigningKey::from_bytes(&[0; 32])
                    .verifying_key()
                    .to_bytes(),
            )
            .unwrap(),
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
        assert!(
            !text.contains("--signing-key"),
            "bench signs with its own key"
        );
    }

    #[test]
    fn an_agentic_verdict_names_the_pair_the_level_and_the_anchor_and_never_a_key() {
        let files = CandidateFiles {
            candidate: "/d/candidate.json".into(),
            predecessor: Some("/d/predecessor.json".into()),
        };
        let args = verdict_arguments(
            Path::new("/p/base-job"),
            Path::new("/p/cand-job"),
            &files,
            Some(collection_name("tasks/anchor")),
            0.05,
            Path::new("/d/retention.json"),
        )
        .join(" ");
        assert!(args.starts_with(
            "verdict --base /p/base-job --candidate /p/cand-job --candidate-file \
             /d/candidate.json --predecessor-file /d/predecessor.json --anchor anchor --alpha 0.05"
        ));
        assert!(!args.contains("key"), "{args}");
        let run = run_arguments(
            "tasks/dev",
            Path::new("/d/c.toml"),
            Some("eval"),
            Path::new("/p/j"),
        );
        assert_eq!(
            run.join(" "),
            "run tasks/dev --profile /d/c.toml --set eval --jobs-dir /p/j"
        );
    }

    #[test]
    fn the_agent_protocol_without_its_settings_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let public = dir.path().join("k.pub");
        let key = ed25519_dalek::SigningKey::from_bytes(&[0; 32]).verifying_key();
        let hex: String = key.to_bytes().iter().map(|b| format!("{b:02x}")).collect();
        std::fs::write(&public, hex).unwrap();
        let mut agentic = spec(Protocol::Agent);
        agentic.public_key = public;
        assert!(matches!(
            BenchExaminer::new(agentic),
            Err(CampaignError::Config(_))
        ));
    }
}
