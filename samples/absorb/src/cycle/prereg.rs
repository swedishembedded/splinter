// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it without
// forgetting what it knew, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The pre-registration: everything that decides what a run means, written
//! and hashed before its first session, and the ledger that counts every full
//! run.

use std::io::Write as _;
use std::path::Path;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::absorb::AbsorbRequest;
use splinter_sdk::train::Tuning;
use splinter_sdk::vocabulary::role::Role;

use super::plan::CycleRole;
use super::verdict::Rule;
use crate::probes::sha256_hex;
use crate::runtime::{DEFAULT_JUDGE, DEFAULT_MODEL};

/// The pre-registration's schema name.
pub const SCHEMA: &str = "absorb-cycle-prereg-v1";

/// Its file name under the cycle's directory.
pub const FILE: &str = "prereg.json";

/// The ledger of runs, under the directory of the run's manifest.
pub const LEDGER: &str = "cycle-runs.jsonl";

/// The sources of the analysis: what groups the facts, grades the answers,
/// applies the rule and words the verdict. Their hash is registered, so a
/// report from other code is refused.
const ANALYSIS_SOURCES: [(&str, &str); 6] = [
    ("stats.rs", include_str!("stats.rs")),
    ("items.rs", include_str!("items.rs")),
    ("grade.rs", include_str!("grade.rs")),
    ("analysis.rs", include_str!("analysis.rs")),
    ("verdict.rs", include_str!("verdict.rs")),
    ("report.rs", include_str!("report.rs")),
];

/// The hash of the analysis sources compiled into this binary.
#[must_use]
pub fn analysis_sha256() -> String {
    let mut bytes = Vec::new();
    for (name, text) in ANALYSIS_SOURCES {
        bytes.extend_from_slice(format!("{name}\0{}\0{text}\0", text.len()).as_bytes());
    }
    sha256_hex(&bytes)
}

/// The settings of `splinter absorb` every night runs with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbsorbSettings {
    /// The base the policy is, with its context limit.
    pub policy: String,
    /// The model that plays the user.
    pub simulator: String,
    /// The generator role; the policy plays it when absent.
    pub generator: Option<String>,
    /// The teacher role; the policy plays it when absent.
    pub teacher: Option<String>,
    /// The judge role.
    pub judge: String,
    /// Differently worded questions written per claim.
    pub paraphrases: usize,
    /// The share of the training draws that are the base's own answers.
    pub rehearsal_share: f64,
    /// The most passes over the night's records.
    pub epochs: u32,
    /// The step budget, in place of the epochs.
    pub steps: Option<u32>,
    /// LoRA rank.
    pub rank: u32,
    /// Hold the base at bf16 while training.
    pub bf16_base: bool,
    /// Continue the release instead of training again from the base.
    pub continue_from_release: bool,
    /// The share of earlier records replayed when continuing.
    pub replay_fraction: f64,
    /// Extraction passes per session.
    pub extraction_passes: u32,
    /// The judge checks that the person's words assert each claim.
    pub entail: bool,
}

impl AbsorbSettings {
    /// The policy unless another is named.
    pub const DEFAULT_POLICY: &'static str = DEFAULT_MODEL;
    /// The model that plays the user unless another is named: the judge's, a
    /// different model from the policy.
    pub const DEFAULT_SIMULATOR: &'static str = DEFAULT_JUDGE;
    /// The judge unless another is named.
    pub const DEFAULT_JUDGE: &'static str = DEFAULT_JUDGE;
}

impl Default for AbsorbSettings {
    fn default() -> Self {
        let request = AbsorbRequest::default();
        Self {
            policy: Self::DEFAULT_POLICY.to_string(),
            simulator: Self::DEFAULT_SIMULATOR.to_string(),
            generator: None,
            teacher: None,
            judge: Self::DEFAULT_JUDGE.to_string(),
            paraphrases: request.paraphrases,
            rehearsal_share: request.rehearsal_share,
            epochs: request.epochs,
            steps: request.steps,
            rank: request.rank,
            bf16_base: true,
            continue_from_release: request.continue_from_release,
            replay_fraction: request.replay_fraction,
            extraction_passes: request.extraction_passes,
            entail: request.entail,
        }
    }
}

impl AbsorbSettings {
    /// The `absorb` of a night: `sessions` is the night's directory of
    /// recordings, `subjects` the file naming what each teaches, `sealed` the
    /// sealed probes no training record may contain. The policy is the
    /// release in effect (`policy:default`).
    ///
    /// # Errors
    /// A model reference is malformed.
    pub fn request(
        &self,
        sessions: &Path,
        subjects: &Path,
        sealed: &Path,
    ) -> anyhow::Result<AbsorbRequest> {
        let role = |reference: &str| crate::runtime::model_ref(reference);
        let mut roles = splinter_sdk::vocabulary::role::RoleOverrides::new();
        roles.insert(Role::Judge, role(&self.judge)?);
        if let Some(generator) = &self.generator {
            roles.insert(Role::Generator, role(generator)?);
        }
        if let Some(teacher) = &self.teacher {
            roles.insert(Role::Teacher, role(teacher)?);
        }
        Ok(AbsorbRequest {
            sessions: vec![sessions.to_path_buf()],
            roles,
            paraphrases: self.paraphrases,
            rehearsal_share: self.rehearsal_share,
            epochs: self.epochs,
            steps: self.steps,
            rank: self.rank,
            tuning: Tuning {
                bf16_base: self.bf16_base,
                ..Tuning::default()
            },
            continue_from_release: self.continue_from_release,
            replay_fraction: self.replay_fraction,
            extraction_passes: self.extraction_passes,
            entail: self.entail,
            subjects: vec![subjects.to_path_buf()],
            sealed_probes: vec![sealed.to_path_buf()],
            ..AbsorbRequest::default()
        })
    }
}

/// Everything registered before the first session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prereg {
    /// [`SCHEMA`].
    pub schema: String,
    /// Which full run of this fact pool this is, counted from 1.
    pub run_counter: usize,
    /// The commit of the pipeline.
    pub pipeline_commit: String,
    /// Whether the tree had changes beyond that commit.
    pub dirty: bool,
    /// Which facts the cycle teaches.
    pub role: CycleRole,
    /// The nights the run goes on for.
    pub days: usize,
    /// The arm of the protocol the run is: `main` or an ablation.
    pub arm: String,
    /// `splinter absorb` as every night runs it.
    pub absorb: AbsorbSettings,
    /// The sha256 of the sealed probes.
    pub probes_sha256: String,
    /// The sha256 of the manifest.
    pub manifest_sha256: String,
    /// The sha256 of the plan file.
    pub plan_sha256: String,
    /// The sha256 of the item list every night is evaluated on.
    pub items_sha256: String,
    /// The sha256 of the anchor suite.
    pub anchor_sha256: String,
    /// The hash of the analysis sources ([`analysis_sha256`]).
    pub analysis_sha256: String,
    /// The rule the verdict applies.
    pub rule: Rule,
    /// When it was written.
    pub registered_at: String,
    /// The hash of everything above, with this field empty.
    pub hash: String,
}

impl Prereg {
    /// The hash of `self` with its `hash` field empty.
    ///
    /// # Errors
    /// The registration cannot be serialised.
    pub fn compute_hash(&self) -> anyhow::Result<String> {
        let mut bare = self.clone();
        bare.hash = String::new();
        Ok(sha256_hex(&serde_json::to_vec(&bare)?))
    }

    /// Sets the hash.
    ///
    /// # Errors
    /// The registration cannot be serialised.
    pub fn sealed(mut self) -> anyhow::Result<Self> {
        self.hash = self.compute_hash()?;
        Ok(self)
    }

    /// Reads the pre-registration under `dir`.
    ///
    /// # Errors
    /// There is none, or it is not one.
    pub fn read(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join(FILE);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("{}: no pre-registration", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("{} is not one", path.display()))
    }

    /// Writes it under `dir`, once: a registered run is never re-registered.
    ///
    /// # Errors
    /// One exists, or the file cannot be written.
    pub fn write(&self, dir: &Path) -> anyhow::Result<()> {
        let path = dir.join(FILE);
        anyhow::ensure!(
            !path.exists(),
            "{} exists: a registered run is not registered again",
            path.display()
        );
        std::fs::write(&path, serde_json::to_string_pretty(self)? + "\n")
            .with_context(|| format!("writing {}", path.display()))
    }

    /// Refuses a pre-registration that was edited after it was hashed, or
    /// that registered other analysis code than this binary holds.
    ///
    /// # Errors
    /// The hash does not match, or the analysis differs.
    pub fn verify(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == SCHEMA,
            "schema {:?}, not {SCHEMA:?}",
            self.schema
        );
        let found = self.compute_hash()?;
        anyhow::ensure!(
            self.hash == found,
            "the pre-registration hashes to {found}, not the {} it records: it was edited after it was registered",
            self.hash
        );
        anyhow::ensure!(
            self.analysis_sha256 == analysis_sha256(),
            "the pre-registration registered the analysis {}, and this binary holds {}",
            self.analysis_sha256,
            analysis_sha256()
        );
        Ok(())
    }
}

/// One line of the ledger of runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Entry {
    /// A full run began; its directory.
    Start {
        /// The run's number.
        run: usize,
        /// Its directory.
        dir: String,
    },
    /// The pre-registration of a run was written.
    Registered {
        /// The run's number.
        run: usize,
        /// The pre-registration's hash.
        hash: String,
    },
    /// The verdict of a run was computed.
    Finished {
        /// The run's number.
        run: usize,
        /// Whether the rule was met.
        met: bool,
        /// The verdict in words.
        text: String,
    },
}

/// Every entry of the ledger under `manifest_dir`.
///
/// # Errors
/// A line is not an entry.
pub fn read_ledger(manifest_dir: &Path) -> anyhow::Result<Vec<Entry>> {
    let path = manifest_dir.join(LEDGER);
    if !path.exists() {
        return Ok(Vec::new());
    }
    std::fs::read_to_string(&path)?
        .lines()
        .enumerate()
        .map(|(n, l)| {
            serde_json::from_str(l).with_context(|| format!("{} line {}", path.display(), n + 1))
        })
        .collect()
}

/// Appends `entry` to the ledger under `manifest_dir`.
///
/// # Errors
/// The ledger cannot be written.
pub fn record(manifest_dir: &Path, entry: &Entry) -> anyhow::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(manifest_dir.join(LEDGER))?;
    writeln!(file, "{}", serde_json::to_string(entry)?)?;
    Ok(())
}

/// The number of the next full run of the pool in `manifest_dir`.
///
/// # Errors
/// The ledger cannot be read.
pub fn next_run(manifest_dir: &Path) -> anyhow::Result<usize> {
    Ok(1 + read_ledger(manifest_dir)?
        .iter()
        .filter(|e| matches!(e, Entry::Start { .. }))
        .count())
}
