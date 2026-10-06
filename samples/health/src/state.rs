// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible experiment bookkeeping for
// risk-model campaigns: what was imported, split, trained, measured and
// released, kept beside the state it names. If your team needs expertise in
// auditable model-development workflows, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The run directory: where one pass through the sample keeps its files.
//!
//! ```text
//! <dir>/source/    the synthetic source file and its declaration (`synth`)
//! <dir>/state/     Splinter's state root: episodes, datasets, candidates,
//!                  evaluations, releases, aliases
//! <dir>/split.json the split `split` stored, which `train` and `eval` read
//! <dir>/labels.json names given to candidates and evaluations
//! <dir>/predict/   releases unpacked for `predict`
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::store::StateRoot;
use splinter_sdk::timeline::data::SplitReport;
use splinter_sdk::{Config, Context};

/// The secret the opaque participant keys are derived with when none is
/// given. The sample's data is synthetic; a campaign on real data keeps its
/// own.
pub const DEFAULT_SECRET: &str = "splinter-health-sample";

/// A run directory.
#[derive(Clone, Debug)]
pub struct Run {
    /// The directory.
    pub dir: PathBuf,
}

/// What `split` recorded for the later commands: the split and which
/// imported dataset it cut.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitRecord {
    /// The imported dataset the split cut.
    pub dataset: String,
    /// The stored parts and the counts.
    pub report: SplitReportRecord,
}

/// [`SplitReport`], as the run directory keeps it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitReportRecord {
    /// The split's address.
    pub split: String,
    /// The stored training part.
    pub train: String,
    /// The stored validation part.
    pub validation: String,
    /// The stored test part.
    pub test: String,
    /// Participants in the dataset.
    pub participants: usize,
    /// Records in each part.
    pub records: BTreeMap<String, usize>,
    /// Participants left out, by reason.
    pub excluded: BTreeMap<String, usize>,
    /// Participants a part's projection left out.
    pub unprojected: usize,
}

impl From<&SplitReport> for SplitReportRecord {
    fn from(r: &SplitReport) -> Self {
        Self {
            split: r.split.to_string(),
            train: r.train.to_string(),
            validation: r.validation.to_string(),
            test: r.test.to_string(),
            participants: r.participants,
            records: r
                .records
                .iter()
                .map(|(p, n)| (p.name().to_owned(), *n))
                .collect(),
            excluded: r.excluded.clone(),
            unprojected: r.unprojected,
        }
    }
}

impl Run {
    /// The run directory `dir`; it is created by the commands that write.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Where the synthetic source and its declaration are written.
    pub fn source_dir(&self) -> PathBuf {
        self.dir.join("source")
    }

    /// Splinter's state root.
    pub fn state_root(&self) -> StateRoot {
        StateRoot::new(self.dir.join("state"))
    }

    /// The release unpacked for `predict`.
    pub fn predict_dir(&self) -> PathBuf {
        self.dir.join("predict")
    }

    /// A context over this run's state: no model is loaded, no network is
    /// reachable, nothing is read from the environment.
    pub fn context(&self) -> Result<Context> {
        let root = self.state_root();
        std::fs::create_dir_all(root.path())
            .with_context(|| format!("creating {}", root.path().display()))?;
        let unused = self.dir.join("unused");
        let config = Config {
            state_root: root,
            model_store: unused.join("models"),
            policy_base: unused.join("policy"),
            policy_context_tokens: None,
            openrouter_api_key: None,
            brain_api_key: None,
            allow_remote: false,
            command_env: BTreeMap::new(),
            working_dir: self.dir.clone(),
            brain_binary: None,
            front_door_model: None,
            assistant_model: None,
            judge_model: None,
            bf16_base: false,
            default_budget: None,
            remote_concurrency: 1,
            min_calibration_controls: 1,
            thinking: false,
        };
        Ok(Context::new(config, false)?)
    }

    fn json_path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    fn read<T: for<'de> Deserialize<'de>>(&self, name: &str, hint: &str) -> Result<T> {
        let path = self.json_path(name);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}: {hint}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    fn write<T: Serialize>(&self, name: &str, value: &T) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.json_path(name);
        let pending = path.with_extension("pending");
        std::fs::write(&pending, serde_json::to_vec_pretty(value)?)?;
        std::fs::rename(&pending, &path).with_context(|| format!("writing {}", path.display()))
    }

    /// Keeps a command's report as `reports/<name>.json`: brain writes its
    /// training log to the same standard output, so a script reads the report
    /// from the file.
    pub fn report<T: Serialize>(&self, name: &str, value: &T) -> Result<PathBuf> {
        let dir = self.dir.join("reports");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{name}.json"));
        std::fs::write(&path, serde_json::to_vec_pretty(value)?)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }

    /// The split `split` stored last.
    pub fn split(&self) -> Result<SplitRecord> {
        self.read("split.json", "run `split` first")
    }

    /// Remembers the split.
    pub fn save_split(&self, record: &SplitRecord) -> Result<()> {
        self.write("split.json", record)
    }

    fn labels(&self) -> Result<BTreeMap<String, String>> {
        if self.json_path("labels.json").exists() {
            self.read("labels.json", "")
        } else {
            Ok(BTreeMap::new())
        }
    }

    /// Names `id` as `label`; a label names one thing, so naming it again
    /// for another is refused.
    pub fn label(&self, label: &str, id: &str) -> Result<()> {
        let mut labels = self.labels()?;
        if let Some(existing) = labels.get(label) {
            anyhow::ensure!(
                existing == id,
                "label {label:?} already names {existing}; choose another"
            );
        }
        labels.insert(label.to_owned(), id.to_owned());
        self.write("labels.json", &labels)
    }

    /// What `label` names, or `label` itself when it names nothing (an id).
    pub fn resolve(&self, label: &str) -> Result<String> {
        Ok(self
            .labels()?
            .get(label)
            .cloned()
            .unwrap_or_else(|| label.to_owned()))
    }
}
