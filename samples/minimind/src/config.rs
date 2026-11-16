// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines sized from a laptop smoke
// test to a GPU run from one small configuration file, for its clients. If
// your team needs expertise in reproducible small-model training, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The run's configuration: a profile's defaults, overridden by
//! `minimind.yaml`.
//!
//! A profile is a complete configuration; the file only changes what it
//! names, and a name it does not know is refused rather than ignored. Every
//! stage reads its own slice ([`Config::slice`]), and only that slice enters
//! its ledger key, so changing the SFT step count recomputes SFT and what
//! follows it, never the world or pretraining.

use std::path::Path;

use anyhow::{bail, Context as _};
use serde::{Deserialize, Serialize};

/// A named, complete configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    /// About a million parameters on the CPU in minutes: the tests' and CI's
    /// profile.
    Smoke,
    /// About fifteen million parameters, sized for a fast turnaround.
    Default,
    /// About sixty-four million parameters; needs a GPU.
    #[serde(rename = "64m")]
    M64,
}

impl Profile {
    /// The profile named `name` (`smoke`, `default`, `64m`).
    ///
    /// # Errors
    /// No profile has that name.
    pub fn parse(name: &str) -> anyhow::Result<Self> {
        match name {
            "smoke" => Ok(Self::Smoke),
            "default" => Ok(Self::Default),
            "64m" => Ok(Self::M64),
            other => bail!("no profile {other:?}: use smoke, default or 64m"),
        }
    }

    /// The profile's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Default => "default",
            Self::M64 => "64m",
        }
    }
}

/// One model the training sequence is run on: an architecture brain knows
/// and that architecture's own configuration, handed to brain unread. The
/// sample has no code path per architecture; brain refuses one that lacks a
/// capability a stage needs, by name.
pub use splinter_sdk::model::scratch::ArchSpec as ModelSpec;

/// One training run's budget.
pub use splinter_sdk::model::scratch::Budget as TrainBudget;

/// An Ouro looped decoder's configuration.
fn ouro(hidden: u32, layers: u32, passes: u32, heads: u32, vocab: u32, context: u32) -> ModelSpec {
    // The architecture's own `config.json` spelling, handed to brain unread.
    let config = serde_json::json!({
        "architectures": ["OuroForCausalLM"], "model_type": "ouro",
        "hidden_size": hidden, "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": heads, "head_dim": hidden / heads,
        "intermediate_size": (hidden * 11 / 4).div_ceil(32) * 32,
        "max_position_embeddings": context, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
        "total_ut_steps": passes, "tie_word_embeddings": false,
    });
    ModelSpec {
        arch: "ouro".into(),
        vocab,
        context,
        config,
    }
}

/// How much of the synthetic world is written.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldSize {
    /// Towns described; each brings its people and goods.
    pub towns: u32,
    /// Arithmetic expressions with their values.
    pub expressions: u32,
    /// Tiny programs with their outputs.
    pub programs: u32,
    /// Serialized conversations in the model's chat template.
    pub conversations: u32,
    /// Worked tool-use conversations of the training families, tools run
    /// for real, drawn from a stream no split uses.
    pub trajectories: u32,
}

/// How many task instances each use draws.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCounts {
    /// Training-family instances the expert demonstrates.
    pub demonstrations: u32,
    /// Fresh training-family instances the model explores.
    pub explore: u32,
    /// Development instances probed after every model stage.
    pub probes: u32,
}

/// Bounds on one agent episode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentLimits {
    /// Tool calls before the episode is stopped.
    pub max_tool_calls: u32,
    /// Wall-clock limit of one episode, in seconds.
    pub deadline_secs: u64,
    /// Longest reply the model may write, in tokens.
    pub max_output_tokens: u32,
    /// Episodes run at once; the model server batches their requests.
    pub concurrency: u32,
}

/// A corpus fetched at run time when the world cannot provide it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FetchedCorpus {
    /// The corpus's name; its cache entry is named after it.
    pub name: String,
    /// The pinned URL (a specific revision, never a moving branch).
    pub url: String,
    /// The SHA-256 the download must have.
    pub sha256: String,
    /// The SPDX identifier of its license, recorded in the run manifest.
    pub license: String,
    /// The most bytes of it the pretraining corpus takes.
    pub slice_bytes: u64,
}

/// The whole configuration of one run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The profile the defaults came from.
    pub profile: Profile,
    /// The models the sequence is run on, by label; each gets its own
    /// checkpoints, probes and examination over the shared data stages.
    pub models: std::collections::BTreeMap<String, ModelSpec>,
    /// The synthetic world's size.
    pub world: WorldSize,
    /// Task instance counts.
    pub tasks: TaskCounts,
    /// Pretraining on raw text.
    pub pretrain: TrainBudget,
    /// Supervised fine-tuning on the expert's demonstrations.
    pub sft: TrainBudget,
    /// Training on the model's own verified episodes.
    pub improve: TrainBudget,
    /// Episode bounds.
    pub agent: AgentLimits,
    /// Corpora fetched at run time, none by default.
    #[serde(default)]
    pub fetch: Vec<FetchedCorpus>,
}

impl Config {
    /// The complete configuration of `profile`.
    #[must_use]
    pub fn of(profile: Profile) -> Self {
        let budget = |steps, batch, lr, block| TrainBudget {
            steps,
            batch,
            lr,
            block,
        };
        let world = |towns, expressions, programs, conversations, trajectories| WorldSize {
            towns,
            expressions,
            programs,
            conversations,
            trajectories,
        };
        let tasks = |demonstrations, explore, probes| TaskCounts {
            demonstrations,
            explore,
            probes,
        };
        let (model, world, tasks, pretrain, sft, improve) = match profile {
            Profile::Smoke => (
                ouro(256, 4, 2, 4, 1024, 2048),
                world(200, 4_000, 1_500, 1_000, 6_000),
                tasks(1_600, 96, 12),
                budget(600, 128, 2e-3, 256),
                budget(4_000, 4, 7e-4, 1024),
                budget(1_000, 4, 3e-4, 1024),
            ),
            Profile::Default => (
                ouro(256, 4, 4, 4, 4096, 2048),
                world(60, 20_000, 6_000, 3_000, 3_000),
                tasks(400, 200, 8),
                budget(3_000, 16, 1e-3, 512),
                budget(600, 8, 3e-4, 2048),
                budget(300, 8, 1e-4, 2048),
            ),
            Profile::M64 => (
                ouro(512, 8, 4, 8, 8192, 2048),
                world(200, 100_000, 30_000, 15_000, 15_000),
                tasks(2_000, 1_000, 8),
                budget(20_000, 32, 6e-4, 1024),
                budget(2_000, 16, 2e-4, 2048),
                budget(1_000, 16, 1e-4, 2048),
            ),
        };
        Self {
            profile,
            models: [("ouro".to_string(), model)].into_iter().collect(),
            world,
            tasks,
            pretrain,
            sft,
            improve,
            agent: AgentLimits {
                max_tool_calls: 8,
                deadline_secs: 300,
                max_output_tokens: 256,
                concurrency: 16,
            },
            fetch: Vec::new(),
        }
    }

    /// The configuration in `minimind.yaml` at `path`: its `profile` (or
    /// `default_profile` when it names none) with the file's tables laid over
    /// it key by key.
    ///
    /// # Errors
    /// The file cannot be read, is not YAML, names an unknown profile or key,
    /// or gives a key a value of the wrong type.
    /// The file is YAML.
    pub fn load(path: &Path, default_profile: Profile) -> anyhow::Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text, default_profile).with_context(|| format!("in {}", path.display()))
    }

    /// [`Config::load`] of text already read.
    ///
    /// # Errors
    /// As [`Config::load`].
    pub fn parse(text: &str, default_profile: Profile) -> anyhow::Result<Self> {
        let file: serde_json::Value = serde_yaml::from_str(text)?;
        let mut file = match file {
            serde_json::Value::Object(map) => map,
            serde_json::Value::Null => serde_json::Map::new(),
            other => bail!("the configuration is a mapping, not {other}"),
        };
        let profile = match file.remove("profile") {
            Some(serde_json::Value::String(name)) => Profile::parse(&name)?,
            Some(other) => bail!("profile must be a string, not {other}"),
            None => default_profile,
        };
        let mut merged = serde_json::to_value(Self::of(profile))?;
        let serde_json::Value::Object(base) = &mut merged else {
            unreachable!("a configuration serialises to a mapping")
        };
        merge(base, file, "")?;
        Ok(serde_json::from_value(merged)?)
    }
}

/// Lays `over` onto `base` key by key. A key `base` does not have is
/// refused, except where new names are the point: a new model label under
/// `models`, an architecture's own `config` settings, and `fetch` entries.
fn merge(
    base: &mut serde_json::Map<String, serde_json::Value>,
    over: serde_json::Map<String, serde_json::Value>,
    path: &str,
) -> anyhow::Result<()> {
    use serde_json::Value;
    let open = path == "models" || (path.starts_with("models.") && path.ends_with(".config"));
    for (key, value) in over {
        let here = if path.is_empty() {
            key.clone()
        } else {
            format!("{path}.{key}")
        };
        match (base.get_mut(&key), value) {
            (Some(Value::Object(b)), Value::Object(o)) => merge(b, o, &here)?,
            (Some(slot), value) if same_kind(slot, &value) => *slot = value,
            (None, value) if open => {
                base.insert(key, value);
            }
            (None, _) => bail!("no setting {here:?}"),
            (Some(_), value) => bail!("{here} cannot be set to {value}"),
        }
    }
    Ok(())
}

/// Two values of one kind; any two numbers are one kind.
fn same_kind(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_overrides_only_what_it_names() {
        let config = Config::parse("profile: smoke\nsft:\n  steps: 7\n", Profile::Default).unwrap();
        assert_eq!(config.profile, Profile::Smoke);
        assert_eq!(config.sft.steps, 7);
        assert_eq!(config.pretrain, Config::of(Profile::Smoke).pretrain);
        let two = Config::parse(
            "models:\n  ouro:\n    config:\n      hidden_size: 32\n  qwen3:\n    arch: qwen3\n    vocab: 1024\n    context: 512\n    config:\n      hidden_size: 64\n",
            Profile::Smoke,
        )
        .unwrap();
        assert_eq!(
            two.models.len(),
            2,
            "a second architecture is configuration"
        );
        assert_eq!(two.models["ouro"].config["hidden_size"], 32);
        assert_eq!(
            two.models["ouro"].config["num_hidden_layers"],
            Config::of(Profile::Smoke).models["ouro"].config["num_hidden_layers"]
        );
    }

    #[test]
    fn an_unknown_setting_is_refused() {
        assert!(Config::parse("sft:\n  stepz: 7\n", Profile::Smoke).is_err());
        assert!(Config::parse("nonsense:\n  a: 1\n", Profile::Smoke).is_err());
        assert!(Config::parse("profile: huge\n", Profile::Smoke).is_err());
        assert!(Config::parse("sft:\n  steps: many\n", Profile::Smoke).is_err());
    }

    #[test]
    fn a_profile_is_taken_from_the_caller_when_the_file_names_none() {
        let config = Config::parse("", Profile::Default).unwrap();
        assert_eq!(config, Config::of(Profile::Default));
    }
}
