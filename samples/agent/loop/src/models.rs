// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Model versions: candidates, the decision to promote one, and the way back.
//!
//! A trained adapter is a candidate and never the model in use. Whether it
//! replaces the current one is decided by a rule declared here, before any
//! result exists, over paired outcomes of the same tasks run once with the
//! current model and once with the candidate: a candidate is promoted only
//! when it solves more of them, by a margin a one-sided sign test over the
//! tasks only one of the two solved does not put down to chance. A rejected
//! candidate stays on record with the numbers that rejected it, and the
//! model in use does not change. Promotion keeps what it replaced, so
//! `rollback` returns to it at once. The registry is a single file written
//! atomically, under a lock file so two writers cannot interleave.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::model::stats::sign_test;
use splinter_sdk::vocabulary::clock::utc_now;

use crate::store::{read_json, write_json, LoopHome};

/// The version of the registry file.
pub const REGISTRY_SCHEMA: u32 = 1;

/// Fewest paired tasks a decision may rest on.
pub const MIN_TASKS: usize = 8;

/// The one-sided sign-test p-value at or below which a candidate that solves
/// more tasks is promoted.
pub const MAX_P_VALUE: f64 = 0.10;

/// Where a version stands.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Standing {
    /// Trained, not yet judged.
    Candidate,
    /// Judged better and made current.
    Promoted,
    /// Judged and left out.
    Rejected,
}

/// What a decision rested on and what it was.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Decision {
    /// When it was made.
    pub at: String,
    /// The rule, in words, as it was applied.
    pub rule: String,
    /// Paired tasks.
    pub tasks: usize,
    /// Tasks the model in use solved.
    pub baseline_solved: usize,
    /// Tasks the candidate solved.
    pub candidate_solved: usize,
    /// Tasks only the candidate solved.
    pub gained: usize,
    /// Tasks only the model in use solved.
    pub lost: usize,
    /// One-sided sign-test p-value over the discordant tasks.
    pub p_value: f64,
    /// Whether the candidate was promoted.
    pub promoted: bool,
    /// Why, in a sentence.
    pub reason: String,
}

/// One trained adapter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Version {
    /// Its id.
    pub id: String,
    /// The adapter file.
    pub adapter: PathBuf,
    /// Its digest.
    pub adapter_digest: String,
    /// The base model it applies to, as `local:` names it.
    pub base: String,
    /// The digest of the dataset it was trained on.
    pub dataset_digest: String,
    /// Brain's training record of the run.
    pub training_record: PathBuf,
    /// When it was registered.
    pub created_at: String,
    /// Where it stands.
    pub standing: Standing,
    /// The decision, once there is one.
    pub decision: Option<Decision>,
}

/// The registry file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Registry {
    /// The file's version.
    pub schema: u32,
    /// The model in use when no adapter is promoted.
    pub base: String,
    /// Every version ever registered, oldest first.
    pub versions: Vec<Version>,
    /// The version in use, if an adapter is.
    pub current: Option<String>,
    /// The versions that were current before, most recent last.
    pub history: Vec<String>,
}

/// The registry's file.
fn path(home: &LoopHome) -> PathBuf {
    home.root().join("models").join("registry.json")
}

/// An exclusive hold on the registry, released when dropped.
struct Locked(PathBuf);

impl Locked {
    fn take(home: &LoopHome) -> Result<Self> {
        let lock = path(home).with_extension("lock");
        std::fs::create_dir_all(lock.parent().context("registry directory")?)?;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock)
            .with_context(|| {
                format!("the registry is locked by another writer ({} exists; remove it if no writer is running)", lock.display())
            })?;
        Ok(Self(lock))
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The registry, or a fresh one over `base` when none exists.
pub fn load(home: &LoopHome, base: &str) -> Result<Registry> {
    let file = path(home);
    if !file.exists() {
        return Ok(Registry {
            schema: REGISTRY_SCHEMA,
            base: base.to_string(),
            versions: Vec::new(),
            current: None,
            history: Vec::new(),
        });
    }
    let registry: Registry = read_json(&file)?;
    if registry.schema != REGISTRY_SCHEMA {
        bail!(
            "{} has schema {}, this build reads {REGISTRY_SCHEMA}",
            file.display(),
            registry.schema
        );
    }
    Ok(registry)
}

/// Runs `change` on the registry under the lock and writes the result.
fn update<R>(
    home: &LoopHome,
    base: &str,
    change: impl FnOnce(&mut Registry) -> Result<R>,
) -> Result<R> {
    let _lock = Locked::take(home)?;
    let mut registry = load(home, base)?;
    let result = change(&mut registry)?;
    write_json(&path(home), &registry)?;
    Ok(result)
}

/// Adds a trained adapter as a candidate and returns its id.
pub fn register(
    home: &LoopHome,
    base: &str,
    adapter: &Path,
    adapter_digest: &str,
    dataset_digest: &str,
    training_record: &Path,
) -> Result<String> {
    update(home, base, |r| {
        let id = format!("v{}", r.versions.len() + 1);
        r.versions.push(Version {
            id: id.clone(),
            adapter: adapter.to_path_buf(),
            adapter_digest: adapter_digest.to_string(),
            base: base.to_string(),
            dataset_digest: dataset_digest.to_string(),
            training_record: training_record.to_path_buf(),
            created_at: utc_now(),
            standing: Standing::Candidate,
            decision: None,
        });
        Ok(id)
    })
}

/// One task solved or not, by the two models.
#[derive(Clone, Copy, Debug)]
pub struct Paired {
    /// The model in use solved it.
    pub baseline: bool,
    /// The candidate solved it.
    pub candidate: bool,
}

/// The tasks both result sets cover, by name: a result file is a task's
/// `agent-loop run --json` output, solved when its status is `accepted`.
pub fn pair_results(baseline: &Path, candidate: &Path) -> Result<BTreeMap<String, Paired>> {
    let solved = |dir: &Path| -> Result<BTreeMap<String, bool>> {
        let mut out = BTreeMap::new();
        for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let file = entry?.path();
            if file.extension().is_none_or(|x| x != "json") {
                continue;
            }
            let value: serde_json::Value = read_json(&file)?;
            let name = file
                .file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            out.insert(name, value["status"] == "accepted");
        }
        Ok(out)
    };
    let (b, c) = (solved(baseline)?, solved(candidate)?);
    let mut paired = BTreeMap::new();
    for (name, base) in b {
        if let Some(cand) = c.get(&name) {
            paired.insert(
                name,
                Paired {
                    baseline: base,
                    candidate: *cand,
                },
            );
        }
    }
    Ok(paired)
}

/// The decision the declared rule makes over `pairs`.
#[must_use]
pub fn decide(pairs: &BTreeMap<String, Paired>) -> Decision {
    let rule = format!(
        "promote when at least {MIN_TASKS} tasks are paired, the candidate solves more of them, \
         and the one-sided sign test over the tasks only one solved has p <= {MAX_P_VALUE}"
    );
    let tasks = pairs.len();
    let baseline_solved = pairs.values().filter(|p| p.baseline).count();
    let candidate_solved = pairs.values().filter(|p| p.candidate).count();
    let gained = pairs
        .values()
        .filter(|p| p.candidate && !p.baseline)
        .count();
    let lost = pairs
        .values()
        .filter(|p| p.baseline && !p.candidate)
        .count();
    let test = sign_test(
        &pairs
            .values()
            .map(|p| (p.candidate, p.baseline))
            .collect::<Vec<_>>(),
    );
    let (promoted, reason) = if tasks < MIN_TASKS {
        (
            false,
            format!("only {tasks} paired tasks, fewer than {MIN_TASKS}"),
        )
    } else if candidate_solved <= baseline_solved {
        (false, format!("the candidate solved {candidate_solved}, no more than the model in use ({baseline_solved})"))
    } else if test.p_value > MAX_P_VALUE {
        (
            false,
            format!(
                "gained {gained}, lost {lost}: p = {:.3} is above {MAX_P_VALUE}",
                test.p_value
            ),
        )
    } else {
        (
            true,
            format!("gained {gained}, lost {lost}: p = {:.3}", test.p_value),
        )
    };
    Decision {
        at: utc_now(),
        rule,
        tasks,
        baseline_solved,
        candidate_solved,
        gained,
        lost,
        p_value: test.p_value,
        promoted,
        reason,
    }
}

/// Judges `candidate` over `pairs`: promotes it or records its rejection.
/// The decision is returned and kept on the version.
pub fn judge(
    home: &LoopHome,
    base: &str,
    candidate: &str,
    pairs: &BTreeMap<String, Paired>,
) -> Result<Decision> {
    let decision = decide(pairs);
    update(home, base, |r| {
        let Some(index) = r.versions.iter().position(|v| v.id == candidate) else {
            bail!("no version {candidate}");
        };
        if r.versions[index].standing != Standing::Candidate {
            bail!(
                "{candidate} was already judged ({:?})",
                r.versions[index].standing
            );
        }
        r.versions[index].decision = Some(decision.clone());
        if decision.promoted {
            r.versions[index].standing = Standing::Promoted;
            if let Some(old) = r.current.replace(candidate.to_string()) {
                r.history.push(old);
            }
        } else {
            r.versions[index].standing = Standing::Rejected;
        }
        Ok(())
    })?;
    Ok(decision)
}

/// Returns to the version that was current before the present one, or to the
/// base model with no adapter when there was none.
pub fn rollback(home: &LoopHome, base: &str) -> Result<Option<String>> {
    update(home, base, |r| {
        if r.current.is_none() {
            bail!("no adapter is in use: nothing to roll back");
        }
        r.current = r.history.pop();
        Ok(r.current.clone())
    })
}

/// The model reference of what is in use: the base, with the current
/// adapter when one is promoted.
pub fn current_ref(registry: &Registry) -> Result<String> {
    let Some(id) = &registry.current else {
        return Ok(registry.base.clone());
    };
    let version = registry
        .versions
        .iter()
        .find(|v| &v.id == id)
        .with_context(|| format!("the current version {id} is not in the registry"))?;
    Ok(format!("{}+{}", registry.base, version.adapter.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(rows: &[(bool, bool)]) -> BTreeMap<String, Paired> {
        rows.iter()
            .enumerate()
            .map(|(i, (baseline, candidate))| {
                (
                    format!("t{i:02}"),
                    Paired {
                        baseline: *baseline,
                        candidate: *candidate,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn a_clear_gain_is_promoted_and_a_loss_or_a_tie_is_not() {
        let gain = pairs(
            &[(false, true); 8]
                .into_iter()
                .chain([(true, true), (false, false)])
                .collect::<Vec<_>>(),
        );
        assert!(decide(&gain).promoted);
        let loss = pairs(
            &[(true, false); 8]
                .into_iter()
                .chain([(true, true)])
                .collect::<Vec<_>>(),
        );
        assert!(!decide(&loss).promoted);
        let tie = pairs(&[(true, true); 10]);
        assert!(!decide(&tie).promoted);
    }

    #[test]
    fn too_few_tasks_or_a_gain_that_could_be_chance_is_not_promoted() {
        assert!(
            !decide(&pairs(&[(false, true); 5])).promoted,
            "five tasks decide nothing"
        );
        let marginal = pairs(&[
            (false, true),
            (false, true),
            (true, false),
            (true, true),
            (false, false),
            (true, true),
            (false, false),
            (true, true),
            (false, true),
        ]);
        let d = decide(&marginal);
        assert!(!d.promoted && d.reason.contains("p ="), "{}", d.reason);
    }

    #[test]
    fn promotion_keeps_what_it_replaced_and_rollback_returns_to_it() {
        let dir = tempfile::tempdir().unwrap();
        let home = LoopHome::under(dir.path());
        let reg = |home: &LoopHome| {
            register(
                home,
                "local:Qwen/Qwen3-8B",
                Path::new("a.brain"),
                "sha256:a",
                "sha256:d",
                Path::new("r.json"),
            )
            .unwrap()
        };
        let (v1, v2) = (reg(&home), reg(&home));
        let win = pairs(&[(false, true); 9]);
        assert!(
            judge(&home, "local:Qwen/Qwen3-8B", &v1, &win)
                .unwrap()
                .promoted
        );
        assert!(
            judge(&home, "local:Qwen/Qwen3-8B", &v2, &win)
                .unwrap()
                .promoted
        );
        let r = load(&home, "local:Qwen/Qwen3-8B").unwrap();
        assert_eq!(r.current.as_deref(), Some("v2"));
        assert_eq!(
            rollback(&home, "local:Qwen/Qwen3-8B").unwrap().as_deref(),
            Some("v1")
        );
        assert_eq!(rollback(&home, "local:Qwen/Qwen3-8B").unwrap(), None);
        let r = load(&home, "local:Qwen/Qwen3-8B").unwrap();
        assert_eq!(
            current_ref(&r).unwrap(),
            "local:Qwen/Qwen3-8B",
            "no adapter: the base"
        );
        assert!(rollback(&home, "local:Qwen/Qwen3-8B").is_err());
    }

    #[test]
    fn a_rejected_candidate_stays_on_record_and_the_current_model_does_not_change() {
        let dir = tempfile::tempdir().unwrap();
        let home = LoopHome::under(dir.path());
        let id = register(
            &home,
            "local:Qwen/Qwen3-8B",
            Path::new("a.brain"),
            "sha256:a",
            "sha256:d",
            Path::new("r.json"),
        )
        .unwrap();
        let none = pairs(&[(true, false); 9]);
        let d = judge(&home, "local:Qwen/Qwen3-8B", &id, &none).unwrap();
        assert!(!d.promoted);
        let r = load(&home, "local:Qwen/Qwen3-8B").unwrap();
        assert_eq!(r.current, None);
        assert_eq!(r.versions[0].standing, Standing::Rejected);
        assert_eq!(r.versions[0].decision.as_ref().unwrap().lost, 9);
        assert!(
            judge(&home, "local:Qwen/Qwen3-8B", &id, &none).is_err(),
            "a version is judged once"
        );
    }

    #[test]
    fn two_writers_cannot_interleave() {
        let dir = tempfile::tempdir().unwrap();
        let home = LoopHome::under(dir.path());
        let held = Locked::take(&home).unwrap();
        assert!(register(&home, "b", Path::new("a"), "x", "y", Path::new("r")).is_err());
        drop(held);
        assert!(register(&home, "b", Path::new("a"), "x", "y", Path::new("r")).is_ok());
    }
}
