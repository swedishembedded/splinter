// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that promote a model only on
// a signed verdict from an examiner the learner cannot reach, for its
// clients. If your team needs expertise in sealed evaluation of learning
// agents, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Bench's sealed workspace tool-use exam: a checkpoint served by brain
//! *sits* the exam and its anchor through sven, once, and any two sittings
//! are paired into a signed verdict of aggregates.
//!
//! The learner never reads the exam: bench generates it from seed files the
//! learner does not open, runs every sitting itself, keeps the jobs (which
//! hold item-level records) in its own data directory, and issues one signed
//! document of aggregates per pair. What crosses back is that document: the
//! verdict, and per holdout tier the items and the pass rates of both sides.
//! The candidate is promoted only when the document verifies under the
//! signing key and its verdict is Promote; an unsigned, altered or
//! unverifiable document is invalid evidence and promotes nothing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context as _};
use serde::{Deserialize, Serialize};

use splinter_model::scratch::Served as Target;

/// One tier's aggregates, as the verdict document states them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TierRates {
    /// Exam items in the tier.
    pub items: u64,
    /// The base side's pass rate.
    pub base: f64,
    /// The candidate side's pass rate.
    pub candidate: f64,
}

/// What a paired examination concluded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Examined {
    /// The verdict as the examiner wrote it (`Promote`, `Reject`,
    /// `Inconclusive`, `Invalid`).
    pub verdict: String,
    /// True when the signed document verified.
    pub verified: bool,
    /// The document, kept in the run directory.
    pub document: PathBuf,
    /// Per holdout tier (`a`, `b`, `c`) and for the anchor (`anchor`), the
    /// pass rates of both sides.
    pub tiers: BTreeMap<String, TierRates>,
    /// Why, in the examiner's words (aggregates only).
    pub reasons: Vec<String>,
}

impl Examined {
    /// True only for a verified Promote.
    #[must_use]
    pub fn promotes(&self) -> bool {
        self.verified && self.verdict.eq_ignore_ascii_case("promote")
    }
}

/// One sitting of the exam and its anchor: where the examiner keeps its
/// jobs. Only the examiner reads them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sitting {
    /// The exam job.
    pub exam: PathBuf,
    /// The anchor job.
    pub anchor: PathBuf,
}

impl Sitting {
    /// True while both jobs are still where the examiner left them.
    #[must_use]
    pub fn present(&self) -> bool {
        self.exam.is_dir() && self.anchor.is_dir()
    }
}

/// Who examines checkpoints.
pub trait Examiner {
    /// `target` sits the exam and its anchor, as the sitting named `name`
    /// (a name the caller keeps stable for the same checkpoint; an
    /// interrupted sitting of that name starts over). A `control` sitting is
    /// a spurious-reward control arm: it must never be promoted.
    ///
    /// # Errors
    /// The exam could not be generated or run.
    fn sit(&self, target: &Target, name: &str, control: bool) -> anyhow::Result<Sitting>;

    /// The paired verdict of `candidate` against `base`, with an optional
    /// spurious-reward `control` arm; the signed document is written to
    /// `out`.
    ///
    /// # Errors
    /// The verdict could not be issued.
    fn verdict(
        &self,
        base: &Sitting,
        candidate: &Sitting,
        control: Option<&Sitting>,
        out: &Path,
    ) -> anyhow::Result<Examined>;
}

/// Bench, run from its checkout.
#[derive(Clone, Debug)]
pub struct Bench {
    /// The bench checkout.
    pub dir: PathBuf,
    /// The sven binary the exam runs the agent with.
    pub sven: PathBuf,
    /// The private seed file the exam is generated from (drawn into it when
    /// missing).
    pub seed_file: PathBuf,
    /// The anchor's own private seed file.
    pub anchor_seed_file: PathBuf,
    /// The verdict signing key.
    pub signing_key: PathBuf,
    /// The smallest gain a promotion must show.
    pub min_gain: f64,
    /// The serving build every arm declares.
    pub serving_build: String,
    /// Episodes a sitting runs at once.
    pub concurrency: u32,
}

/// The exam collection and the anchor, inside bench's own data directory.
const EXAM: &str = ".bench-data/exams/workspace-tools";
const ANCHOR: &str = ".bench-data/exams/workspace-tools-anchor";

/// The anchor collection as `bench verdict --anchor` names it.
const ANCHOR_NAME: &str = "workspace-tools-anchor";

impl Bench {
    fn run(&self, what: &str, key: &str, command: &mut Command) -> anyhow::Result<()> {
        let status = command
            .current_dir(&self.dir)
            .env("BENCH_SVEN_BUILD", &self.sven)
            .env("BRAIN_KEY", key)
            .status()
            .with_context(|| format!("starting bench to {what}"))?;
        if !status.success() {
            bail!("bench could not {what} ({status})");
        }
        Ok(())
    }

    /// The arm profile for `target`, from bench's own profile (the control
    /// profile for a control arm) with the model, the serving build and the
    /// concurrency replaced, written beside the jobs; the endpoint is given
    /// to `bench run` itself.
    fn profile(&self, target: &Target, control: bool, path: &Path) -> anyhow::Result<()> {
        let arm = if control { "control" } else { "base" };
        let base = self
            .dir
            .join(format!("profiles/workspace-tools-{arm}.toml"));
        let text = std::fs::read_to_string(&base)
            .with_context(|| format!("reading {}", base.display()))?;
        let mut profile: toml::Table =
            toml::from_str(&text).with_context(|| format!("parsing {}", base.display()))?;
        let brain = profile
            .get_mut("upstreams")
            .and_then(|u| u.get_mut("brain"))
            .and_then(toml::Value::as_table_mut)
            .context("bench's profile has no [upstreams.brain]")?;
        brain.insert("model".into(), target.model.clone().into());
        let run = profile
            .get_mut("run")
            .and_then(toml::Value::as_table_mut)
            .context("bench's profile has no [run]")?;
        run.insert("serving_build".into(), self.serving_build.clone().into());
        run.insert("concurrency".into(), i64::from(self.concurrency).into());
        std::fs::write(path, toml::to_string(&profile)?)?;
        Ok(())
    }

    /// Generates the exam and the anchor from their seed files; bench admits
    /// the same collection for the same seed, so this is safe to repeat.
    fn generate(&self) -> anyhow::Result<()> {
        for (target, seed) in [
            ("workspace-tools/exam/generate", &self.seed_file),
            ("workspace-tools/anchor/generate", &self.anchor_seed_file),
        ] {
            let seed = format!("WT_SEED_FILE={}", seed.display());
            self.run(
                "generate the exam",
                "",
                Command::new("make").args([target, &seed]),
            )?;
        }
        Ok(())
    }
}

impl Examiner for Bench {
    fn sit(&self, target: &Target, name: &str, control: bool) -> anyhow::Result<Sitting> {
        // The jobs hold item-level records, which stay with bench: they are
        // written into bench's own data directory, one directory per sitting.
        let root = self
            .dir
            .join(".bench-data/jobs")
            .join(format!("splinter-{name}"));
        // An interrupted sitting starts over: no partial job is reused.
        if root.exists() {
            std::fs::remove_dir_all(&root)
                .with_context(|| format!("clearing {}", root.display()))?;
        }
        std::fs::create_dir_all(&root)?;
        self.generate()?;
        let profile = root.join("profile.toml");
        self.profile(target, control, &profile)?;
        let sitting = Sitting {
            exam: root.join("exam"),
            anchor: root.join("anchor"),
        };
        for (suite, collection, job) in [
            ("exam", EXAM, &sitting.exam),
            ("anchor", ANCHOR, &sitting.anchor),
        ] {
            self.run(
                &format!("run the {suite} for {}", target.model),
                &target.key,
                Command::new("./bench")
                    .arg("run")
                    .arg("--profile")
                    .arg(&profile)
                    .arg("--endpoint")
                    .arg(&target.base_url)
                    .arg(collection)
                    .arg("--jobs-dir")
                    .arg(job),
            )?;
        }
        Ok(sitting)
    }

    fn verdict(
        &self,
        base: &Sitting,
        candidate: &Sitting,
        control: Option<&Sitting>,
        out: &Path,
    ) -> anyhow::Result<Examined> {
        for sitting in [Some(base), Some(candidate), control].into_iter().flatten() {
            if !sitting.present() {
                bail!(
                    "bench no longer holds the jobs of a sitting ({}): sit the exam again",
                    sitting.exam.display()
                );
            }
        }
        let mut verdict = Command::new("./bench");
        verdict.arg("verdict");
        for (side, sitting) in [("base", base), ("candidate", candidate)] {
            verdict
                .arg(format!("--{side}"))
                .arg(&sitting.exam)
                .arg(format!("--{side}"))
                .arg(&sitting.anchor);
        }
        if let Some(control) = control {
            verdict.arg("--control").arg(&control.exam);
        }
        verdict
            .args([
                "--anchor",
                ANCHOR_NAME,
                "--min-gain",
                &self.min_gain.to_string(),
                "--signing-key",
            ])
            .arg(&self.signing_key)
            .arg("--out")
            .arg(out);
        self.run("issue the verdict", "", &mut verdict)?;
        read_verdict(&self.dir, &self.signing_key, out)
    }
}

/// The aggregates of one section of a verdict document.
fn rates(section: &serde_json::Value) -> Option<TierRates> {
    Some(TierRates {
        items: section["items"].as_u64()?,
        base: section["base_pass_rate"].as_f64()?,
        candidate: section["candidate_pass_rate"].as_f64()?,
    })
}

/// The verdict in `document`, verified by bench under `signing_key`.
///
/// # Errors
/// The document cannot be read or names no verdict.
pub fn read_verdict(bench: &Path, signing_key: &Path, document: &Path) -> anyhow::Result<Examined> {
    let verified = Command::new("./bench")
        .current_dir(bench)
        .arg("verify-verdict")
        .arg(document)
        .arg("--signing-key")
        .arg(signing_key)
        .status()
        .is_ok_and(|s| s.success());
    let text = std::fs::read_to_string(document)
        .with_context(|| format!("reading {}", document.display()))?;
    let doc: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", document.display()))?;
    let verdict = doc["verdict"]
        .as_str()
        .context("the verdict document names no verdict")?
        .to_string();
    let mut tiers: BTreeMap<String, TierRates> = doc["tiers"]
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(tier, section)| Some((tier.clone(), rates(section)?)))
        .collect();
    if let Some(anchor) = rates(&doc["retention"]) {
        tiers.insert("anchor".into(), anchor);
    }
    let reasons = doc["reasons"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|r| r.as_str().map(String::from))
        .collect();
    Ok(Examined {
        verdict,
        verified,
        document: document.to_path_buf(),
        tiers,
        reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_verified_promote_promotes() {
        let doc = PathBuf::from("verdict.json");
        let e = |verdict: &str, verified| Examined {
            verdict: verdict.into(),
            verified,
            document: doc.clone(),
            tiers: BTreeMap::new(),
            reasons: Vec::new(),
        };
        assert!(e("Promote", true).promotes());
        assert!(
            !e("Promote", false).promotes(),
            "unverifiable evidence never promotes"
        );
        assert!(!e("Inconclusive", true).promotes());
        assert!(!e("Reject", true).promotes());
    }

    /// A bench checkout whose `verify-verdict` accepts only documents
    /// containing `"signature":"good"`.
    fn fake_bench(dir: &Path) {
        use std::os::unix::fs::PermissionsExt;
        let script =
            "#!/bin/sh\n[ \"$1\" = verify-verdict ] && grep -q '\"signature\":\"good\"' \"$2\"\n";
        std::fs::write(dir.join("bench"), script).unwrap();
        std::fs::set_permissions(dir.join("bench"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
    }

    #[test]
    fn a_forged_or_unsigned_verdict_does_not_verify() {
        let dir = tempfile::tempdir().unwrap();
        fake_bench(dir.path());
        let key = dir.path().join("key");
        let doc = dir.path().join("verdict.json");
        std::fs::write(&doc, r#"{"verdict":"Promote","signature":"good"}"#).unwrap();
        assert!(read_verdict(dir.path(), &key, &doc).unwrap().promotes());
        std::fs::write(&doc, r#"{"verdict":"Promote","signature":"forged"}"#).unwrap();
        let forged = read_verdict(dir.path(), &key, &doc).unwrap();
        assert!(!forged.verified && !forged.promotes());
        std::fs::write(&doc, r#"{"signature":"good"}"#).unwrap();
        assert!(
            read_verdict(dir.path(), &key, &doc).is_err(),
            "a document without a verdict is refused"
        );
    }

    #[test]
    fn the_tier_aggregates_and_the_anchor_are_read_from_the_document() {
        let dir = tempfile::tempdir().unwrap();
        fake_bench(dir.path());
        let doc = dir.path().join("verdict.json");
        std::fs::write(
            &doc,
            r#"{"verdict":"Inconclusive","signature":"good","reasons":["MDE exceeds min_gain"],
                "tiers":{"a":{"items":15,"base_pass_rate":0.2,"candidate_pass_rate":0.4},"b":{"items":0,"families":0}},
                "retention":{"items":20,"base_pass_rate":0.1,"candidate_pass_rate":0.15}}"#,
        )
        .unwrap();
        let e = read_verdict(dir.path(), &dir.path().join("key"), &doc).unwrap();
        assert_eq!(
            e.tiers["a"],
            TierRates {
                items: 15,
                base: 0.2,
                candidate: 0.4
            }
        );
        assert!(
            !e.tiers.contains_key("b"),
            "a tier no item was run in has no rates"
        );
        assert_eq!(e.tiers["anchor"].items, 20);
        assert_eq!(e.reasons, ["MDE exceeds min_gain"]);
    }
}
