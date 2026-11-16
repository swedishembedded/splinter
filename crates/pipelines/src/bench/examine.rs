// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that promote a model only on
// a signed verdict from an examiner the learner cannot reach, for its
// clients. If your team needs expertise in sealed evaluation of learning
// agents, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A paired verdict of a candidate against its predecessor from bench's
//! sealed workspace tool-use exam, both served by brain and worked through
//! sven. Bench keeps its jobs (item-level records) in its own data
//! directory; only the signed document of aggregates is written for the
//! caller.
//!
//! The learner never reads the exam: bench generates it from a seed file the
//! learner does not open, runs both arms through sven against the served
//! models, and issues one signed document of aggregates. The candidate is
//! promoted only when that document verifies under the signing key and its
//! verdict is Promote; an unsigned, altered or unverifiable document is
//! invalid evidence and promotes nothing.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context as _};
use serde::{Deserialize, Serialize};

use splinter_model::scratch::Served as Target;

/// What the examination concluded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Examined {
    /// The verdict as the examiner wrote it (`Promote`, `Reject`,
    /// `Inconclusive`, `Invalid`).
    pub verdict: String,
    /// True when the signed document verified.
    pub verified: bool,
    /// The document, kept in the run directory.
    pub document: PathBuf,
}

impl Examined {
    /// True only for a verified Promote.
    #[must_use]
    pub fn promotes(&self) -> bool {
        self.verified && self.verdict.eq_ignore_ascii_case("promote")
    }
}

/// Who examines a candidate.
pub trait Examiner {
    /// The paired verdict of `candidate` against `predecessor`; the signed
    /// document is written to `out`.
    ///
    /// # Errors
    /// The examination could not be run.
    fn examine(
        &self,
        candidate: &Target,
        predecessor: &Target,
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
    /// The private seed file the exam and anchor are generated from.
    pub seed_file: PathBuf,
    /// The verdict signing key.
    pub signing_key: PathBuf,
    /// The smallest gain a promotion must show.
    pub min_gain: f64,
    /// The serving build both arms declare.
    pub serving_build: String,
}

/// The exam collection and the anchor, inside bench's own data directory.
const EXAM: &str = ".bench-data/exams/workspace-tools";
const ANCHOR: &str = ".bench-data/exams/workspace-tools-anchor";

impl Bench {
    fn run(&self, what: &str, command: &mut Command) -> anyhow::Result<()> {
        let status = command
            .current_dir(&self.dir)
            .env("BENCH_SVEN_BUILD", &self.sven)
            .env("BRAIN_KEY", "minimind")
            .status()
            .with_context(|| format!("starting bench to {what}"))?;
        if !status.success() {
            bail!("bench could not {what} ({status})");
        }
        Ok(())
    }

    /// The arm profile for `target`, from bench's own base profile with the
    /// model and the serving build replaced, written outside bench's tree;
    /// the endpoint is given to `bench run` itself.
    fn profile(&self, arm: &str, target: &Target, dir: &Path) -> anyhow::Result<PathBuf> {
        let base = self.dir.join("profiles/workspace-tools-base.toml");
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
        let path = dir.join(format!("{arm}.toml"));
        std::fs::write(&path, toml::to_string(&profile)?)?;
        Ok(path)
    }
}

impl Examiner for Bench {
    fn examine(
        &self,
        candidate: &Target,
        predecessor: &Target,
        out: &Path,
    ) -> anyhow::Result<Examined> {
        let work = out
            .parent()
            .context("the verdict has a directory")?
            .join("bench");
        // The jobs hold item-level records, which stay with bench: they are
        // written into bench's own data directory, one directory per verdict.
        let tag = splinter_core::digest::Digest::of(out.to_string_lossy().as_bytes());
        let jobs_root = self
            .dir
            .join(".bench-data/jobs")
            .join(format!("splinter-{}", &tag.hex()[..16]));
        // An interrupted examination starts over: no partial job is reused.
        if jobs_root.exists() {
            std::fs::remove_dir_all(&jobs_root)
                .with_context(|| format!("clearing {}", jobs_root.display()))?;
        }
        std::fs::create_dir_all(&work)?;
        let seed = format!("WT_SEED_FILE={}", self.seed_file.display());
        for target in [
            "workspace-tools/exam/generate",
            "workspace-tools/anchor/generate",
        ] {
            self.run(
                "generate the exam",
                Command::new("make").args([target, &seed]),
            )?;
        }
        let mut jobs = Vec::new();
        for (arm, model) in [("base", predecessor), ("candidate", candidate)] {
            let profile = self.profile(arm, model, &work)?;
            for (suite, collection) in [("exam", EXAM), ("anchor", ANCHOR)] {
                let job = jobs_root.join(format!("{arm}-{suite}"));
                self.run(
                    &format!("run the {suite} for {arm}"),
                    Command::new("./bench")
                        .arg("run")
                        .arg("--profile")
                        .arg(&profile)
                        .arg("--endpoint")
                        .arg(&model.base_url)
                        .arg(collection)
                        .arg("--jobs-dir")
                        .arg(&job),
                )?;
                jobs.push((arm, job));
            }
        }
        let mut verdict = Command::new("./bench");
        verdict.arg("verdict");
        for (arm, job) in &jobs {
            verdict.arg(format!("--{arm}")).arg(job);
        }
        verdict
            .args([
                "--anchor",
                "workspace-tools-anchor",
                "--min-gain",
                &self.min_gain.to_string(),
                "--signing-key",
            ])
            .arg(&self.signing_key)
            .arg("--out")
            .arg(out);
        self.run("issue the verdict", &mut verdict)?;
        read_verdict(&self.dir, &self.signing_key, out)
    }
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
    Ok(Examined {
        verdict,
        verified,
        document: document.to_path_buf(),
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
}
