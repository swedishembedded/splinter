// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that hand a trained model to an
// independent examiner with everything needed to serve and audit it, for its
// clients. If your team needs expertise in separating the learning side of a
// model-improvement program from the side that measures it you can procure
// our services by sending an email to info@swedishembedded.com.

//! The one-way hand-off of a generation's champion to the examiner: a
//! candidate record (what to serve and what built it) and the list of every
//! task the generation trained or was gated on, so the examiner can audit them
//! for overlap with an exam the learner never sees. Nothing flows back except
//! the examiner's aggregate verdicts.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::Serialize;
use serde_json::{json, Value};
use splinter_sdk::vocabulary::digest::Digest;

use crate::family::TaskRecord;
use crate::reasoning::{GateDecoding, Reasoning};

/// What built the candidate: the commits and the binary, as the launch record
/// states them.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Build {
    /// Splinter's commit.
    pub splinter_commit: String,
    /// Sven's commit.
    pub sven_commit: String,
    /// Brain's commit.
    pub brain_commit: String,
    /// The SHA-256 of the executable that trained it.
    pub binary_digest: Option<String>,
}

/// One candidate: a generation's champion of one lineage.
#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    /// The series (the lineage).
    pub series: String,
    /// The step (the generation); 0 is the base, which has no candidate.
    pub step: u32,
    /// The base model.
    pub base: String,
    /// The digest of the base checkpoint ([`base_digest`]).
    pub base_digest: String,
    /// The adapter, absent for the base and for a generation that kept its
    /// predecessor's champion when that was the base.
    pub adapter: Option<PathBuf>,
    /// The adapter's SHA-256.
    pub adapter_digest: Option<String>,
    /// How to serve it.
    pub serving: Value,
    /// Whether the generation's arm produced a new champion.
    pub advanced: bool,
    /// How the generation's arm ended, as the controller's ledger says.
    pub arm_verdict: Option<String>,
    /// What built it.
    #[serde(flatten)]
    pub build: Build,
}

/// How a candidate is served: brain's plain serving with the adapter, the
/// reply read greedily with reasoning off.
#[must_use]
pub fn serving(base: &Path, adapter: Option<&Path>) -> Value {
    serving_for(base, adapter, None)
}

/// [`serving`] under the protocol the lineage ran: with `reasoning`, thinking
/// is on and the terms state the decoding of the exams the candidate was
/// gated on and the token budget of a reply.
#[must_use]
pub fn serving_for(base: &Path, adapter: Option<&Path>, reasoning: Option<&Reasoning>) -> Value {
    let mut flags = vec!["--checkpoint".to_string(), base.display().to_string()];
    if let Some(adapter) = adapter {
        flags.push("--adapter".into());
        flags.push(adapter.display().to_string());
        flags.push("--adapter-targets".into());
        flags.push("all".into());
    }
    let Some(r) = reasoning else {
        return json!({"flags": flags, "thinking": false, "decoding": "greedy"});
    };
    let mut terms = json!({
        "flags": flags,
        "thinking": true,
        "decoding": "greedy",
        "max_tokens": r.max_tokens,
    });
    if let GateDecoding::Sampled { seed } = r.gate {
        terms["decoding"] = json!("sampled");
        terms["temperature"] = json!(r.temperature);
        terms["top_p"] = json!(r.top_p);
        terms["top_k"] = json!(r.top_k);
        terms["seed"] = json!(seed);
    }
    terms
}

/// A digest of a checkpoint directory: the SHA-256 of the lines
/// `<file name> <sha256 of the file>` of its `*.safetensors` files in name
/// order.
///
/// # Errors
/// The directory cannot be read.
pub fn base_digest(model: &Path) -> Result<String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(model)
        .with_context(|| format!("reading {}", model.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    anyhow::ensure!(
        !files.is_empty(),
        "{} holds no safetensors file",
        model.display()
    );
    let mut lines = String::new();
    for file in files {
        let digest = Digest::sha256_of_reader(std::fs::File::open(&file)?)
            .with_context(|| format!("hashing {}", file.display()))?;
        let name = file.file_name().map(|n| n.to_string_lossy().into_owned());
        lines.push_str(&format!("{} {digest}\n", name.unwrap_or_default()));
    }
    Ok(Digest::sha256_of(lines.as_bytes()).to_string())
}

/// Copies `adapter` to `kept`/`lineage`/`generation`/adapter.safetensors, a
/// path nothing collects, and returns it with the copy's SHA-256. An existing
/// copy is kept as it is and must be the same bytes.
///
/// # Errors
/// The copy fails, or an earlier copy differs.
pub fn keep_adapter(
    kept: &Path,
    lineage: &str,
    generation: u32,
    adapter: &Path,
) -> Result<(PathBuf, String)> {
    let dir = kept.join(lineage).join(generation.to_string());
    let target = dir.join("adapter.safetensors");
    let digest = |p: &Path| -> Result<String> {
        Ok(Digest::sha256_of_reader(std::fs::File::open(p)?)?.to_string())
    };
    if target.exists() {
        anyhow::ensure!(
            digest(&target)? == digest(adapter)?,
            "{} is already kept and differs from {}",
            target.display(),
            adapter.display()
        );
    } else {
        std::fs::create_dir_all(&dir)?;
        let partial = dir.join("adapter.safetensors.partial");
        std::fs::copy(adapter, &partial)
            .with_context(|| format!("copying {}", adapter.display()))?;
        std::fs::rename(&partial, &target)?;
    }
    let sum = digest(&target)?;
    Ok((target, sum))
}

/// Writes `tasks` and then `candidate` under `root`/candidates/`series`/`step`
/// as `training-tasks.jsonl` and `candidate.json`. The examiner starts on a
/// candidate the moment its `candidate.json` appears, so that file comes
/// last and appears whole. Returns false, and writes nothing, when the
/// candidate was already handed over.
///
/// # Errors
/// The files cannot be written.
pub fn write(root: &Path, candidate: &Candidate, tasks: &[TaskRecord]) -> Result<bool> {
    let dir = root
        .join("candidates")
        .join(&candidate.series)
        .join(candidate.step.to_string());
    let target = dir.join("candidate.json");
    if target.exists() {
        return Ok(false);
    }
    std::fs::create_dir_all(&dir)?;
    let mut file = std::fs::File::create(dir.join("training-tasks.jsonl"))?;
    for task in tasks {
        writeln!(file, "{}", serde_json::to_string(task)?)?;
    }
    file.sync_all()?;
    let partial = dir.join(".candidate.json.partial");
    std::fs::write(&partial, serde_json::to_vec_pretty(candidate)?)?;
    std::fs::rename(&partial, &target)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reasoning(gate: GateDecoding) -> Reasoning {
        Reasoning {
            max_tokens: 4096,
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            gate,
        }
    }

    #[test]
    fn the_terms_without_reasoning_are_the_ones_every_earlier_candidate_carried() {
        let terms = serving_for(Path::new("/m"), None, None);
        assert_eq!(
            terms.to_string(),
            r#"{"decoding":"greedy","flags":["--checkpoint","/m"],"thinking":false}"#
        );
        assert_eq!(terms, serving(Path::new("/m"), None));
    }

    #[test]
    fn the_terms_of_a_reasoning_lineage_state_thinking_and_the_decoding_it_was_gated_with() {
        let greedy = serving_for(
            Path::new("/m"),
            None,
            Some(&reasoning(GateDecoding::Greedy)),
        );
        assert_eq!(
            (
                &greedy["thinking"],
                &greedy["decoding"],
                &greedy["max_tokens"]
            ),
            (&json!(true), &json!("greedy"), &json!(4096))
        );
        assert!(greedy.get("seed").is_none());
        let sampled = serving_for(
            Path::new("/m"),
            None,
            // seed: it is the value the hand-off must state back verbatim
            Some(&reasoning(GateDecoding::Sampled { seed: 7 })),
        );
        assert_eq!(sampled["decoding"], "sampled");
        assert_eq!(
            (&sampled["seed"], &sampled["top_p"], &sampled["top_k"]),
            (&json!(7), &json!(0.95f32), &json!(20))
        );
    }

    fn candidate(generation: u32) -> Candidate {
        Candidate {
            series: "r0".into(),
            step: generation,
            base: "Qwen/Qwen3-4B".into(),
            base_digest: "sha256:aa".into(),
            adapter: None,
            adapter_digest: None,
            serving: serving(Path::new("/m"), None),
            advanced: false,
            arm_verdict: None,
            build: Build::default(),
        }
    }

    #[test]
    fn a_candidate_appears_whole_and_is_never_rewritten() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskRecord {
            id: "t".into(),
            role: "train",
            origin: "generator".into(),
            family: None,
            program: "print(1)".into(),
            reference: "1".into(),
            digest: "sha256:bb".into(),
        };
        assert!(write(dir.path(), &candidate(0), &[task]).unwrap());
        let at = dir.path().join("candidates/r0/0");
        let json: Value =
            serde_json::from_slice(&std::fs::read(at.join("candidate.json")).unwrap()).unwrap();
        assert_eq!(json["series"], "r0");
        assert_eq!(json["step"], 0);
        assert_eq!(json["base"], "Qwen/Qwen3-4B");
        assert!(json["adapter"].is_null());
        assert_eq!(json["serving"]["flags"][0], "--checkpoint");
        let tasks = std::fs::read_to_string(at.join("training-tasks.jsonl")).unwrap();
        assert_eq!(tasks.lines().count(), 1);
        let again = write(dir.path(), &candidate(0), &[]).unwrap();
        assert!(!again, "a candidate already handed over is left as it is");
        assert_eq!(
            std::fs::read_to_string(at.join("training-tasks.jsonl")).unwrap(),
            tasks
        );
    }

    #[test]
    fn a_kept_adapter_is_a_copy_that_a_later_collection_cannot_remove() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("adapter.safetensors");
        std::fs::write(&source, b"weights").unwrap();
        let (kept, digest) = keep_adapter(&dir.path().join("kept"), "r1", 2, &source).unwrap();
        std::fs::remove_file(&source).unwrap();
        assert_eq!(std::fs::read(&kept).unwrap(), b"weights");
        assert_eq!(digest, Digest::sha256_of(b"weights").to_string());
        std::fs::write(&source, b"other").unwrap();
        assert!(keep_adapter(&dir.path().join("kept"), "r1", 2, &source).is_err());
    }

    #[test]
    fn the_base_digest_follows_the_files_and_their_names() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.safetensors"), b"one").unwrap();
        let first = base_digest(dir.path()).unwrap();
        std::fs::write(dir.path().join("b.safetensors"), b"two").unwrap();
        assert_ne!(base_digest(dir.path()).unwrap(), first);
        assert!(base_digest(&dir.path().join("none")).is_err());
    }
}
