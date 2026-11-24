// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distillation pipelines in which a larger
// model's verified agent episodes teach a small one, for its clients. If your
// team needs expertise in training small tool-using models from a teacher,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! A teacher model: a checkpoint brain serves (`--teacher DIR`) works the
//! training-family instances of the `teach` split through sven exactly as
//! the student is examined, sampled several times each; the episodes the
//! verifier accepts and admission keeps join the scripted expert's
//! demonstrations as the fine-tune's data. Its instances come from this
//! sample's own generator, never from the examination.

use std::path::Path;

use crate::ledger::{Commit, StageWork};
use anyhow::Context as _;
use serde_json::{json, Value};
use splinter_sdk::model::scratch::Checkpoint;
use splinter_sdk::vocabulary::digest::Digest;

use crate::collect::Actor;
use crate::recipe;
use crate::stages::{write, Run};
use crate::tasks::Split;

/// A teacher brain serves, identified by the digest of its files.
#[derive(Clone, Debug)]
pub struct Teacher {
    /// How it is served.
    pub checkpoint: Checkpoint,
    /// The digest of its files: another teacher is other episodes.
    pub digest: Digest,
}

impl Teacher {
    /// The checkpoint directory `dir`, made servable by linking it into the
    /// brain model store `models` as `local/teacher-<its name>`.
    ///
    /// # Errors
    /// The directory cannot be read or linked.
    pub fn install(dir: &Path, models: &Path) -> anyhow::Result<Self> {
        let dir =
            std::fs::canonicalize(dir).with_context(|| format!("reading {}", dir.display()))?;
        let base = dir
            .file_name()
            .context("the teacher directory has no name")?
            .to_string_lossy()
            .to_lowercase();
        let name: String = format!("teacher-{base}")
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        // The digest of every file, in name order.
        let mut files: Vec<_> = std::fs::read_dir(&dir)?
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
            .map(|e| e.path())
            .collect();
        files.sort();
        let mut parts = Vec::new();
        for f in &files {
            let file =
                std::fs::File::open(f).with_context(|| format!("reading {}", f.display()))?;
            let d = Digest::of_reader(file).with_context(|| format!("reading {}", f.display()))?;
            parts.push(format!(
                "{} {d}",
                f.file_name().unwrap_or_default().to_string_lossy()
            ));
        }
        let digest = Digest::of(parts.join("\n").as_bytes());
        let link = models.join("local").join(&name);
        std::fs::create_dir_all(models.join("local"))?;
        match std::fs::read_link(&link) {
            Ok(target) if target == dir => {}
            Ok(_) => {
                std::fs::remove_file(&link)?;
                std::os::unix::fs::symlink(&dir, &link)?;
            }
            Err(_) => std::os::unix::fs::symlink(&dir, &link)
                .with_context(|| format!("linking {} into {}", dir.display(), models.display()))?,
        }
        Ok(Self {
            checkpoint: Checkpoint {
                model: format!("local/{name}"),
                dir,
            },
            digest,
        })
    }
}

/// The path of output `name` of `commit`.
///
/// # Errors
/// It has none.
pub fn output_path(commit: &Commit, name: &str) -> anyhow::Result<String> {
    commit
        .outputs
        .iter()
        .find(|o| o.name == name)
        .map(|o| o.path.clone())
        .with_context(|| format!("{} has no {name}", commit.unit))
}

/// Bytes of a record's JSON per token of the student, at the least, in
/// tenths: a record longer than the chat block at this rate is left out
/// rather than refused by the trainer. The teacher's context is counted in
/// its own tokens, so its episodes can run past the student's.
const TENTHS_OF_A_BYTE_PER_TOKEN: usize = 25;

impl Run<'_> {
    /// The teacher's episodes on the `teach` split, gathered as
    /// `data/teach.jsonl`; `None` when the configuration asks for none.
    ///
    /// # Errors
    /// No teacher was given, an episode cannot be run, or none is accepted.
    pub fn teach(&mut self) -> anyhow::Result<Option<Commit>> {
        if self.config.tasks.teach == 0 {
            return Ok(None);
        }
        let teacher = self.teacher.clone().context(
            "tasks.teach asks for teacher episodes: give --teacher DIR, a checkpoint brain serves",
        )?;
        let actor = Actor::Model {
            checkpoint: teacher.checkpoint,
            digest: teacher.digest,
        };
        let how = self.config.teach.clone();
        Ok(Some(
            self.gather("teach", Split::Teach, &actor, true, Some(&how))?
                .0,
        ))
    }

    /// The fine-tune's data, `data/sft.jsonl`: the expert's demonstrations
    /// and the teacher's accepted episodes, records too long for the chat
    /// block left out and counted.
    ///
    /// # Errors
    /// A record file cannot be read or the data written.
    pub fn sft_data(
        &mut self,
        demonstrations: &Commit,
        taught: Option<&Commit>,
    ) -> anyhow::Result<Commit> {
        let sources: Vec<Commit> = std::iter::once(demonstrations.clone())
            .chain(taught.cloned())
            .collect();
        let mut inputs = Vec::new();
        for c in &sources {
            inputs.push((
                c.unit.clone(),
                c.output("records")
                    .context("a source has no records")?
                    .clone(),
            ));
        }
        let refs: Vec<(&str, &Digest)> = inputs.iter().map(|(n, d)| (n.as_str(), d)).collect();
        let block = self.config.sft.block;
        let key = self.key(
            &json!({ "block": block }),
            "sft-data",
            &refs,
            recipe::GATHER,
            &[],
        );
        self.unit("sft-data", &key, move |run, _| {
            let limit =
                usize::try_from(block).unwrap_or(usize::MAX) * TENTHS_OF_A_BYTE_PER_TOKEN / 10;
            let mut text = String::new();
            let mut counts = serde_json::Map::new();
            for c in &sources {
                let path = run.path(&output_path(c, "records")?);
                let records = std::fs::read_to_string(&path)
                    .with_context(|| format!("reading {}", path.display()))?;
                let (mut kept, mut long) = (0u64, 0u64);
                for line in records.lines() {
                    if line.len() > limit {
                        long += 1;
                    } else {
                        text.push_str(line);
                        text.push('\n');
                        kept += 1;
                    }
                }
                counts.insert(c.unit.clone(), json!({ "records": kept, "too_long": long }));
            }
            write(&run.path("data/sft.jsonl"), &text)?;
            Ok(StageWork::new(Value::Object(counts)).output("records", "data/sft.jsonl"))
        })
    }
}
