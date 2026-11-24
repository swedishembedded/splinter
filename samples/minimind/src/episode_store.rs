// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that keep thousands of
// agent episodes as a few verifiable files on shared storage, for its
// clients. If your team needs expertise in durable experiment records, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Where a stage's episodes live: their tree in the run's scratch while the
//! stage runs (the run directory holds a link to it), each finished episode
//! appended to the stage's one pack at once, and the pack rewritten in path
//! order when the stage is done ([`crate::pack`]). Readers take an episode's
//! files from the tree while it exists and from the pack after it.

use crate::ledger::Commit;
use anyhow::{bail, Context as _};
use splinter_sdk::vocabulary::digest::Digest;

use crate::stages::Run;

/// The files of one stage's episodes, from their tree while it exists and
/// from the stage's pack after it, each checked against the digest its
/// episode committed.
pub struct EpisodeReader {
    stage: String,
    tree: std::path::PathBuf,
    pack: Option<crate::pack::Pack>,
}

impl EpisodeReader {
    /// Output `name` (`records`, `outcome`) of the committed `episode`.
    ///
    /// # Errors
    /// It is in neither the tree nor the pack, or no longer has its digest.
    pub fn read(&self, episode: &Commit, name: &str) -> anyhow::Result<String> {
        let file = match name {
            "records" => "records.jsonl",
            "outcome" => "outcome.json",
            other => bail!("an episode has no output {other:?}"),
        };
        let task = episode
            .unit
            .strip_prefix(&format!("{}/", self.stage))
            .with_context(|| format!("{} is not an episode of {}", episode.unit, self.stage))?;
        let path = self.tree.join(task).join(file);
        let text = if path.exists() {
            std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?
        } else {
            self.pack
                .as_ref()
                .and_then(|p| p.read(&format!("{task}/{file}")))
                .with_context(|| {
                    format!("{} has no {file}, in its tree or its pack", episode.unit)
                })?
                .to_string()
        };
        let want = episode
            .output(name)
            .with_context(|| format!("{} committed no {name}", episode.unit))?;
        if &Digest::of(text.as_bytes()) != want {
            bail!(
                "{}'s {file} no longer has the digest it committed",
                episode.unit
            );
        }
        Ok(text)
    }
}

impl Run<'_> {
    /// The tree of `stage`'s episodes and the pack it becomes.
    pub(crate) fn episode_tree(&self, stage: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        (
            self.path(&format!("episodes/{stage}")),
            self.path(&format!("episodes/{stage}.jsonl")),
        )
    }

    /// A reader of `stage`'s episode files.
    ///
    /// # Errors
    /// The stage's pack cannot be read.
    pub fn episode_reader(&self, stage: &str) -> anyhow::Result<EpisodeReader> {
        let (tree, packed) = self.episode_tree(stage);
        let pack = if packed.exists() {
            Some(crate::pack::Pack::open(&packed)?)
        } else {
            None
        };
        Ok(EpisodeReader {
            stage: stage.to_string(),
            tree,
            pack,
        })
    }

    /// Points `stage`'s episode tree in the run directory at a directory in
    /// scratch, so its files never land on the run's storage.
    pub(crate) fn link_episode_tree(&self, stage: &str) -> anyhow::Result<()> {
        let (tree, _) = self.episode_tree(stage);
        let target = self.scratch.path(&format!("episodes/{stage}"));
        std::fs::create_dir_all(&target)?;
        match std::fs::symlink_metadata(&tree) {
            Ok(m) if m.file_type().is_symlink() => std::fs::remove_file(&tree)?,
            // A tree an earlier version left in place is packed with the rest.
            Ok(_) => return Ok(()),
            Err(_) => {}
        }
        if let Some(parent) = tree.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::os::unix::fs::symlink(&target, &tree)
            .with_context(|| format!("linking {}", tree.display()))
    }

    /// Packs `stage`'s episode tree into one file, once every episode of it
    /// committed, and removes the tree; returns the pack's path in the run
    /// directory.
    pub(crate) fn pack_episodes(&self, stage: &str) -> anyhow::Result<String> {
        let (tree, packed) = self.episode_tree(stage);
        let linked = std::fs::symlink_metadata(&tree).is_ok_and(|m| m.file_type().is_symlink());
        if linked {
            let target = std::fs::read_link(&tree)?;
            if target.exists() {
                crate::pack::pack_tree(&target, &packed)?;
            }
            std::fs::remove_file(&tree)?;
        } else if tree.exists() {
            crate::pack::pack_tree(&tree, &packed)?;
        } else if packed.exists() {
            crate::pack::canonical(&packed)?;
        }
        Ok(format!("episodes/{stage}.jsonl"))
    }
}
