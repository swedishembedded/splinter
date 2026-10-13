// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a synthetic corpus is kept: a directory outside the repository, so
//! nothing generated is ever checked in. An author is drawn again from its
//! seed; the articles, what was turned away and what was spent live here.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::gates::Article;
use crate::persona::Persona;

/// One kept article and what it was told from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredArticle {
    /// Its place in the corpus.
    pub index: usize,
    /// The topic it is about, an index into the topics.
    pub topic: usize,
    /// Its title.
    pub title: String,
    /// Its text.
    pub body: String,
    /// The cases it tells, by name.
    pub cases: Vec<String>,
    /// Which attempt at it was kept, from one.
    pub attempts: usize,
}

/// What a corpus has cost so far, estimated from text length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Requests put to the writer.
    pub calls: usize,
    /// Tokens shown to it, at four characters to the token.
    pub input_tokens: u64,
    /// Tokens it wrote, at four characters to the token.
    pub output_tokens: u64,
}

/// A corpus directory.
pub struct Cache {
    dir: PathBuf,
}

impl Cache {
    /// The corpus in `dir`, created if it is not there.
    ///
    /// # Errors
    /// The directory cannot be made.
    pub fn open(dir: PathBuf) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir.join("articles"))
            .with_context(|| format!("creating {}", dir.display()))?;
        Ok(Self { dir })
    }

    /// The directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn article_path(&self, index: usize) -> PathBuf {
        self.dir.join("articles").join(format!("{index:04}.json"))
    }

    /// Writes `bytes` to `path` whole or not at all.
    fn write_whole(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))
    }

    /// Records the author the corpus is written for.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn write_persona(&self, persona: &Persona) -> anyhow::Result<()> {
        let beliefs: Vec<_> = persona
            .beliefs
            .iter()
            .map(|b| serde_json::json!({"tenet": b.tenet, "polarity": b.polarity, "statement": b.statement()}))
            .collect();
        let doc =
            serde_json::json!({"seed": persona.seed, "name": persona.name, "beliefs": beliefs});
        Self::write_whole(
            &self.dir.join("persona.json"),
            &serde_json::to_vec_pretty(&doc)?,
        )
    }

    /// Whether article `index` is already kept.
    #[must_use]
    pub fn has_article(&self, index: usize) -> bool {
        self.article_path(index).exists()
    }

    /// Keeps `article`.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn write_article(&self, article: &StoredArticle) -> anyhow::Result<()> {
        Self::write_whole(
            &self.article_path(article.index),
            &serde_json::to_vec_pretty(article)?,
        )
    }

    /// Every kept article, in order.
    ///
    /// # Errors
    /// A file cannot be read or is not an article.
    pub fn articles(&self) -> anyhow::Result<Vec<StoredArticle>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(self.dir.join("articles"))? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "json") {
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("reading {}", path.display()))?;
                out.push(
                    serde_json::from_str(&text)
                        .with_context(|| format!("parsing {}", path.display()))?,
                );
            }
        }
        out.sort_by_key(|a: &StoredArticle| a.index);
        Ok(out)
    }

    /// Notes an attempt that was turned away, with the draft when there is
    /// one, so that a gate can be judged against what it turned away.
    ///
    /// # Errors
    /// The file cannot be appended to.
    pub fn note_rejected(
        &self,
        index: usize,
        attempt: usize,
        reason: &str,
        draft: Option<&Article>,
    ) -> anyhow::Result<()> {
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("rejected.jsonl"))?;
        writeln!(
            file,
            "{}",
            serde_json::json!({"article": index, "attempt": attempt, "reason": reason, "draft": draft})
        )?;
        Ok(())
    }

    /// What the corpus has cost so far.
    #[must_use]
    pub fn usage(&self) -> Usage {
        std::fs::read_to_string(self.dir.join("usage.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    /// Records what the corpus has cost.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn write_usage(&self, usage: &Usage) -> anyhow::Result<()> {
        Self::write_whole(
            &self.dir.join("usage.json"),
            &serde_json::to_vec_pretty(usage)?,
        )
    }
}
