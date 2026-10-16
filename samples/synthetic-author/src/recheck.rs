// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Judging the drafts a corpus turned away by the gates as they are now, so
//! that a change to a gate can be measured against what it once refused.

use anyhow::Context as _;

use crate::episode::Episode;
use crate::gates::{check, Article, Rejection};
use crate::persona::Persona;

/// A saved draft judged again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rechecked {
    /// The article the draft was written for.
    pub article: usize,
    /// The attempt at it, as the log has it.
    pub attempt: serde_json::Value,
    /// Why the log says it was turned away.
    pub was: String,
    /// What the gates make of it now.
    pub now: Result<(), Rejection>,
}

/// Judges again every draft kept in the log of attempts `log` (the text of
/// `rejected.jsonl`), against the episodes `planned` for its article.
///
/// # Errors
/// A line is not a record of an attempt, has no article index, names an article
/// that is not planned, or holds a draft that is not an article; the error
/// names the line.
pub fn recheck(
    log: &str,
    persona: &Persona,
    planned: &[(usize, Vec<Episode>)],
    min_words: usize,
) -> anyhow::Result<Vec<Rechecked>> {
    let mut out = Vec::new();
    for (n, line) in log.lines().enumerate() {
        let line_no = n + 1;
        let row: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("rejected.jsonl line {line_no} is not JSON"))?;
        let article = row["article"]
            .as_u64()
            .and_then(|i| usize::try_from(i).ok())
            .with_context(|| format!("rejected.jsonl line {line_no} has no article index"))?;
        let Some(draft) = row.get("draft").filter(|d| !d.is_null()) else {
            continue;
        };
        let episodes = &planned
            .get(article)
            .with_context(|| {
                format!(
                    "rejected.jsonl line {line_no} is for article {article}, but only {} are planned",
                    planned.len()
                )
            })?
            .1;
        let draft: Article = serde_json::from_value(draft.clone())
            .with_context(|| format!("rejected.jsonl line {line_no} holds no article"))?;
        out.push(Rechecked {
            article,
            attempt: row["attempt"].clone(),
            was: row["reason"].as_str().unwrap_or_default().to_string(),
            now: check(&draft, persona, episodes, min_words),
        });
    }
    Ok(out)
}
