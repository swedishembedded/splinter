// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data-debugging checks that stop degenerate
// generated training data before a model learns it, for its clients. If your
// team needs expertise in auditing synthetic agent data, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Detectors for degenerate generated data, run where the data is made.
//!
//! A model learns what dominates its data: one request template, one stock
//! final reply, the same document many times. Each detector measures a
//! share, the data stage records it, and a share over its configured bound
//! fails the stage loudly, naming what dominated, instead of letting it train.

use std::collections::BTreeMap;

use anyhow::bail;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// The bounds the data must stay within.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataChecks {
    /// The largest share of episodes whose request has one template (its
    /// rare words, the names and values, masked).
    pub max_template_share: f64,
    /// The largest share of episodes ending in one final reply (its numbers
    /// masked).
    pub max_reply_share: f64,
    /// The largest share of documents or records that repeat an earlier one.
    pub max_duplicate_share: f64,
}

impl Default for DataChecks {
    fn default() -> Self {
        Self {
            max_template_share: 0.25,
            max_reply_share: 0.5,
            max_duplicate_share: 0.05,
        }
    }
}

/// `texts` with their rare words masked: a word in fewer than 2% of them (a
/// name, a value, a path) becomes `_`, digits become `#`.
fn templates(texts: &[String]) -> Vec<String> {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for t in texts {
        let mut words: Vec<&str> = t.split_whitespace().collect();
        words.sort_unstable();
        words.dedup();
        for w in words {
            *seen.entry(w).or_default() += 1;
        }
    }
    let common = |w: &str| seen.get(w).copied().unwrap_or(0) * 50 >= texts.len();
    texts
        .iter()
        .map(|t| {
            t.split_whitespace()
                .map(|w| {
                    if common(w) && !w.chars().any(|c| c.is_ascii_digit()) {
                        w
                    } else {
                        "_"
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// The most frequent item of `items` and its share.
fn top_share(items: &[String]) -> (String, f64) {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for i in items {
        *counts.entry(i).or_default() += 1;
    }
    let (top, n) = counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .unwrap_or(("", 0));
    // A share of a count, both at most the corpus size.
    #[allow(clippy::cast_precision_loss)]
    let share = if items.is_empty() {
        0.0
    } else {
        n as f64 / items.len() as f64
    };
    (top.to_string(), share)
}

/// The share of `items` that repeat an earlier one.
fn duplicate_share(items: &[String]) -> f64 {
    let mut seen = std::collections::BTreeSet::new();
    let repeats = items.iter().filter(|i| !seen.insert(i.as_str())).count();
    // A share of a count, both at most the corpus size.
    #[allow(clippy::cast_precision_loss)]
    let share = if items.is_empty() {
        0.0
    } else {
        repeats as f64 / items.len() as f64
    };
    share
}

/// The demonstration records `lines` (one per model call), checked: the
/// request templates, the final replies and repeated records. Returns the
/// statistics.
///
/// # Errors
/// A share is over its bound.
pub fn demonstrations(lines: &str, bounds: &DataChecks) -> anyhow::Result<Value> {
    let records: Vec<Value> = lines
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    // One request and one final reply per episode: the record of its last
    // call holds both.
    let mut last: BTreeMap<String, &Value> = BTreeMap::new();
    for r in &records {
        last.insert(
            r["metadata"]["episode"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            r,
        );
    }
    let request = |r: &Value| -> String {
        r["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|m| m["role"] == "user")
            .and_then(|m| m["content"].as_str())
            .unwrap_or_default()
            .to_string()
    };
    let reply = |r: &Value| -> String {
        let text = r["messages"]
            .as_array()
            .and_then(|m| m.last())
            .and_then(|m| m["content"].as_str())
            .unwrap_or_default();
        text.chars()
            .map(|c| if c.is_ascii_digit() { '#' } else { c })
            .collect()
    };
    let requests: Vec<String> = last.values().map(|r| request(r)).collect();
    let replies: Vec<String> = last.values().map(|r| reply(r)).collect();
    let (template, template_share) = top_share(&templates(&requests));
    let (final_reply, reply_share) = top_share(&replies);
    let duplicates = duplicate_share(&lines.lines().map(String::from).collect::<Vec<_>>());
    let mut over = Vec::new();
    if template_share > bounds.max_template_share {
        over.push(format!(
            "one request template is {template_share:.2} of the episodes (bound {}): {template:?}",
            bounds.max_template_share
        ));
    }
    if reply_share > bounds.max_reply_share {
        over.push(format!(
            "one final reply is {reply_share:.2} of the episodes (bound {}): {final_reply:?}",
            bounds.max_reply_share
        ));
    }
    if duplicates > bounds.max_duplicate_share {
        over.push(format!(
            "{duplicates:.2} of the records repeat an earlier one (bound {})",
            bounds.max_duplicate_share
        ));
    }
    if !over.is_empty() {
        bail!("the demonstrations are degenerate: {}", over.join("; "));
    }
    Ok(json!({
        "templates": templates(&requests).iter().collect::<std::collections::BTreeSet<_>>().len(),
        "top_template_share": template_share, "top_reply_share": reply_share, "duplicate_share": duplicates,
    }))
}

/// The world's documents, checked for repeats. Returns the statistics.
///
/// # Errors
/// More documents repeat an earlier one than the bound allows.
pub fn corpus(texts: &[String], bounds: &DataChecks) -> anyhow::Result<Value> {
    let duplicates = duplicate_share(texts);
    if duplicates > bounds.max_duplicate_share {
        bail!(
            "{duplicates:.2} of the world's documents repeat an earlier one (bound {})",
            bounds.max_duplicate_share
        );
    }
    Ok(json!({ "duplicate_share": duplicates }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(episode: &str, request: &str, reply: &str) -> String {
        json!({ "metadata": { "episode": episode }, "messages": [
            { "role": "user", "content": request }, { "role": "assistant", "content": reply } ] })
        .to_string()
    }

    #[test]
    fn a_dominant_template_or_reply_fails_and_varied_data_passes() {
        let bounds = DataChecks::default();
        let varied: String = (0..40)
            .map(|i| {
                let verbs = ["Find", "Get", "Fetch", "Read", "Look up"];
                let saves = ["write", "save", "store", "put", "record"];
                let replies = [
                    "Done: # is in answer.txt.",
                    "Wrote # to answer.txt.",
                    "#",
                    "answer.txt now holds #.",
                ];
                format!(
                    "{}\n",
                    record(
                        &format!("e{i}"),
                        &format!(
                            "{} key{i} in file{i}.txt and {} it",
                            verbs[i % 5],
                            saves[i / 5 % 5]
                        ),
                        &replies[i % 4].replace('#', &i.to_string())
                    )
                )
            })
            .collect();
        assert!(
            demonstrations(&varied, &bounds).is_ok(),
            "{:?}",
            demonstrations(&varied, &bounds)
        );
        let stock: String = (0..40)
            .map(|i| {
                format!(
                    "{}\n",
                    record(
                        &format!("e{i}"),
                        &format!("Find key{i} in file{i}.txt"),
                        &format!("The answer is {i}. It is written to answer.txt.")
                    )
                )
            })
            .collect();
        let err = demonstrations(&stock, &bounds).unwrap_err().to_string();
        assert!(
            err.contains("request template") && err.contains("final reply"),
            "{err}"
        );
        let docs: Vec<String> = (0..20).map(|i| format!("doc {}", i % 10)).collect();
        assert!(corpus(&docs, &bounds).is_err(), "half the documents repeat");
    }
}
