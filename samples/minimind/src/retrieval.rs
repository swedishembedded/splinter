// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements cheap capability probes that track what a
// model learns while it pretrains, for its clients. If your team needs
// expertise in measuring in-context retrieval in small models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The retrieval probe: can the model copy a value out of its context?
//!
//! Every task needs it (a code from a listing, a file name from the request,
//! numbers into a command), and it is what a small model acquires only from
//! enough text that demands it. The probe is raw-text continuation of
//! held-out prompts in the surface forms the tasks use, drawn from a stream
//! no corpus document uses, scored by exact match of the copied value; it
//! is cheap enough to run after every pretraining segment, so the run shows
//! the curve of retrieval against pretraining tokens.

use std::fmt::Write as _;

use anyhow::Context as _;
use serde_json::json;
use splinter_sdk::model::scratch::Checkpoint;
use splinter_sdk::store::stage_ledger::{Commit, StageWork};
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::recipe;
use crate::stages::Run;
use crate::world::word;

/// A world constant: the stream the probe's prompts are drawn from.
const SALT: u64 = 0x7265_7472_6965_7665;

/// Prompts per kind.
const PER_KIND: usize = 12;

/// The longest continuation read, in tokens.
const MAX_TOKENS: u32 = 12;

/// One prompt: its kind, the text, and the value its continuation must
/// start with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    /// `listing` (a code out of a read listing) or `path` (a file name out
    /// of a request into a call).
    pub kind: &'static str,
    /// The text the model continues.
    pub text: String,
    /// What the continuation must start with.
    pub expected: String,
}

/// The probe's prompts for `seed`.
#[must_use]
pub fn prompts(seed: u64) -> Vec<Prompt> {
    let mut rng = SplitMix64::new(seed ^ SALT);
    let mut out = Vec::new();
    for _ in 0..PER_KIND {
        let entries: Vec<(String, u64)> = (0..6)
            .map(|_| (word(&mut rng, 2).to_lowercase(), rng.between(1000, 9999)))
            .collect();
        let (name, code) = entries[rng.below(entries.len())].clone();
        let listing: String = entries
            .iter()
            .enumerate()
            .map(|(i, (n, c))| format!("L{}:{n}: {c}\n", i + 1))
            .collect();
        out.push(Prompt {
            kind: "listing",
            text: format!("{listing}The code of {name} is"),
            expected: format!(" {code}"),
        });
    }
    for _ in 0..PER_KIND {
        let file = format!("{}.txt", word(&mut rng, 2).to_lowercase());
        out.push(Prompt {
            kind: "path",
            text: format!("Read the number in inputs/{file}.\nread_file {{\"path\": \"inputs/"),
            expected: format!("{file}\""),
        });
    }
    out
}

impl Run<'_> {
    /// The retrieval probe of `checkpoint`, as unit `retrieval/<name>`:
    /// exact matches by kind, shown on one line.
    ///
    /// # Errors
    /// The model cannot be served.
    pub fn retrieval(
        &mut self,
        name: &str,
        checkpoint: (&Commit, &Checkpoint),
    ) -> anyhow::Result<Commit> {
        let digest = checkpoint
            .0
            .output("checkpoint")
            .context("the stage has no checkpoint")?
            .clone();
        let unit = format!("retrieval/{name}");
        let key = self.key(
            &json!({ "per_kind": PER_KIND }),
            &unit,
            &[("checkpoint", &digest)],
            recipe::PROBE,
            &["brain"],
        );
        let ckpt = checkpoint.1.clone();
        let commit = self.unit(&unit, &key, |run, _| {
            let mut listing = 0;
            let mut path = 0;
            let mut examples = Vec::new();
            for prompt in prompts(run.seed) {
                let got = run.brain.complete(&ckpt, &prompt.text, MAX_TOKENS)?;
                let hit = got.starts_with(&prompt.expected);
                match (prompt.kind, hit) {
                    ("listing", true) => listing += 1,
                    ("path", true) => path += 1,
                    _ => {}
                }
                if examples.len() < 2 && prompt.kind == "listing" {
                    examples.push(json!({ "expected": prompt.expected, "got": got }));
                }
            }
            Ok(StageWork::new(json!({
                "listing": listing, "path": path, "per_kind": PER_KIND, "examples": examples,
            })))
        })?;
        let mut shown = String::new();
        let _ = writeln!(
            shown,
            "  retrieval {name}: copies a code out of a listing {}/{PER_KIND}, a file name into a call {}/{PER_KIND}",
            commit.summary["listing"], commit.summary["path"]
        );
        write!(self.out, "{shown}")?;
        Ok(commit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompts_hold_their_answer_in_context_and_repeat_for_a_seed() {
        // seed: determinism is the property under test; any seed works.
        let p = prompts(5);
        assert_eq!(p, prompts(5));
        assert_eq!(p.len(), 2 * PER_KIND);
        for prompt in &p {
            let value = prompt.expected.trim().trim_end_matches('"');
            assert!(prompt.text.contains(value), "{prompt:?}");
        }
    }
}
