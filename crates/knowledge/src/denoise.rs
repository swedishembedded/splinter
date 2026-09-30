// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements task generators that turn source documents
// into verifiable training tasks, for its clients. If your team needs
// expertise in knowledge acquisition for small models, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The denoise task generator: a passage of a source, corrupted, to be
//! restored.
//!
//! [`Denoise::generate`] picks a window of [`SPAN_WORDS`] whitespace
//! separated words of the source, drops one word and swaps two adjacent
//! ones, and asks for the original passage back. Everything is a function
//! of the source and the seed, so a task is reproducible from them alone.
//! The original passage is the task's evidence span and its privileged
//! reference - what the verifier compares against and the student never
//! sees.

use splinter_lab::denoise::KIND;
use splinter_store::experience::{
    Digest, Environment, ExperienceError, Privileged, PrivilegedKind, Span, Task,
};

/// Words in a denoise passage, when the source has that many.
pub const SPAN_WORDS: usize = 12;

/// The fewest words a source may have: enough that dropping one and
/// swapping two still leaves a passage to restore.
pub const MIN_WORDS: usize = 4;

/// The generator's name in a task's environment spec and an experience's
/// provenance.
pub const GENERATOR: &str = "splinter-knowledge/denoise@1";

/// Why no task could be generated.
#[derive(Debug, thiserror::Error)]
pub enum DenoiseError {
    /// The digest given is not the digest of the text given.
    #[error("the source hashes to {actual}, not the {claimed} it was given as")]
    Digest {
        /// The digest the caller gave.
        claimed: Digest,
        /// The digest of the text.
        actual: Digest,
    },
    /// The source has too few words to corrupt.
    #[error("the source has {words} word(s); a denoise task needs at least {MIN_WORDS}")]
    TooShort {
        /// Words in the source.
        words: usize,
    },
    /// The task failed validation.
    #[error(transparent)]
    Task(#[from] ExperienceError),
}

/// A seeded denoise task generator.
#[derive(Clone, Copy, Debug)]
pub struct Denoise {
    seed: u64,
}

impl Denoise {
    /// The generator for `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// A denoise task over `source`, whose digest is `digest` (checked).
    pub fn generate(&self, source: &str, digest: &Digest) -> Result<Task, DenoiseError> {
        let actual = Digest::of(source.as_bytes());
        if actual != *digest {
            return Err(DenoiseError::Digest {
                claimed: digest.clone(),
                actual,
            });
        }
        let words = word_ranges(source);
        if words.len() < MIN_WORDS {
            return Err(DenoiseError::TooShort { words: words.len() });
        }
        let mut rng = SplitMix64(self.seed);
        let len = SPAN_WORDS.min(words.len());
        let first = rng.below(words.len() - len + 1);
        let window = &words[first..first + len];
        let (start, end) = (window[0].0, window[len - 1].1);

        let mut corrupted: Vec<&str> = window.iter().map(|&(s, e)| &source[s..e]).collect();
        corrupted.remove(rng.below(corrupted.len()));
        let swap = rng.below(corrupted.len() - 1);
        corrupted.swap(swap, swap + 1);

        let span = Span::new(digest.clone(), start as u64, end as u64)?;
        let instruction = format!(
            "The passage below was corrupted: one word was dropped and two adjacent words were \
             swapped. Restore the original passage. Reply with the restored passage only.\n\n{}",
            corrupted.join(" ")
        );
        Ok(Task::new(
            KIND,
            vec![span.clone()],
            Environment {
                kind: "closed-book".into(),
                spec: serde_json::json!({ "generator": GENERATOR, "seed": self.seed }),
                snapshot: None,
            },
            instruction,
            vec![Privileged {
                kind: PrivilegedKind::Reference,
                content: source[start..end].to_string(),
                span: Some(span),
            }],
        )?)
    }
}

/// The byte range of every whitespace-separated word of `text`.
fn word_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut words = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (c.is_whitespace(), start) {
            (true, Some(s)) => {
                words.push((s, i));
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }
    if let Some(s) = start {
        words.push((s, text.len()));
    }
    words
}

/// SplitMix64: a small, fixed, well-mixed generator, so a seed means the
/// same task on every platform and every build.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n`; `n` is at least 1 at every call site.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}
