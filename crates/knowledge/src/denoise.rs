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
//! separated words of one part of a source, drops one word and swaps two
//! adjacent ones, and asks for the original passage back. Everything is a
//! function of the part's content and the seed - the random draws are
//! seeded by both, so one seed corrupts different parts at different
//! positions - and a task is reproducible from them alone. The original
//! passage is the task's evidence span - naming the source and part, so
//! the source store resolves it to the passage's bytes - and its
//! privileged reference: what the verifier compares against and the
//! student never sees.

use splinter_lab::denoise::KIND;
use splinter_record::experience::{
    Digest, Environment, ExperienceError, Privileged, PrivilegedKind, Span, Task,
};
use splinter_record::source::{PartRef, Source, SourceId};

/// Words in a denoise passage, when the part has that many.
pub const SPAN_WORDS: usize = 12;

/// The fewest words a part may have: enough that dropping one and
/// swapping two still leaves a passage to restore.
pub const MIN_WORDS: usize = 4;

/// The generator's name in an experience's provenance.
pub const GENERATOR: &str = "splinter-knowledge/denoise@2";

/// Why no task could be generated.
#[derive(Debug, thiserror::Error)]
pub enum DenoiseError {
    /// The source has no part of the name given.
    #[error("source {source_id} has no part {part:?}")]
    UnknownPart {
        /// The source.
        source_id: SourceId,
        /// The part name given.
        part: String,
    },
    /// The content given is not the content of the part.
    #[error("the content given hashes to {actual}, not to the part's {claimed}")]
    Digest {
        /// The part's content digest.
        claimed: Digest,
        /// The digest of the content given.
        actual: Digest,
    },
    /// The part's content is not UTF-8 text.
    #[error("part {part:?} of {source_id} is not UTF-8 text")]
    NotText {
        /// The source.
        source_id: SourceId,
        /// The part.
        part: String,
    },
    /// The part has too few words to corrupt.
    #[error("the part has {words} word(s); a denoise task needs at least {MIN_WORDS}")]
    TooShort {
        /// Words in the part.
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

    /// A denoise task over part `part` of `source`, whose content is
    /// `content` (checked against the part's digest).
    pub fn generate(
        &self,
        source: &Source,
        part: &str,
        content: &[u8],
    ) -> Result<Task, DenoiseError> {
        let found = source.part(part).ok_or_else(|| DenoiseError::UnknownPart {
            source_id: source.id.clone(),
            part: part.to_string(),
        })?;
        let actual = Digest::of(content);
        if actual != found.content {
            return Err(DenoiseError::Digest {
                claimed: found.content.clone(),
                actual,
            });
        }
        let text = std::str::from_utf8(content).map_err(|_| DenoiseError::NotText {
            source_id: source.id.clone(),
            part: part.to_string(),
        })?;
        let part_ref = PartRef {
            source: source.id.clone(),
            name: part.to_string(),
        };
        self.generate_text(text, part_ref, &found.content)
    }

    /// The task over `text`, the content of `part` whose digest is
    /// `digest`.
    fn generate_text(
        &self,
        text: &str,
        part: PartRef,
        digest: &Digest,
    ) -> Result<Task, DenoiseError> {
        let words = word_ranges(text);
        if words.len() < MIN_WORDS {
            return Err(DenoiseError::TooShort { words: words.len() });
        }
        let mut rng = SplitMix64(self.seed ^ digest_bits(digest));
        let len = SPAN_WORDS.min(words.len());
        let first = rng.below(words.len() - len + 1);
        let window = &words[first..first + len];
        let (start, end) = (window[0].0, window[len - 1].1);

        let mut corrupted: Vec<&str> = window.iter().map(|&(s, e)| &text[s..e]).collect();
        corrupted.remove(rng.below(corrupted.len()));
        let swap = rng.below(corrupted.len() - 1);
        corrupted.swap(swap, swap + 1);

        let span = Span::in_part(part, digest.clone(), start as u64, end as u64)?;
        let instruction = format!(
            "The passage below was corrupted: one word was dropped and two adjacent words were \
             swapped. Restore the original passage. Reply with the restored passage only.\n\n{}",
            corrupted.join(" ")
        );
        Ok(Task::new(
            KIND,
            vec![span.clone()],
            Environment::closed_book(),
            instruction,
            vec![Privileged {
                kind: PrivilegedKind::Reference,
                content: text[start..end].to_string(),
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

/// The first 64 bits of `digest`, which mix the content into the draws.
fn digest_bits(digest: &Digest) -> u64 {
    digest
        .hex()
        .get(..16)
        .and_then(|hex| u64::from_str_radix(hex, 16).ok())
        .unwrap_or(0)
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
