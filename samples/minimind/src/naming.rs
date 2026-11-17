// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements task generators whose answers can only be
// copied from the environment, never guessed, for its clients. If your team
// needs expertise in training data for tool-using agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The names and values a task's workspace is made of: names of files,
//! directories, keys and helpers, data paths, answer files and the values
//! the files hold, drawn so nothing an agent must copy is guessable.

use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::formats::{Format, HELD_OUT, TRAINING_EXTENSIONS};
use crate::phrasing::pick;
use crate::skills::ANSWER_FILE;

/// A name for a file, a directory, a key or a helper: a made-up word of
/// `syllables` syllables, a run of random letters about as long, or a word
/// of the `lexicon` (natural words, when the run has them), in equal
/// shares, so copying a name is copying its letters whatever tokens they
/// make, and a real word is copied like any other.
#[must_use]
pub fn name(rng: &mut SplitMix64, syllables: usize, lexicon: &[String]) -> String {
    let kinds = if lexicon.is_empty() { 2 } else { 3 };
    match rng.below(kinds) {
        0 => return crate::world::word(rng, syllables).to_lowercase(),
        2 => return lexicon[rng.below(lexicon.len())].clone(),
        _ => {}
    }
    let n = 3 * syllables.max(1) + rng.below(3) - 1;
    (0..n)
        .map(|_| char::from(b'a' + u8::try_from(rng.below(26)).unwrap_or(0)))
        .collect()
}

/// The lexicon of natural words names are drawn from: the lowercase words
/// of three to ten letters in `texts`, by how often they occur, without the
/// `skip` commonest (articles, pronouns) and at most `size` of them, in
/// alphabetical order.
#[must_use]
pub fn lexicon_of(texts: &[String], skip: usize, size: usize) -> Vec<String> {
    let mut counts: std::collections::HashMap<&str, u64> = std::collections::HashMap::new();
    for text in texts {
        for w in text.split(|c: char| !c.is_ascii_alphabetic()) {
            if (3..=10).contains(&w.len()) && w.bytes().all(|b| b.is_ascii_lowercase()) {
                *counts.entry(w).or_default() += 1;
            }
        }
    }
    let mut ranked: Vec<(&str, u64)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let mut words: Vec<String> = ranked
        .into_iter()
        .skip(skip)
        .take(size)
        .map(|(w, _)| w.to_string())
        .collect();
    words.sort();
    words
}

/// The names of this module's files, keys, directories and helpers.
pub(crate) fn word(rng: &mut SplitMix64, syllables: usize, lexicon: &[String]) -> String {
    name(rng, syllables, lexicon)
}

/// Distinct lowercase names.
pub(crate) fn names(rng: &mut SplitMix64, n: usize, lexicon: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    while out.len() < n {
        let w = word(rng, 2, lexicon).to_lowercase();
        if !out.contains(&w) {
            out.push(w);
        }
    }
    out
}

/// A data file's path for `format`, not in `taken`: at the workspace's
/// root half the time, else in a directory (in training sometimes one of a
/// made-up name). Held out, it has the format's own extension; in training,
/// a third of the time the format's own, a third any of
/// [`TRAINING_EXTENSIONS`] and a third a few random letters: nothing in a
/// path is predictable from habit, so every part of it, whether it has a
/// directory at all included, is copied from the request and the listing.
pub fn data_path(
    rng: &mut SplitMix64,
    format: Format,
    held_out: bool,
    taken: &[String],
    lexicon: &[String],
) -> String {
    loop {
        let dir = if rng.coin() {
            String::new()
        } else if !held_out && rng.coin() {
            format!("{}/", word(rng, 1, lexicon).to_lowercase())
        } else {
            pick(rng, &["data/", "config/", "inputs/", "notes/"]).to_string()
        };
        let extension = match (held_out, rng.below(3)) {
            (true, _) | (false, 0) => format.extension().to_string(),
            (false, 1) => TRAINING_EXTENSIONS[rng.below(TRAINING_EXTENSIONS.len())].to_string(),
            (false, _) => letters(rng),
        };
        let syllables = 1 + rng.below(2);
        let p = format!(
            "{dir}{}.{extension}",
            word(rng, syllables, lexicon).to_lowercase()
        );
        if !clashes(&p, taken) {
            return p;
        }
    }
}

/// Extensions that name binary files by convention; the agent runtime
/// shows such a file as a hex dump, whatever it holds.
const BINARY_EXTENSIONS: [&str; 38] = [
    "o", "a", "so", "elf", "exe", "dll", "wasm", "pdb", "zip", "gz", "tar", "bz", "xz", "zst",
    "bin", "img", "rom", "fw", "srec", "mot", "pyc", "pyo", "jar", "war", "doc", "docx", "xls",
    "xlsx", "ppt", "pptx", "pdf", "obj", "lib", "png", "jpg", "gif", "iso", "class",
];

/// Whether `path` cannot sit beside `taken`: it is one of them, one of them
/// is a directory of it, it is a directory of one of them, or it has the
/// file name of one of them (a request may name a file by that alone).
pub(crate) fn clashes(path: &str, taken: &[String]) -> bool {
    let name = |p: &str| p.rsplit('/').next().unwrap_or(p).to_string();
    taken.iter().any(|t| {
        t == path
            || path.starts_with(&format!("{t}/"))
            || t.starts_with(&format!("{path}/"))
            || name(t) == name(path)
    })
}

/// How a training request names the file at `path`: by its path, or, for a
/// file in a directory, a third of the time by its file name alone, so the
/// agent takes the place of a file from the workspace's listing, not from
/// the request. Held out, always by its path.
pub(crate) fn mention(rng: &mut SplitMix64, path: &str, held_out: bool) -> String {
    match path.rsplit_once('/') {
        Some((_, name)) if !held_out && rng.below(3) == 0 => name.to_string(),
        _ => path.to_string(),
    }
}

/// Two to four random lowercase letters, never a held-out format's
/// extension nor one that names a binary file.
pub(crate) fn letters(rng: &mut SplitMix64) -> String {
    loop {
        let n = 2 + rng.below(3);
        let s: String = (0..n)
            .map(|_| char::from(b'a' + u8::try_from(rng.below(26)).unwrap_or(0)))
            .collect();
        if !HELD_OUT.iter().any(|f| f.extension() == s) && !BINARY_EXTENSIONS.contains(&s.as_str())
        {
            return s;
        }
    }
}

/// The file a request asks its answer to be written to, not in `taken`:
/// [`ANSWER_FILE`] two times in five, otherwise another name (a directory
/// sometimes), so the agent writes where the request says, not where it
/// always did.
pub fn answer_file(rng: &mut SplitMix64, taken: &[String], lexicon: &[String]) -> String {
    if rng.below(5) < 2 {
        return ANSWER_FILE.into();
    }
    loop {
        let dir = pick(rng, &["", "", "", "results/", "output/"]);
        let stem = if rng.coin() {
            pick(
                rng,
                &[
                    "result", "output", "value", "reply", "solution", "final", "number", "found",
                ],
            )
            .to_string()
        } else {
            word(rng, 2, lexicon).to_lowercase()
        };
        let p = format!("{dir}{stem}.txt");
        if !clashes(&p, taken) && p != ANSWER_FILE {
            return p;
        }
    }
}

/// An operand a helper computes with.
pub(crate) fn number(rng: &mut SplitMix64) -> String {
    rng.between(10, 9999).to_string()
}

/// A value a file holds: a whole number of one to six digits, a word, or a
/// code of letters and digits two to six long, so neither the kind nor the
/// length of an answer can be learned instead of copied.
pub(crate) fn any_value(rng: &mut SplitMix64, lexicon: &[String]) -> String {
    match rng.below(3) {
        0 => {
            let digits = u32::try_from(1 + rng.below(6)).unwrap_or(1);
            let low = if digits == 1 {
                0
            } else {
                10u64.pow(digits - 1)
            };
            rng.between(low, 10u64.pow(digits) - 1).to_string()
        }
        1 => {
            let syllables = 1 + rng.below(2);
            word(rng, syllables, lexicon).to_lowercase()
        }
        _ => loop {
            let n = 2 + rng.below(5);
            let code: String = (0..n)
                .map(|_| {
                    let c = rng.below(36);
                    let c = u8::try_from(c).unwrap_or(0);
                    char::from(if c < 26 { b'a' + c } else { b'0' + c - 26 })
                })
                .collect();
            let digit = code.bytes().any(|b| b.is_ascii_digit());
            let letter = code.bytes().any(|b| b.is_ascii_lowercase());
            if digit && letter {
                break code;
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lexicon_holds_a_corpus_s_lowercase_words_past_the_commonest() {
        let text = "the the the cat sat. The cat ran to a river, and the river ran on. Tom ran."
            .to_string();
        assert_eq!(
            lexicon_of(&[text], 1, 10),
            ["and", "cat", "ran", "river", "sat"]
        );
    }

    #[test]
    fn a_third_of_the_names_are_lexicon_words_when_there_is_one() {
        let lexicon: Vec<String> = ["harbour", "lantern", "meadow"].map(String::from).to_vec();
        // seed: any seed; the share holds for every draw.
        let instances = crate::tasks::instances_named(5, crate::tasks::Split::Train, 200, &lexicon);
        let named = instances
            .iter()
            .flat_map(|i| i.files.keys())
            .filter(|p| lexicon.iter().any(|w| p.contains(w.as_str())))
            .count();
        let files: usize = instances.iter().map(|i| i.files.len()).sum();
        assert!(
            named * 6 > files,
            "{named} of {files} paths hold a lexicon word"
        );
        assert!(crate::tasks::instances(5, crate::tasks::Split::Train, 50)
            .iter()
            .flat_map(|i| i.files.keys())
            .all(|p| lexicon.iter().all(|w| !p.contains(w.as_str()))));
    }
}
