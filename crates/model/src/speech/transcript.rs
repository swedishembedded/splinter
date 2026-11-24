// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for checking that recorded speech
// says what its transcript says before it becomes training data, for its
// clients. If your team needs expertise in curating voice datasets, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Whether a recording says its transcript, judged from what a recogniser
//! heard in it.
//!
//! A recogniser never writes a transcript back letter for letter: it spells
//! names its own way, splits or joins compounds ("kvistfritt" heard as "kvist
//! fritt") and mishears hard sounds. What must be caught is a different text
//! read into the file, a take read twice and a take cut short. So the texts
//! are compared on their letters and digits alone, lower case, with spacing,
//! punctuation and bracketed non-verbal marks (`[laughter]`) removed:
//!
//! * `similarity` is one minus the character edit distance over the
//!   reference's length, at most 1 and below 0 when the hearing is longer
//!   than the reference by more than it shares with it;
//! * `length_ratio` is the hearing's length over the reference's, which a
//!   take read twice puts near 2 and a take cut in half near 0.5 whatever
//!   the recogniser's spelling;
//! * `numerals_differ` says one side writes digits where the other has none:
//!   a recogniser writing "074152378" for a number spelled out in words.
//!   Then neither number above can tell a misread from that, and the take is
//!   for review rather than failed.

use serde::{Deserialize, Serialize};

/// How closely what was heard matches the transcript.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TranscriptMatch {
    /// One minus the character edit distance over the reference's length.
    pub similarity: f64,
    /// The hearing's length over the reference's, in compared characters.
    pub length_ratio: f64,
    /// One side writes digits and the other none.
    pub numerals_differ: bool,
}

/// The characters a transcript is compared on: letters and digits, lower
/// case, with bracketed marks such as `[laughter]` left out.
#[must_use]
pub fn compared_chars(text: &str) -> Vec<char> {
    let mut out = Vec::with_capacity(text.len());
    let mut in_mark = false;
    for c in text.chars() {
        match c {
            '[' => in_mark = true,
            ']' => in_mark = false,
            c if !in_mark && c.is_alphanumeric() => out.extend(c.to_lowercase()),
            _ => {}
        }
    }
    out
}

/// The fewest character substitutions, insertions and deletions that turn
/// `a` into `b`, one table row at a time.
fn edits(a: &[char], b: &[char]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let substitute = diagonal + usize::from(x != y);
            diagonal = row[j + 1];
            row[j + 1] = substitute.min(row[j + 1] + 1).min(row[j] + 1);
        }
    }
    row[b.len()]
}

/// How closely `heard` matches `reference`. A reference with nothing to
/// compare matches only a hearing with nothing either.
#[must_use]
pub fn compare(reference: &str, heard: &str) -> TranscriptMatch {
    let (said, got) = (compared_chars(reference), compared_chars(heard));
    let has_digits = |chars: &[char]| chars.iter().any(char::is_ascii_digit);
    let numerals_differ = has_digits(&said) != has_digits(&got);
    if said.is_empty() {
        let same = if got.is_empty() { 1.0 } else { 0.0 };
        return TranscriptMatch {
            similarity: same,
            length_ratio: same,
            numerals_differ,
        };
    }
    let n = said.len() as f64;
    TranscriptMatch {
        similarity: 1.0 - edits(&said, &got) as f64 / n,
        length_ratio: got.len() as f64 / n,
        numerals_differ,
    }
}

/// What a match means for the recording.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// It says its transcript.
    Pass,
    /// Close, but listen to it: most often the recogniser mishearing hard
    /// sounds, sometimes a misread word.
    Review,
    /// A different text, a repeated or cut-short take, or nothing heard.
    Fail,
}

/// The bounds a match is judged by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchPolicy {
    /// At or above this similarity a take passes.
    pub pass_at: f64,
    /// Below this similarity a take fails; between the two it is reviewed.
    pub fail_below: f64,
    /// The length ratios a take may have; outside them it fails, whatever
    /// its similarity.
    pub length_ratio: (f64, f64),
}

impl Default for MatchPolicy {
    /// A similarity of 0.90 passes and below 0.60 fails; a hearing 0.8 to
    /// 1.25 times the transcript's length is in range. The bands come from a
    /// Qwen3-ASR hearing of correct English, Swedish and mixed takes against
    /// a take of the wrong prompt, one read twice and one cut in half; the
    /// tests below pin each case.
    fn default() -> Self {
        MatchPolicy {
            pass_at: 0.90,
            fail_below: 0.60,
            length_ratio: (0.80, 1.25),
        }
    }
}

impl MatchPolicy {
    /// The verdict on `m`, with the reason for anything but a pass.
    #[must_use]
    pub fn judge(&self, m: &TranscriptMatch) -> (Verdict, Option<String>) {
        let (low, high) = self.length_ratio;
        let off = m.length_ratio < low || m.length_ratio > high || m.similarity < self.pass_at;
        if m.numerals_differ && off {
            return (Verdict::Review, Some(format!("similarity {:.2}, {:.2} of the length, but numbers are written as digits on one side and words on the other: listen to it", m.similarity, m.length_ratio)));
        }
        if m.length_ratio > high {
            return (Verdict::Fail, Some(format!("heard {:.2} times the transcript's length: a repeated take, or more than the prompt", m.length_ratio)));
        }
        if m.length_ratio < low {
            return (
                Verdict::Fail,
                Some(format!(
                    "heard {:.2} of the transcript's length: cut short, or a different prompt",
                    m.length_ratio
                )),
            );
        }
        if m.similarity < self.fail_below {
            return (
                Verdict::Fail,
                Some(format!(
                    "similarity {:.2}: a different text than the transcript",
                    m.similarity
                )),
            );
        }
        if m.similarity < self.pass_at {
            return (
                Verdict::Review,
                Some(format!(
                    "similarity {:.2}: listen to it; a misread word or the recogniser mishearing",
                    m.similarity
                )),
            );
        }
        (Verdict::Pass, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAID: &str = "Kvistfritt kvastskaft, kvistfritt kvastskaft. Sex laxar i en laxask!";

    #[test]
    fn spacing_case_punctuation_and_marks_do_not_count() {
        let m = compare(
            SAID,
            "kvist fritt KVAST skaft kvistfritt kvastskaft sex laxar i en laxask",
        );
        assert_eq!(m.similarity, 1.0);
        assert_eq!(
            compare(
                "Oh fantastic. [laughter] No, really.",
                "oh fantastic no really"
            )
            .similarity,
            1.0
        );
        assert_eq!(
            compared_chars("Mååål, 2027!"),
            "mååål2027".chars().collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_recogniser_mishearing_a_few_sounds_still_passes() {
        let said = "Sju sjösjuka sjömän sköttes av sju sköna sjuksköterskor på skeppet Shanghai.";
        let heard =
            "Sju sjukhjärta sjömän sköttes av sju sköna sjuksköterskor på skeppet Shanghai.";
        let m = compare(said, heard);
        assert!(m.similarity > 0.9 && m.similarity < 1.0, "{m:?}");
        assert_eq!(MatchPolicy::default().judge(&m).0, Verdict::Pass);
    }

    #[test]
    fn another_text_a_repeated_take_and_a_cut_take_fail() {
        let policy = MatchPolicy::default();
        let other = compare(
            SAID,
            "Det regnade hela dagen när vi sa hejdå på perrongen vid tåget.",
        );
        assert_eq!(policy.judge(&other).0, Verdict::Fail, "{other:?}");
        let twice = compare(SAID, &format!("{SAID} {SAID}"));
        assert!(
            (twice.length_ratio - 2.0).abs() < 1e-9 && twice.similarity < 0.1,
            "{twice:?}"
        );
        let (verdict, why) = policy.judge(&twice);
        assert_eq!(verdict, Verdict::Fail);
        assert!(why.unwrap().contains("repeated"));
        let half: String = SAID.chars().take(SAID.len() / 2).collect();
        assert_eq!(policy.judge(&compare(SAID, &half)).0, Verdict::Fail);
        assert_eq!(
            policy.judge(&compare(SAID, "")).0,
            Verdict::Fail,
            "nothing heard"
        );
    }

    #[test]
    fn a_close_but_imperfect_hearing_is_reviewed() {
        let policy = MatchPolicy::default();
        let m = compare(
            SAID,
            "Kvist fritt kast skaft kvist fritt kast skaft sex laxa en lak sask",
        );
        assert!(m.similarity < 0.9 && m.similarity > 0.6, "{m:?}");
        assert_eq!(policy.judge(&m).0, Verdict::Review);
    }

    /// A recogniser writes "074152378" where the transcript spells the
    /// number out: neither the length nor the letters can tell a misread
    /// from that, so the take is for review, never failed on it.
    #[test]
    fn numbers_written_as_digits_against_words_are_reviewed_not_failed() {
        let said = "Ring mig på noll sjuttio, fyrahundrafemtio, tjugotre, sjuttioåtta om du blir sen. Det kostar tretusen kronor.";
        let heard = "Ring mig på 074152378 om du blir sen. Det kostar 3000 kronor.";
        let m = compare(said, heard);
        assert!(m.numerals_differ, "{m:?}");
        let (verdict, why) = MatchPolicy::default().judge(&m);
        assert_eq!(verdict, Verdict::Review);
        assert!(why.unwrap().contains("digits"));
        assert!(
            !compare("Room 3B at 9:45", "room 3b at 9 45").numerals_differ,
            "digits on both sides compare as they are"
        );
    }

    #[test]
    fn an_empty_reference_matches_only_silence() {
        assert_eq!(compare("[laughter]", "").similarity, 1.0);
        assert_eq!(compare("", "words").similarity, 0.0);
    }
}
