// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines that never train a
// model on what a conversation itself says was wrong, for its clients. If your
// team needs expertise in learning from dialogue then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Whether a turn of a conversation corrects the one before it: the rule a
//! dataset uses to keep a corrected reply out of what it trains.

use crate::teaching::{parse_teaching, Teaching, CORRECTION_OPENERS};

/// Words that open a reply refusing what was just said, when punctuation
/// follows them (`No, ...`, `Wrong.`). Lowercase.
const REFUSALS: [&str; 3] = ["no", "wrong", "incorrect"];

/// Whether `transcript` corrects what was said just before it: a correcting
/// fact ([`parse_teaching`]: `Actually, ...`, `That is wrong, ...`), or a
/// reply that opens by refusing it (`No, ...`, `Wrong.`, `Incorrect:`). The
/// refusal must be followed by punctuation, so `No problem` and `Not now`
/// correct nothing; a correction opener alone needs none.
#[must_use]
pub fn corrects(transcript: &str) -> bool {
    if matches!(
        parse_teaching(transcript),
        Some(Teaching::Fact { corrects: true, .. })
    ) {
        return true;
    }
    let text = transcript
        .trim()
        .replace('\u{2019}', "'")
        .replace("hat's ", "hat is ");
    let lower = text.to_ascii_lowercase();
    let punctuated = |rest: &str| rest.starts_with([',', '.', '!', ';', ':', '-']);
    REFUSALS
        .iter()
        .any(|o| lower.strip_prefix(o).is_some_and(punctuated))
        || CORRECTION_OPENERS.iter().any(|o| {
            lower
                .strip_prefix(o)
                .is_some_and(|rest| rest.is_empty() || punctuated(rest) || rest.starts_with(' '))
        })
}
