// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements before-and-after evidence reports for
// fine-tuned language models for its clients. If your team needs expertise in
// proving what a fine-tune changed, with the statistics to back it, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The before-and-after report: what the base answered, what the trained
//! model answered, on the same frozen questions, with the paired sign test
//! Splinter's release gate rests on.

use splinter_sdk::model::report::render as render_table;

use crate::exam::Graded;
use crate::tasks::Kind;

/// The order the kinds of question are reported in.
const KIND_ORDER: [&str; 3] = ["recipient", "year", "work"];

/// The chance level of a kind of question, where one exists: a closed list of
/// `works` offers one in that many; a year or a name has no fixed chance.
#[must_use]
pub fn chance(kind: Kind, works: usize) -> Option<f64> {
    (kind == Kind::Work && works > 0).then(|| 1.0 / works as f64)
}

/// The report as a Markdown table, with the evidence a reader needs to judge
/// it: how many questions, how many went unanswered, and who gained and lost.
#[must_use]
pub fn render(before: &[Graded], after: &[Graded], works: usize) -> String {
    render_table(before, after, &KIND_ORDER, &|kind| {
        chance(
            if kind == Kind::Work.name() {
                Kind::Work
            } else {
                Kind::Year
            },
            works,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chance_is_named_only_for_the_closed_list_of_works() {
        assert_eq!(chance(Kind::Year, 17), None);
        assert!((chance(Kind::Work, 17).unwrap_or(0.0) - 1.0 / 17.0).abs() < 1e-12);
        assert_eq!(chance(Kind::Work, 0), None);
    }
}
