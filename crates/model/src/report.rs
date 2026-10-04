// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements frozen, resumable evaluations of language
// models against source material for its clients. If your team needs
// expertise in measuring what a fine-tune changed, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Two arms of one exam, paired question by question.
//!
//! A fine-tune is judged on the questions both arms answered: how many each
//! got right, who gained and who lost, how many went unanswered, and whether
//! the gain is more than the paired sign test allows chance to explain.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::exam::Graded;
use crate::stats::sign_test;

/// The graded pairs of one (split, kind-position, kind) group, the base's
/// answer first.
type Pairs<'a> = HashMap<(String, usize, String), Vec<(&'a Graded, &'a Graded)>>;

/// One row: a kind of question on one side of the split.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// What the questions ask.
    pub kind: String,
    /// `exam` (never trained on) or `seen` (trained on).
    pub split: String,
    /// Questions both arms answered.
    pub n: usize,
    /// Right answers before.
    pub before: usize,
    /// Right answers after.
    pub after: usize,
    /// Questions the base gave no answer to.
    pub before_unanswered: usize,
    /// Questions the trained model gave no answer to.
    pub after_unanswered: usize,
    /// Pairs where only the trained model was right.
    pub gained: usize,
    /// Pairs where only the base was right.
    pub lost: usize,
    /// The one-sided p-value that the trained model is better.
    pub p_value: f64,
}

/// Rows over the questions both arms answered, ordered by split, then by the
/// position of the kind in `kind_order` (kinds it does not list come last,
/// alphabetically).
#[must_use]
pub fn rows(before: &[Graded], after: &[Graded], kind_order: &[&str]) -> Vec<Row> {
    let after_by_id: HashMap<&str, &Graded> = after.iter().map(|g| (g.id.as_str(), g)).collect();
    let mut groups: Pairs = HashMap::new();
    for b in before {
        if let Some(a) = after_by_id.get(b.id.as_str()) {
            let position = kind_order
                .iter()
                .position(|k| *k == b.kind)
                .unwrap_or(kind_order.len());
            groups
                .entry((b.split.clone(), position, b.kind.clone()))
                .or_default()
                .push((b, a));
        }
    }
    let mut keys: Vec<_> = groups.keys().cloned().collect();
    keys.sort();
    keys.into_iter()
        .map(|key| {
            let pairs = &groups[&key];
            let outcomes: Vec<(bool, bool)> =
                pairs.iter().map(|(b, a)| (a.correct, b.correct)).collect();
            let test = sign_test(&outcomes);
            Row {
                kind: key.2,
                split: key.0,
                n: pairs.len(),
                before: pairs.iter().filter(|(b, _)| b.correct).count(),
                after: pairs.iter().filter(|(_, a)| a.correct).count(),
                before_unanswered: pairs.iter().filter(|(b, _)| b.answer.is_none()).count(),
                after_unanswered: pairs.iter().filter(|(_, a)| a.answer.is_none()).count(),
                gained: pairs
                    .iter()
                    .filter(|(b, a)| a.correct && !b.correct)
                    .count(),
                lost: pairs
                    .iter()
                    .filter(|(b, a)| b.correct && !a.correct)
                    .count(),
                p_value: test.p_value,
            }
        })
        .collect()
}

/// The report as a Markdown table, with the evidence a reader needs to judge
/// it: how many questions, how many went unanswered, and who gained and lost.
/// `chance` gives the chance level of a kind where one exists.
#[must_use]
pub fn render(
    before: &[Graded],
    after: &[Graded],
    kind_order: &[&str],
    chance: &dyn Fn(&str) -> Option<f64>,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "| split | question | n | before | after | gained | lost | unanswered (before/after) | p (after better) |");
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|");
    for row in rows(before, after, kind_order) {
        let percent = |k: usize| format!("{k} ({:.0}%)", 100.0 * k as f64 / row.n.max(1) as f64);
        let chance_note =
            chance(&row.kind).map_or(String::new(), |c| format!(" (chance {:.0}%)", 100.0 * c));
        let _ = writeln!(
            out,
            "| {} | {}{} | {} | {} | {} | {} | {} | {}/{} | {:.4} |",
            row.split,
            row.kind,
            chance_note,
            row.n,
            percent(row.before),
            percent(row.after),
            row.gained,
            row.lost,
            row.before_unanswered,
            row.after_unanswered,
            row.p_value
        );
    }
    let only = |a: &[Graded], b: &[Graded]| {
        let ids: HashSet<&str> = b.iter().map(|g| g.id.as_str()).collect();
        a.iter().filter(|g| !ids.contains(g.id.as_str())).count()
    };
    let (unpaired_before, unpaired_after) = (only(before, after), only(after, before));
    if unpaired_before + unpaired_after > 0 {
        let _ = writeln!(out, "\n{unpaired_before} questions were answered only before and {unpaired_after} only after; they are left out.");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graded(id: &str, split: &str, kind: &str, correct: bool, answered: bool) -> Graded {
        Graded {
            id: id.into(),
            kind: kind.into(),
            split: split.into(),
            reference: "r".into(),
            answer: answered.then(|| "a".to_string()),
            correct,
            tokens: 10,
            truncated: !answered,
            seconds: 1.0,
        }
    }

    #[test]
    fn rows_pair_the_arms_on_the_same_questions_and_count_gains_and_losses() {
        let before = vec![
            graded("a", "exam", "year", false, false),
            graded("b", "exam", "year", true, true),
            graded("c", "exam", "year", false, true),
            graded("only-before", "exam", "year", true, true),
        ];
        let after = vec![
            graded("a", "exam", "year", true, true),
            graded("b", "exam", "year", false, true),
            graded("c", "exam", "year", true, true),
        ];
        let found = rows(&before, &after, &[]);
        assert_eq!(found.len(), 1);
        let row = &found[0];
        assert_eq!((row.n, row.before, row.after), (3, 1, 2));
        assert_eq!((row.gained, row.lost), (2, 1));
        assert_eq!((row.before_unanswered, row.after_unanswered), (1, 0));
        assert!(row.p_value > 0.0 && row.p_value < 1.0);
    }

    #[test]
    fn a_clear_win_is_significant_and_a_tie_is_not() {
        let before: Vec<Graded> = (0..12)
            .map(|n| graded(&n.to_string(), "exam", "work", false, true))
            .collect();
        let after: Vec<Graded> = (0..12)
            .map(|n| graded(&n.to_string(), "exam", "work", true, true))
            .collect();
        assert!(rows(&before, &after, &[])[0].p_value < 0.001);
        assert!(
            (rows(&before, &before, &[])[0].p_value - 1.0).abs() < 1e-12,
            "no discordant pair is no evidence"
        );
    }

    #[test]
    fn rows_follow_the_order_the_caller_names_within_each_split() {
        let both = |kind| vec![graded(kind, "exam", kind, true, true)];
        let mut before = both("year");
        before.extend(both("attribution"));
        before.extend(both("recipient"));
        let order = ["recipient", "year", "attribution"];
        let kinds: Vec<String> = rows(&before, &before, &order)
            .into_iter()
            .map(|r| r.kind)
            .collect();
        assert_eq!(kinds, ["recipient", "year", "attribution"]);
    }

    #[test]
    fn the_table_names_chance_where_there_is_one_and_flags_unpaired_questions() {
        let before = vec![
            graded("a", "seen", "work", true, true),
            graded("x", "seen", "work", true, true),
        ];
        let after = vec![graded("a", "seen", "work", true, true)];
        let table = render(&before, &after, &[], &|kind| {
            (kind == "work").then_some(1.0 / 17.0)
        });
        assert!(table.contains("chance 6%"), "{table}");
        assert!(
            table.contains("1 questions were answered only before"),
            "{table}"
        );
    }
}
