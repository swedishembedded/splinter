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

use std::collections::HashMap;
use std::fmt::Write;

use splinter_model::stats::sign_test;

use crate::exam::Graded;
use crate::tasks::Kind;

/// One row: a kind of question on one side of the split.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// What the questions ask.
    pub kind: Kind,
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

/// Rows over the questions both arms answered, ordered by split then kind.
#[must_use]
pub fn rows(before: &[Graded], after: &[Graded]) -> Vec<Row> {
    let after_by_id: HashMap<&str, &Graded> = after.iter().map(|g| (g.id.as_str(), g)).collect();
    let mut groups: HashMap<(String, u8), Vec<(&Graded, &Graded)>> = HashMap::new();
    for b in before {
        if let Some(a) = after_by_id.get(b.id.as_str()) {
            groups
                .entry((b.split.clone(), kind_order(b.kind)))
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
                kind: pairs[0].0.kind,
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

fn kind_order(kind: Kind) -> u8 {
    match kind {
        Kind::Recipient => 0,
        Kind::Year => 1,
        Kind::Work => 2,
    }
}

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
    let mut out = String::new();
    let _ = writeln!(out, "| split | question | n | before | after | gained | lost | unanswered (before/after) | p (after better) |");
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|");
    for row in rows(before, after) {
        let percent = |k: usize| format!("{k} ({:.0}%)", 100.0 * k as f64 / row.n.max(1) as f64);
        let chance_note = chance(row.kind, works)
            .map_or(String::new(), |c| format!(" (chance {:.0}%)", 100.0 * c));
        let _ = writeln!(
            out,
            "| {} | {:?}{} | {} | {} | {} | {} | {} | {}/{} | {:.4} |",
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
        let ids: std::collections::HashSet<&str> = b.iter().map(|g| g.id.as_str()).collect();
        a.iter().filter(|g| !ids.contains(g.id.as_str())).count()
    };
    let (unpaired_before, unpaired_after) = (only(before, after), only(after, before));
    if unpaired_before + unpaired_after > 0 {
        let _ = writeln!(
            out,
            "\n{unpaired_before} questions were answered only before and {unpaired_after} only after; they are left out."
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graded(id: &str, split: &str, kind: Kind, correct: bool, answered: bool) -> Graded {
        Graded {
            id: id.into(),
            kind,
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
            graded("a", "exam", Kind::Year, false, false),
            graded("b", "exam", Kind::Year, true, true),
            graded("c", "exam", Kind::Year, false, true),
            graded("only-before", "exam", Kind::Year, true, true),
        ];
        let after = vec![
            graded("a", "exam", Kind::Year, true, true),
            graded("b", "exam", Kind::Year, false, true),
            graded("c", "exam", Kind::Year, true, true),
        ];
        let found = rows(&before, &after);
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
            .map(|n| graded(&n.to_string(), "exam", Kind::Work, false, true))
            .collect();
        let after: Vec<Graded> = (0..12)
            .map(|n| graded(&n.to_string(), "exam", Kind::Work, true, true))
            .collect();
        assert!(rows(&before, &after)[0].p_value < 0.001);
        assert!(
            (rows(&before, &before)[0].p_value - 1.0).abs() < 1e-12,
            "no discordant pair is no evidence"
        );
    }

    #[test]
    fn the_table_names_chance_for_the_closed_list_and_flags_unpaired_questions() {
        let before = vec![
            graded("a", "seen", Kind::Work, true, true),
            graded("x", "seen", Kind::Work, true, true),
        ];
        let after = vec![graded("a", "seen", Kind::Work, true, true)];
        let table = render(&before, &after, 17);
        assert!(table.contains("chance 6%"), "{table}");
        assert!(
            table.contains("1 questions were answered only before"),
            "{table}"
        );
        assert_eq!(chance(Kind::Year, 17), None);
    }
}
