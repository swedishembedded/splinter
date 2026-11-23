// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the stages that make and choose tasks print: the tasks stage with
//! what the generator or the invent harness yielded, and the validate stage.

use std::fmt::Write as _;

use splinter_sdk::canaries::CanariesMade;
use splinter_sdk::tasks::pick::Picked;
use splinter_sdk::tasks::{duration_words, TasksGenerated};
use splinter_sdk::variants::VariantsGenerated;

use crate::output::{tally, Report};

impl Report for TasksGenerated {
    fn human(&self) -> String {
        let mut out = format!(
            "task set {}: {} task(s) from {} of {} text part(s) in {}\n",
            self.task_set,
            self.tasks,
            self.parts,
            self.parts_total,
            duration_words(self.elapsed_secs)
        );
        for (kind, counts) in &self.per_kind {
            let _ = writeln!(
                out,
                "  {kind}: {} admitted, {} rejected",
                counts.admitted, counts.rejected
            );
        }
        if !self.calls.is_empty() {
            let model_secs = self.calls.iter().map(|c| c.elapsed_ms).sum::<u64>() / 1000;
            let _ = writeln!(
                out,
                "  {} request(s) to the generator, {} of model time",
                self.calls.len(),
                duration_words(model_secs)
            );
        }
        if self.sections > 0 {
            let _ = writeln!(
                out,
                "  {} queued concept section(s) generated from",
                self.sections
            );
        }
        if !self.rejected.is_empty() {
            let _ = writeln!(out, "  rejected: {}", tally(&self.rejected));
        }
        // The first few say why; `--json` carries every one, whole.
        for r in self.rejections.iter().take(REJECTIONS_SHOWN) {
            let detail: String = r.detail.chars().take(REJECTION_DETAIL_CHARS).collect();
            let _ = writeln!(out, "    {} {}: {}", r.kind, r.reason, detail.trim());
        }
        if let Some(hint) = generator_hint(
            self.tasks,
            self.rejected.values().sum(),
            !self.calls.is_empty(),
        ) {
            let _ = writeln!(out, "  {hint}");
        }
        for (kind, why) in &self.dropped {
            let _ = writeln!(out, "  gave up on {kind}: {why}");
        }
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "  stopped: {why}");
        }
        out
    }
}

/// What to say when a model was asked for tasks and none passed admission: the
/// admission rules are the same for every generator, and a small one often
/// cannot meet them.
fn generator_hint(admitted: usize, rejected: usize, generator_asked: bool) -> Option<&'static str> {
    (generator_asked && admitted == 0 && rejected > 0).then_some(
        "every proposal was rejected: a small generator often cannot meet the admission rules, so name a stronger one with --generator",
    )
}

/// Rejections a task report shows with their detail.
const REJECTIONS_SHOWN: usize = 3;
/// Characters of a rejection's detail shown.
const REJECTION_DETAIL_CHARS: usize = 300;

impl Report for splinter_sdk::tasks::drills::Drilled {
    fn human(&self) -> String {
        let mut out = format!(
            "drills  {} of {} function(s) (seed {}, at most {} each, tree {}): {} of the exam's \
             files, {} of training files\n  refused {}\n",
            self.admitted,
            self.functions,
            self.seed.value(),
            self.per_function,
            &self.tree_id[..self.tree_id.len().min(12)],
            self.exam_drills,
            self.training_drills,
            tally(&self.refused)
        );
        if let Some(scope) = &self.scope {
            let _ = writeln!(
                out,
                "  for {} task(s) ({} without a reference change): {} function(s) touched, \
                 {} file(s) changed outside any function",
                scope.tasks,
                scope.without_reference,
                scope.functions.len(),
                scope.outside_functions.len()
            );
        }
        if !self.unreadable.is_empty() {
            let _ = writeln!(out, "  unreadable  {}", self.unreadable.join(", "));
        }
        for (label, set) in [
            ("set", &self.task_set),
            ("exam", &self.exam_task_set),
            ("training", &self.training_task_set),
        ] {
            if let Some(set) = set {
                let _ = writeln!(out, "  {label:<9} {set}");
            }
        }
        for (family, count) in &self.families {
            let _ = writeln!(out, "  {count:>3}  {family}");
        }
        out
    }
}

impl Report for VariantsGenerated {
    fn human(&self) -> String {
        let mut out = match &self.variant_set {
            Some(set) => format!(
                "task set {set}: {} variant(s) of {} task(s)\n",
                self.variants, self.tasks
            ),
            None => format!("no variant admitted for {} task(s)\n", self.tasks),
        };
        if !self.ineligible.is_empty() {
            let _ = writeln!(out, "  not varied: {}", tally(&self.ineligible));
        }
        if !self.rejected.is_empty() {
            let _ = writeln!(out, "  rejected: {}", tally(&self.rejected));
        }
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "  stopped: {why}");
        }
        out
    }
}

impl Report for CanariesMade {
    fn human(&self) -> String {
        let mut out = match &self.canary_set {
            Some(set) => format!(
                "task set {set}: {} canar{}; hand it to the gate with `release --canaries {set}`\n",
                self.canaries,
                if self.canaries == 1 { "y" } else { "ies" }
            ),
            None => "no canary made: the set holds no workspace task with a check\n".into(),
        };
        if !self.skipped.is_empty() {
            let _ = writeln!(out, "  left out: {}", tally(&self.skipped));
        }
        out
    }
}

impl Report for splinter_sdk::tasks::check_requests::RequestsChecked {
    fn human(&self) -> String {
        format!(
            "kept  {}  {} task(s)\ndropped  {} whose request states no returned value the fix restores, {} whose request states no error the fix restores\n",
            self.kept, self.kept_tasks, self.unstated_return_value, self.unstated_raised_error,
        )
    }
}

impl Report for Picked {
    fn human(&self) -> String {
        format!(
            "picked  {}  {} task(s) in {} family(ies)\nrest    {}  {} task(s) in {} family(ies)\n",
            self.picked,
            self.picked_tasks,
            self.picked_families,
            self.rest,
            self.rest_tasks,
            self.rest_families,
        )
    }
}

impl Report for splinter_sdk::tasks::combine::Combined {
    fn human(&self) -> String {
        use std::fmt::Write as _;

        let mut out = format!(
            "combined  {} task(s) of {} faults each (seed {}): {} of the exam's files, {} of \
             training files\n  tried     {} combination(s), refused {}\n",
            self.admitted,
            self.faults,
            self.seed.value(),
            self.exam_tasks,
            self.training_tasks,
            self.attempted,
            tally(&self.refused)
        );
        if !self.unusable.is_empty() {
            let _ = writeln!(out, "  unusable  {}", tally(&self.unusable));
        }
        for (label, set) in [
            ("set", &self.task_set),
            ("exam", &self.exam_task_set),
            ("training", &self.training_task_set),
        ] {
            if let Some(set) = set {
                let _ = writeln!(out, "  {label:<9} {set}");
            }
        }
        for (family, count) in &self.families {
            let _ = writeln!(out, "  {count:>3}  {family}");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::generator_hint;

    #[test]
    fn a_stage_where_a_model_proposed_and_nothing_passed_suggests_a_stronger_generator() {
        assert!(generator_hint(0, 2, true).unwrap().contains("--generator"));
    }

    #[test]
    fn no_hint_when_something_was_admitted_nothing_was_rejected_or_no_model_was_asked() {
        assert!(generator_hint(1, 2, true).is_none());
        assert!(generator_hint(0, 0, true).is_none());
        assert!(
            generator_hint(0, 2, false).is_none(),
            "code-built kinds have no generator to blame"
        );
    }
}
