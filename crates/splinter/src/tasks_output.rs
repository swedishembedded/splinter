// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the stages that make and choose tasks print: the tasks stage with
//! what the generator or the invent harness yielded, and the validate stage.

use std::fmt::Write as _;

use splinter_sdk::tasks::{duration_words, TasksGenerated};
use splinter_sdk::validate::Validated;

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
        if let Some(invent) = &self.invent {
            let y = &invent.yielded;
            let _ = writeln!(
                out,
                "  invent: {} run(s) logged {} admitted, {} rejected, {} duplicate, {} failed; {} committed task(s) refused here",
                invent.runs.len(),
                y.admitted,
                y.rejected,
                y.duplicate,
                y.failed,
                invent.refused.len()
            );
            for refused in invent.refused.iter().take(REJECTIONS_SHOWN) {
                let _ = writeln!(
                    out,
                    "    {}: {}",
                    refused.directory.display(),
                    refused.reason
                );
            }
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

impl Report for Validated {
    fn human(&self) -> String {
        let mut out = format!(
            "{} of {} passing attempt(s) kept; {} discarded by a validator, {} validator run(s) without a decision\n",
            self.kept,
            self.passing,
            self.discarded.len(),
            self.undecided.len()
        );
        for note in self.discarded.iter().take(REJECTIONS_SHOWN) {
            let _ = writeln!(
                out,
                "  discarded {} ({}): {}",
                note.experience,
                note.focus,
                note.reasons.join("; ")
            );
        }
        for note in self.undecided.iter().take(REJECTIONS_SHOWN) {
            let _ = writeln!(
                out,
                "  no decision for {} ({}): {}",
                note.experience, note.focus, note.why
            );
        }
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "  stopped: {why}");
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
