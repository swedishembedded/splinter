// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the stages that make and choose tasks print: the tasks stage with
//! what the generator or the invent harness yielded, and the validate stage.

use std::fmt::Write as _;

use splinter_sdk::canaries::CanariesMade;
use splinter_sdk::tasks::pick::Picked;
use splinter_sdk::variants::VariantsGenerated;

use crate::output::{tally, Report};

impl Report for splinter_sdk::tasks::drills::Drilled {
    fn human(&self) -> String {
        let mut out = format!(
            "drills  {} of {} function(s) (seed {}, at most {} each, tree {})\n  refused {}\n",
            self.admitted,
            self.functions,
            self.seed.value(),
            self.per_function,
            &self.tree_id[..self.tree_id.len().min(12)],
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
        if let Some(set) = &self.task_set {
            let _ = writeln!(out, "  set       {set}");
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
            "combined  {} task(s) of {} faults each (seed {})\n  tried     {} combination(s), refused {}\n",
            self.admitted,
            self.faults,
            self.seed.value(),
            self.attempted,
            tally(&self.refused)
        );
        if !self.unusable.is_empty() {
            let _ = writeln!(out, "  unusable  {}", tally(&self.unusable));
        }
        if let Some(set) = &self.task_set {
            let _ = writeln!(out, "  set       {set}");
        }
        for (family, count) in &self.families {
            let _ = writeln!(out, "  {count:>3}  {family}");
        }
        out
    }
}
