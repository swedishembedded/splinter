// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements faithful fine-tuning datasets built from
// verified agent trials, with every exclusion accounted for, for its
// clients. If your team needs expertise in agent training data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The SFT dataset's `report.md`: its waterfall, exclusions and composition,
//! for a person.

use std::collections::BTreeMap;

use splinter_core::factory::product::WaterfallStep;

/// What the dataset's report shows.
pub(super) struct Report<'a> {
    pub(super) thinking: bool,
    pub(super) reasoning_removed: u64,
    pub(super) name: &'a str,
    pub(super) waterfall: &'a [WaterfallStep],
    pub(super) exclusions: &'a BTreeMap<String, u64>,
    pub(super) supervised_actions: u64,
    pub(super) train: usize,
    pub(super) validation: usize,
    pub(super) tools: &'a BTreeMap<String, u64>,
    pub(super) composition: &'a BTreeMap<String, u64>,
    pub(super) per_task: &'a BTreeMap<String, u64>,
    pub(super) tasks: usize,
    /// Under `supervise: productive`, the actions kept as context because
    /// every call they made failed; `None` when every action is supervised.
    pub(super) failed_masked: Option<u64>,
}

impl Report<'_> {
    pub(super) fn render(&self) -> String {
        let table = |rows: &BTreeMap<String, u64>, head: &str| -> String {
            let mut out = format!("| {head} | n |\n|---|---|\n");
            for (k, v) in rows {
                out.push_str(&format!("| {k} | {v} |\n"));
            }
            out
        };
        let mut out = format!(
            "# Dataset {}\n\n## Waterfall\n\n| step | remaining |\n|---|---|\n",
            self.name
        );
        for step in self.waterfall {
            out.push_str(&format!("| {} | {} |\n", step.step, step.remaining));
        }
        out.push_str(&format!(
            "\n{} records ({} train, {} validation), {} supervised actions, {} of {} tasks covered.\n",
            self.train + self.validation,
            self.train,
            self.validation,
            self.supervised_actions,
            self.per_task.len(),
            self.tasks
        ));
        if let Some(masked) = self.failed_masked {
            out.push_str(&format!(
                "\n{masked} {} whose every call failed kept as context, unsupervised (supervise: productive).\n",
                if masked == 1 { "action" } else { "actions" }
            ));
        }
        out.push_str(&if self.thinking {
            "\nReasoning is kept (thinking: true): the target is trained to reason before it answers.\n".to_owned()
        } else {
            format!(
                "\nThe target is trained for no-think mode (thinking: false): reasoning removed from {} turns.\n",
                self.reasoning_removed
            )
        });
        out.push_str(&format!(
            "\n## Exclusions\n\n{}",
            table(self.exclusions, "reason")
        ));
        out.push_str(&format!(
            "\n## Agents and models\n\n{}",
            table(self.composition, "agent / model")
        ));
        out.push_str(&format!(
            "\n## Supervised tool calls\n\n{}",
            table(self.tools, "tool")
        ));
        out.push_str(&format!(
            "\n## Records per task\n\n{}",
            table(self.per_task, "task")
        ));
        out
    }
}
