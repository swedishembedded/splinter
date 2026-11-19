// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training datasets whose every record says how
// faithfully it reproduces what the model was served, for its clients. If your
// team needs expertise in building verifiable fine-tuning data, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What `dataset build` prints as text.

use std::fmt::Write as _;

use splinter_sdk::datasets::{BlockMeasure, Built, GroupKind};

use crate::output::{tally, word, Report};

impl Report for Built {
    fn human(&self) -> String {
        let mut out = format!(
            "dataset {}: {} {} record(s), objective {}, format {}\n  file: {}\n  excluded: {}\n",
            self.dataset,
            self.records,
            self.view,
            word(&self.objective),
            word(&self.format),
            self.path.display(),
            tally(&self.excluded)
        );
        match &self.block_measure {
            Some(BlockMeasure::Trainer {
                base,
                first_refusal,
            }) => {
                let _ = writeln!(
                    out,
                    "  block: records counted as the trainer counts them, for the base {}",
                    base.display()
                );
                if let Some(reason) = first_refusal {
                    let _ = writeln!(
                        out,
                        "  block: a record the trainer would refuse was left out: {reason}"
                    );
                }
            }
            Some(BlockMeasure::Floor { reason }) => {
                let _ = writeln!(
                    out,
                    "  block: WARNING records counted by a floor on the trainer's count, which can still refuse a kept record: {reason}"
                );
            }
            None => {}
        }
        if !self.tokens.is_empty() {
            let _ = writeln!(
                out,
                "  sampled tokens: {} action(s) exact; proxy: {}",
                self.tokens.exact,
                tally(&self.tokens.proxy)
            );
        }
        if !self.groups.is_empty() {
            let count = |kind: GroupKind| self.groups.iter().filter(|g| g.kind == kind).count();
            let distilled = self.groups.iter().filter(|g| g.distilled).count();
            let _ = writeln!(
                out,
                "  groups: {} graded, {} all-fail, {} all-pass; {distilled} distilled with a teacher context",
                count(GroupKind::Graded),
                count(GroupKind::AllFail),
                count(GroupKind::AllPass)
            );
        }
        for group in &self.groups {
            let _ = writeln!(
                out,
                "  group {} v{} by {}: {} trial(s), {}{}",
                group.task,
                group.version,
                group.policy,
                group.trials.len(),
                group.kind.name(),
                match (group.signal, group.distilled) {
                    (_, true) => ", distilled",
                    (true, false) => "",
                    (false, false) => ", no signal",
                }
            );
            for trial in &group.trials {
                let _ = writeln!(
                    out,
                    "    {}: reward {:.3}{}",
                    trial.trial,
                    trial.reward,
                    if trial.written { "" } else { ", not written" }
                );
            }
        }
        out
    }
}
