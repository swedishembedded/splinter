// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements audit trails for released risk models, from
// the release back to the raw source file of every participant it was trained
// on, for its clients. If your team needs expertise in model provenance for
// regulated prediction, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `lineage`: a release traced back to the bytes it was made from.

use anyhow::Result;
use splinter_sdk::timeline::lineage::{timeline_lineage, TimelineLineage};

use crate::state::Run;

/// The lineage of `release`, listing `episodes` episodes per dataset.
pub fn lineage(run: &Run, release: &str, episodes: usize) -> Result<TimelineLineage> {
    Ok(timeline_lineage(&run.context()?, release, episodes)?)
}

/// The lineage as text, one level of the chain per indent.
pub fn render(l: &TimelineLineage) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "release {}", l.release);
    let _ = writeln!(
        out,
        "  distribution {:?}; terms {} (training {:?}, commercial use {:?}, redistribution {:?})",
        l.distribution,
        l.terms.name,
        l.terms.training,
        l.terms.commercial_use,
        l.terms.redistribution
    );
    if let Some(parent) = &l.parent {
        let _ = writeln!(out, "  continues release {parent}");
    }
    let _ = writeln!(out, "  training run: candidate {}", l.candidate);
    let _ = writeln!(out, "    checkpoint sha256 {}", l.checkpoint);
    let _ = writeln!(out, "    configuration {} seed {}", l.config, l.seed);
    let _ = writeln!(out, "    split {}", l.split);
    let _ = writeln!(
        out,
        "    brain {}; splinter {}",
        l.brain_commit.as_deref().unwrap_or("not recorded"),
        l.splinter_commit.as_deref().unwrap_or("not recorded")
    );
    match &l.calibration {
        Some(c) => {
            let _ = writeln!(
                out,
                "    calibration sha256 {} fitted on {} validation units of {} (early stopping read the other {}); uncalibrated: {}",
                c.digest,
                c.units,
                c.validation,
                c.early_stopping_units,
                if c.uncalibrated.is_empty() {
                    "none".to_owned()
                } else {
                    format!("{:?}", c.uncalibrated)
                }
            );
        }
        None => {
            let _ = writeln!(out, "    served raw: no calibration");
        }
    }
    for split in &l.evaluation_splits {
        let _ = writeln!(out, "    judged on the held-out units {split}");
    }
    for d in &l.datasets {
        let part = d.part.map_or("unsplit", |p| p.name());
        let _ = writeln!(
            out,
            "  dataset snapshot {} ({part}, {} records, file digest {})",
            d.id, d.records, d.snapshot
        );
        for (file, n) in &d.files {
            let _ = writeln!(
                out,
                "    source file {} digest {}: {n} episodes",
                file.dataset, file.file
            );
        }
        for e in &d.listed {
            let _ = writeln!(
                out,
                "    episode {} participant {} <- {} line {} of file {}",
                e.episode, e.participant, e.dataset, e.line, e.file
            );
        }
        if d.episodes > d.listed.len() {
            let _ = writeln!(out, "    ... {} more episodes", d.episodes - d.listed.len());
        }
    }
    out
}
