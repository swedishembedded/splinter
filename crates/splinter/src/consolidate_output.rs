// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The report of `consolidate`, and the lines `absorb` tells a consolidation
//! and a rebuild by, as text for a person.

use std::fmt::Write as _;

use splinter_sdk::absorb::consolidate::{Consolidated, Consolidation};
use splinter_sdk::absorb::generation::Generation;

use crate::output::Report;

/// The generation a policy is in, in one line.
pub fn generation_line(generation: &Generation) -> String {
    match &generation.base_release {
        Some(base) => format!(
            "generation {} on the checkpoint of release {base}, holding {} claim(s)\n",
            generation.epoch,
            generation.held.len()
        ),
        None => format!(
            "generation {} on the configured base, holding no claim\n",
            generation.epoch
        ),
    }
}

/// What a consolidation did, line by line.
pub fn consolidated_lines(done: &Consolidated) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{} {} claim(s) of release {} into generation {} ({} already held{})",
        if done.rebuilt {
            "rebuilt"
        } else {
            "consolidating"
        },
        done.closing.len(),
        done.champion,
        done.epoch,
        done.retained.len(),
        if done.rebuilt {
            format!(", {} forgotten claim(s) taken out", done.dead.len())
        } else {
            String::new()
        }
    );
    if let Some(data) = &done.dataset {
        let _ = writeln!(
            out,
            "dataset {}: {} records ({} replayed from claims the base holds), {} stopping \
             paraphrases held out; {} step(s), loss {} -> {}",
            data.dataset,
            data.trained,
            done.replayed_records,
            data.held_out,
            done.steps_completed,
            done.initial_loss.map_or("?".into(), |l| format!("{l:.3}")),
            done.final_loss.map_or("?".into(), |l| format!("{l:.3}")),
        );
    }
    if let Some(gate) = &done.gate {
        let _ = writeln!(
            out,
            "gate: {} of {} claim(s) answered, {} regressed: {}",
            gate.answered,
            gate.claims.len(),
            gate.regressed.len(),
            if gate.passed { "passed" } else { "refused" }
        );
        for reason in [
            gate.retention.reason.as_ref(),
            gate.anchor.as_ref().and_then(|a| a.reason.as_ref()),
            gate.serve.reason.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let _ = writeln!(out, "  {reason}");
        }
    }
    if let Some(release) = &done.release {
        let _ = writeln!(
            out,
            "release {release}: the base of generation {}",
            done.epoch
        );
    }
    if let Some(why) = &done.stopped {
        let _ = writeln!(out, "stopped: {why}");
    }
    out
}

impl Report for Consolidation {
    fn human(&self) -> String {
        let mut out = String::new();
        if let Some(generation) = &self.generation {
            out.push_str(&generation_line(generation));
        }
        let _ = writeln!(out, "claims: {} live", self.live);
        if let Some(done) = &self.consolidated {
            out.push_str(&consolidated_lines(done));
        } else if let Some(why) = &self.stopped {
            let _ = writeln!(out, "stopped: {why}");
        }
        out
    }
}
