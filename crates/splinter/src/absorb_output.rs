// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The reports of `session` and `claims`, as text for a person.

use std::fmt::Write as _;

use splinter_sdk::absorb::Absorbed;
use splinter_sdk::claims::{ClaimSetList, ClaimsExtracted, ClaimsGated, LedgerReport};
use splinter_sdk::sessions::{SessionList, SessionsIntake};
use splinter_sdk::vocabulary::claim::ClaimSet;

use crate::output::{source_line, tally, Report};

impl Report for SessionsIntake {
    fn human(&self) -> String {
        let mut out = format!(
            "{} session(s) taken in, {} new, {} secret(s) removed\n",
            self.sessions.len(),
            self.new,
            self.redactions
        );
        for s in &self.sessions {
            let note = if s.new { "" } else { " (already stored)" };
            let _ = writeln!(out, "  {}  {} step(s)  {}{note}", s.source, s.steps, s.path);
        }
        for r in &self.refused {
            let _ = writeln!(out, "  refused {}: {}", r.path, r.reason);
        }
        out
    }
}

impl Report for SessionList {
    fn human(&self) -> String {
        if self.sessions.is_empty() {
            return "no sessions stored\n".into();
        }
        self.sessions
            .iter()
            .map(|s| source_line(s) + "\n")
            .collect()
    }
}

impl Report for ClaimsExtracted {
    fn human(&self) -> String {
        let mut out = format!(
            "claim set {}: {} proposal(s) from {} session(s) ({})\n",
            self.claim_set,
            self.proposals,
            self.sessions,
            tally(&self.by_kind)
        );
        for f in &self.failed {
            let _ = writeln!(out, "  nothing usable for {}: {}", f.session, f.reason);
        }
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "  stopped: {why}");
        }
        out
    }
}

impl Report for ClaimsGated {
    fn human(&self) -> String {
        let mut out = format!(
            "claim set {}: {} proposal(s), {} ruled now ({} already ruled)\n  admitted {} \
             (replacing {}), refused {} ({}), {} live\n",
            self.claim_set,
            self.proposals,
            self.ruled,
            self.already_ruled,
            self.admitted,
            self.superseded,
            self.ruled - self.admitted,
            tally(&self.refused),
            self.live
        );
        for r in &self.rulings {
            let _ = match &r.reason {
                Some(why) => writeln!(out, "  refused {:?}: {why}", r.statement),
                None => writeln!(out, "  admitted {:?}", r.statement),
            };
        }
        out
    }
}

impl Report for ClaimSetList {
    fn human(&self) -> String {
        if self.claim_sets.is_empty() {
            return "no claim sets stored\n".into();
        }
        self.claim_sets
            .iter()
            .map(|s| {
                format!(
                    "{}  {} proposal(s) from {} session(s), {} failed  {}\n",
                    s.id, s.proposals, s.sessions, s.failed, s.extractor
                )
            })
            .collect()
    }
}

impl Report for ClaimSet {
    fn human(&self) -> String {
        let mut out = format!("claim set by {}\n", self.extractor);
        for session in &self.sessions {
            let _ = writeln!(out, "session {}", session.session);
            if let Some(why) = &session.failure {
                let _ = writeln!(out, "  nothing usable: {why}");
            }
            for p in &session.proposals {
                let _ = writeln!(out, "  {} {:?}", p.kind.as_str(), p.statement);
                for q in &p.quotes {
                    let _ = writeln!(out, "    step {}: {:?}", q.step, q.text);
                }
            }
        }
        out
    }
}

impl Report for LedgerReport {
    fn human(&self) -> String {
        let mut out = format!(
            "{} live, {} superseded, {} refused\n",
            self.live.len(),
            self.superseded.len(),
            self.refused.len()
        );
        for c in &self.live {
            let _ = writeln!(out, "  live {} {:?}", c.claim, c.statement);
        }
        for c in &self.superseded {
            let _ = writeln!(
                out,
                "  superseded {} by {}: {:?}",
                c.claim, c.by, c.statement
            );
        }
        for r in &self.refused {
            let _ = writeln!(out, "  refused {:?}: {}", r.statement, r.reason);
        }
        out
    }
}

impl Report for Absorbed {
    fn human(&self) -> String {
        let mut out = String::new();
        if let Some(intake) = &self.intake {
            out.push_str(&intake.human());
        }
        if let Some(extracted) = &self.extract {
            out.push_str(&extracted.human());
        }
        if let Some(gated) = &self.claims {
            let _ = writeln!(
                out,
                "claims: {} admitted, {} superseded, {} live, {} new",
                gated.admitted,
                gated.superseded,
                self.live,
                self.pending.len()
            );
            for (reason, count) in &gated.refused {
                let _ = writeln!(out, "  refused {count}: {reason}");
            }
        }
        if let Some(kits) = &self.kits {
            let _ = writeln!(
                out,
                "kits: {} built, {} reused, {} answers verified ({} failed)",
                kits.built, kits.reused, kits.verified, kits.failed
            );
            for u in &kits.untaught {
                let _ = writeln!(out, "  not taught {}: {}", u.claim, u.reason);
            }
        }
        if let Some(data) = &self.dataset {
            let _ = writeln!(
                out,
                "dataset {}: {} records for {} claim(s), {} stopping paraphrases held out",
                data.dataset,
                data.trained,
                data.claims.len(),
                data.held_out
            );
            for r in &data.refused {
                let _ = writeln!(out, "  refused a record of {}: {}", r.claim, r.leak);
            }
        }
        if let Some(candidate) = &self.candidate {
            let _ = writeln!(
                out,
                "candidate {} (trained from {})",
                candidate.candidate, candidate.from
            );
        }
        if let Some(gate) = &self.gate {
            let _ = writeln!(
                out,
                "gate: {} of {} claim(s) answered, {} gained, {} regressed: {}",
                gate.answered,
                gate.claims.len(),
                gate.gained.len(),
                gate.regressed.len(),
                if gate.passed { "passed" } else { "refused" }
            );
            for reason in [
                gate.improvement.reason.as_ref(),
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
        if let Some(release) = &self.release {
            let _ = writeln!(
                out,
                "release {release}: absorbed {} claim(s)",
                self.absorbed.len()
            );
        }
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "stopped: {why}");
        }
        out
    }
}
