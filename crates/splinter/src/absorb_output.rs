// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The reports of `session` and `claims`, as text for a person.

use std::fmt::Write as _;

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
