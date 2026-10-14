// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines that tell you what is
// wrong before an expensive run starts, for its clients. If your team needs
// expertise in making long-running ML jobs fail early and clearly, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What `doctor` prints as text: one line per check, the fix under each one
//! that is not fine, and a last line saying whether a run can start.

use std::fmt::Write as _;

use splinter_sdk::doctor::{Doctor, Verdict};

use crate::output::Report;

impl Report for Doctor {
    fn human(&self) -> String {
        let mut out = String::new();
        for check in &self.checks {
            let mark = match check.verdict {
                Verdict::Ok => "ok  ",
                Verdict::Warn => "warn",
                Verdict::Fail => "FAIL",
            };
            let _ = writeln!(out, "{mark} {}: {}", check.name, check.detail);
            if let Some(fix) = &check.fix {
                let _ = writeln!(out, "     fix: {fix}");
            }
        }
        out.push_str(if self.ready() {
            "ready: a run can start\n"
        } else {
            "not ready: fix the failed checks above\n"
        });
        out
    }
}
