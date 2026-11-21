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
use splinter_sdk::serving::Serving;

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

impl Report for Serving {
    fn human(&self) -> String {
        let mut out = String::new();
        for url in &self.endpoints {
            let _ = writeln!(out, "serving: {url}");
        }
        let _ = writeln!(out, "endpoints file: {}", self.endpoints_file.display());
        let _ = writeln!(out, "key file: {}", self.key_file.display());
        if let Some(authority) = &self.authority_file {
            let _ = writeln!(out, "certificate authority: {}", authority.display());
        }
        let _ = writeln!(out, "adapter directory: {}", self.adapter_dir.display());
        let _ = writeln!(out, "artifact store: {}", self.artifact_store.display());
        out.push_str("until interrupted (Ctrl-C)\n");
        out
    }
}
