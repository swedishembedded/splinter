// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A separate process for the multi-process specs: it opens a database, does
//! one job and exits.
//!
//! `probe write ROOT RANK COUNT` records COUNT entities and flushes after each,
//! `probe write-then-die ROOT RANK COUNT` records them and aborts without a
//! flush, and `probe signal ROOT NAME NOTE` raises a signal.

use std::process::ExitCode;

use splinter_expdb::model::Entity;
use splinter_expdb::{Config, Database, WriterIdentity};

fn run(args: &[String]) -> splinter_expdb::Result<()> {
    let [mode, root, a, b] = args else {
        return Err(splinter_expdb::Error::invalid(
            "probe arguments",
            "expected MODE ROOT A B",
        ));
    };
    let db = Database::open(root, Config::default())?;
    match mode.as_str() {
        "write" | "write-then-die" => {
            let rank: u32 = a.parse().unwrap_or(0);
            let count: u64 = b.parse().unwrap_or(0);
            let mut collector = db.collector(&WriterIdentity::new("probe", "job", "host", rank))?;
            for n in 0..count {
                collector.put_entity(&Entity::new(
                    "probe",
                    serde_json::json!({"rank": rank, "n": n}),
                ))?;
                if mode == "write" {
                    collector.flush()?;
                }
            }
            if mode == "write-then-die" {
                std::process::abort();
            }
            Ok(())
        }
        "signal" => db.signal(a, b).map(|_| ()),
        other => Err(splinter_expdb::Error::invalid(
            "probe mode",
            format!("unknown mode `{other}`"),
        )),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("probe: {error}");
            ExitCode::FAILURE
        }
    }
}
