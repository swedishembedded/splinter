// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn an interpreter's own
// execution of a program into reasoning a model can be trained on, for its
// clients. If your team needs expertise in verified-by-construction training
// signals for code models you can procure our services by sending an email to
// info@swedishembedded.com.

//! `exec-trace`: the execution trace of every program of a JSONL file.
//!
//! Each input line is an object with a `program` field (other fields are
//! carried through); each output line is that object plus `trace_status`
//! (`fits`, `too_long` or `failed`), and `trace` and `trace_output` when it
//! fits, `trace_chars` when it is too long, `trace_reason` when it failed.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Value};

use campaign_lua::exec_trace::{trace, Style, Traced};

#[derive(Parser)]
#[command(
    name = "exec-trace",
    about = "Deterministic execution traces of Lua 5.1 programs, one JSONL line each"
)]
struct Cli {
    /// The Lua 5.1 interpreter.
    #[arg(long)]
    lua: PathBuf,
    /// The rendering.
    #[arg(long, value_enum, default_value_t = Style::Raw)]
    style: Style,
    /// The longest trace kept, in characters; a longer one is skipped.
    #[arg(long, default_value_t = 6000)]
    max_chars: usize,
    /// The input, JSONL with a `program` field per line.
    #[arg(long)]
    input: PathBuf,
    /// The output, JSONL.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let input = std::fs::File::open(&cli.input)
        .with_context(|| format!("opening {}", cli.input.display()))?;
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&cli.output)
            .with_context(|| format!("creating {}", cli.output.display()))?,
    );
    let (mut fits, mut too_long, mut failed) = (0usize, 0usize, 0usize);
    for (n, line) in std::io::BufReader::new(input).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut record: Value = serde_json::from_str(&line)
            .with_context(|| format!("{}:{}", cli.input.display(), n + 1))?;
        let program = record["program"]
            .as_str()
            .with_context(|| format!("{}:{}: no \"program\" string", cli.input.display(), n + 1))?
            .to_string();
        let fields = match trace(&cli.lua, &program, cli.style, cli.max_chars)? {
            Traced::Fits { trace, output } => {
                fits += 1;
                json!({"trace_status": "fits", "trace": trace, "trace_output": output})
            }
            Traced::TooLong { chars } => {
                too_long += 1;
                json!({"trace_status": "too_long", "trace_chars": chars})
            }
            Traced::Failed { reason } => {
                failed += 1;
                json!({"trace_status": "failed", "trace_reason": reason})
            }
        };
        let object = record
            .as_object_mut()
            .with_context(|| format!("{}:{}: not an object", cli.input.display(), n + 1))?;
        object.extend(
            fields
                .as_object()
                .into_iter()
                .flatten()
                .map(|(k, v)| (k.clone(), v.clone())),
        );
        serde_json::to_writer(&mut out, &record)?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    eprintln!("exec-trace: {fits} fit, {too_long} too long, {failed} failed");
    Ok(())
}
