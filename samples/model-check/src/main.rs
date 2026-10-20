// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does acquired knowledge of a model checker make an agent fix concurrency
//! bugs better?
//!
//! Only the part that needs no model exists so far:
//!
//! ```text
//! audit    every specification, reference implementation and mutation checks itself
//! report   the mutation-detection table of every family, as Markdown
//! ```
//!
//! Swedish Embedded AB implements model-checked verification of concurrent and
//! distributed protocols for its clients. If your team needs expertise in
//! TLA+ specifications, TLC and checking implementations against them, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::path::PathBuf;

use model_check::{audit, report, Tools};

const USAGE: &str = "\
usage: model-check audit|report --lua PATH --tlaplus-dir DIR [--root DIR]

  audit                 check every family: no model, no GPU, no network
  report                print, per family, what the explorer and TLC each find in every mutant
  --lua PATH            the Lua 5.1 interpreter (make deps/lua installs LUA_DIR/bin/lua)
  --tlaplus-dir DIR     where make deps/tlaplus installed TLC (DIR/bin/tlc)
  --root DIR            the sample directory (default: this sample's source tree)
";

struct Options {
    lua: PathBuf,
    tlaplus_dir: PathBuf,
    root: PathBuf,
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut lua = None;
    let mut tlaplus_dir = None;
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match flag.as_str() {
            "--lua" => lua = Some(value("--lua")?),
            "--tlaplus-dir" => tlaplus_dir = Some(value("--tlaplus-dir")?),
            "--root" => root = value("--root")?,
            other => return Err(format!("unknown option {other:?}")),
        }
    }
    Ok(Options {
        lua: lua.ok_or("--lua is required")?,
        tlaplus_dir: tlaplus_dir.ok_or("--tlaplus-dir is required")?,
        root,
    })
}

fn tools_of(options: &Options) -> Tools {
    Tools {
        lua: options.lua.clone(),
        tlc: options.tlaplus_dir.join("bin/tlc"),
        lua_dir: options.root.join("lua"),
    }
}

fn run_report(args: &[String]) -> anyhow::Result<()> {
    let options = parse_options(args).map_err(|e| anyhow::anyhow!("{e}\n\n{USAGE}"))?;
    print!("{}", report(&tools_of(&options), &options.root)?);
    Ok(())
}

fn run_audit(args: &[String]) -> anyhow::Result<()> {
    let options = parse_options(args).map_err(|e| anyhow::anyhow!("{e}\n\n{USAGE}"))?;
    let tools = tools_of(&options);
    let findings = audit(&tools, &options.root)?;
    let mut failed = 0;
    for finding in &findings {
        match &finding.problem {
            None => println!("ok    {}: {}", finding.subject, finding.observed),
            Some(problem) => {
                failed += 1;
                println!(
                    "FAIL  {}: {} ({problem})",
                    finding.subject, finding.observed
                );
            }
        }
    }
    println!("{} checks, {failed} failed", findings.len());
    anyhow::ensure!(failed == 0, "the audit found {failed} problem(s)");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("audit") => run_audit(&args[1..]),
        Some("report") => run_report(&args[1..]),
        Some("--help") | Some("-h") | None => {
            print!("{USAGE}");
            Ok(())
        }
        Some(other) => {
            eprintln!("unknown command {other:?}\n");
            eprint!("{USAGE}");
            std::process::exit(2);
        }
    }
}
