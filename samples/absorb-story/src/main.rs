// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving what a nightly update keeps, replaces and drops, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The `splinter-absorb-story` command line: run a storyline one night at a
//! time, resumably, and report it.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use anyhow::{anyhow, Context as _};
use clap::{Parser, Subcommand};
use splinter_absorb_story::compare::{compare, Side};
use splinter_absorb_story::layout::{freeze, Layout};
use splinter_absorb_story::render::{results_json, results_markdown};
use splinter_absorb_story::runner::{next_run, step, Step, Tools};
use splinter_absorb_story::storylines::{by_name, NAMES};
use splinter_absorb_story::study::{load, Study};

/// The exit status of `status` while runs remain.
const PENDING: u8 = 10;

#[derive(Parser)]
#[command(
    about = "Scripted nights of `splinter absorb` over invented facts, and the evaluation that follows each"
)]
struct Cli {
    #[command(flatten)]
    device: splinter_sdk::device::DeviceFlags,
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// The storylines that can be run.
    List,
    /// Writes a storyline's sessions as ATIF files, night by night.
    Sessions {
        /// The storyline.
        storyline: String,
        /// The directory the nights are written under.
        #[arg(long)]
        out: PathBuf,
    },
    /// Takes every session of a storyline in with `splinter session add` (no model),
    /// to see that each is accepted and what is redacted.
    Check {
        /// The storyline.
        storyline: String,
        /// A scratch directory.
        #[arg(long)]
        work: PathBuf,
        /// The `splinter` binary.
        #[arg(long)]
        splinter: PathBuf,
    },
    /// Carries out the next run that is not done (run 0 asks the base).
    Step {
        /// The storyline.
        storyline: String,
        /// The directory the storyline's directory is made in.
        #[arg(long)]
        out: PathBuf,
        /// The `splinter` binary that carries out the nights.
        #[arg(long)]
        splinter: PathBuf,
    },
    /// Exits 0 when every run is done and 10 while runs remain.
    Status {
        /// The storyline.
        storyline: String,
        /// The directory the storyline's directory is in.
        #[arg(long)]
        out: PathBuf,
    },
    /// Writes `results.json` and `results.md` of the runs done.
    Report {
        /// The storyline.
        storyline: String,
        /// The directory the storyline's directory is in.
        #[arg(long)]
        out: PathBuf,
    },
    /// Two storylines side by side, each given as `name:run`.
    Compare {
        /// The directory the storylines' directories are in.
        #[arg(long)]
        out: PathBuf,
        /// The first side.
        a: String,
        /// The second side.
        b: String,
    },
}

fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    cli.device.apply()?;
    match cli.command {
        Cmd::List => {
            for name in NAMES {
                println!("{name}");
            }
        }
        Cmd::Sessions { storyline, out } => {
            let story = story(&storyline)?;
            for night in &story.nights {
                let dir = out.join(format!("night-{}", night.number));
                std::fs::create_dir_all(&dir)
                    .with_context(|| format!("creating {}", dir.display()))?;
                for session in &night.sessions {
                    session.script.write(&dir)?;
                }
            }
        }
        Cmd::Check {
            storyline,
            work,
            splinter,
        } => check(&storyline, &work, &splinter)?,
        Cmd::Step {
            storyline,
            out,
            splinter,
        } => {
            let story = story(&storyline)?;
            let layout = Layout::new(&out, &storyline);
            freeze(&story, &layout, &splinter)?;
            match step(&story, &layout, &Tools { splinter })? {
                Step::Complete => println!("{storyline}: every run is done"),
                Step::Ran(run) => println!("{storyline}: run {run} done"),
            }
        }
        Cmd::Status { storyline, out } => {
            let story = story(&storyline)?;
            return Ok(match next_run(&story, &Layout::new(&out, &storyline)) {
                None => {
                    println!("{storyline}: complete");
                    ExitCode::SUCCESS
                }
                Some(run) => {
                    println!("{storyline}: next run {run}");
                    ExitCode::from(PENDING)
                }
            });
        }
        Cmd::Report { storyline, out } => {
            let study = study(&storyline, &out)?;
            let layout = Layout::new(&out, &storyline);
            write(
                &layout.root().join("results.json"),
                &serde_json::to_string_pretty(&results_json(&study))?,
            )?;
            write(&layout.root().join("results.md"), &results_markdown(&study))?;
        }
        Cmd::Compare { out, a, b } => {
            let (sa, ra) = side(&out, &a)?;
            let (sb, rb) = side(&out, &b)?;
            let (json, md) = compare(
                &Side {
                    study: &sa,
                    run: ra,
                },
                &Side {
                    study: &sb,
                    run: rb,
                },
            );
            let stem = format!("compare-{}-{}", a.replace(':', "-"), b.replace(':', "-"));
            write(
                &out.join(format!("{stem}.json")),
                &serde_json::to_string_pretty(&json)?,
            )?;
            write(&out.join(format!("{stem}.md")), &md)?;
            println!("{md}");
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn story(name: &str) -> anyhow::Result<splinter_absorb_story::story::Storyline> {
    let story = by_name(name).map_err(|e| anyhow!(e))?;
    story
        .validate()
        .map_err(|e| anyhow!("storyline {name}: {e}"))?;
    Ok(story)
}

fn study(name: &str, out: &Path) -> anyhow::Result<Study> {
    load(&story(name)?, &Layout::new(out, name))
}

fn side(out: &Path, spec: &str) -> anyhow::Result<(Study, u32)> {
    let (name, run) = spec
        .split_once(':')
        .ok_or_else(|| anyhow!("{spec:?} is not name:run"))?;
    let run = run.parse().with_context(|| format!("run in {spec:?}"))?;
    Ok((study(name, out)?, run))
}

fn write(path: &Path, text: &str) -> anyhow::Result<()> {
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

fn check(storyline: &str, work: &Path, splinter: &Path) -> anyhow::Result<()> {
    let story = story(storyline)?;
    let state = work.join("state");
    for night in &story.nights {
        let dir = work.join(format!("night-{}", night.number));
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        for session in &night.sessions {
            session.script.write(&dir)?;
        }
        if night.sessions.is_empty() {
            continue;
        }
        let output = Command::new(splinter)
            .arg("--state")
            .arg(&state)
            .args(["--json", "session", "add"])
            .arg(&dir)
            .output()
            .with_context(|| format!("running {}", splinter.display()))?;
        println!(
            "night {}: {}",
            night.number,
            String::from_utf8_lossy(&output.stdout)
        );
        if !output.status.success() {
            eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        }
    }
    Ok(())
}
