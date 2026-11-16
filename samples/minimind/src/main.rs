// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements end-to-end learning systems that take a
// model from random weights to a working tool-using agent, for its clients.
// If your team needs expertise in training small agents from scratch, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `splinter-minimind`: the whole sequence, from random weights to an agent,
//! as one resumable command.
//!
//! ```text
//! run     every stage in order; committed stages are reused, interrupted ones resumed
//! status  what the ledger holds, with every probe's result; computes nothing
//! ```

use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _};
use splinter_minimind::config::{Config, Profile};
use splinter_minimind::stages::{Revisions, Run};
use splinter_sdk::agent::sven::CancelToken;
use splinter_sdk::bench::examine::{Bench, Examiner};
use splinter_sdk::model::scratch::ServedBrain;
use splinter_sdk::model::service::{ManagedBrain, ServiceSettings};

const USAGE: &str = "\
usage: splinter-minimind <run|status> [options]

  --run-dir DIR          the run directory (required); everything the run makes is written here
  --config FILE          minimind.yaml (default: the sample's own)
  --profile NAME         smoke | default | 64m, when the file names none (default smoke)
  --seed N               the run seed; drawn and printed on a fresh run, read from the
                         run's manifest afterwards; a different seed is a different run
  --cache-dir DIR        where fetched corpora are kept across runs (default RUN_DIR/cache)
  --brain-endpoint URL   a running `brain serve --training` (http://host:port/v1)
  --brain-bin PATH       or a brain binary to start and own for the run (on the CPU)
  --models-dir DIR       brain's model store for a started brain
  --brain-key KEY        the key a given endpoint takes (default: none needed)
  --bench DIR            a bench checkout: examine with its sealed workspace tool-use exam
  --bench-seed-file F    the exam's private seed file (never read by this program)
  --bench-key F          the verdict signing key
  --sven-bin PATH        the sven binary bench runs the exam's agent with
  --min-gain X           the gain a promotion must show (default 0.05)
";

/// The smallest gain a promotion must show, unless given.
const DEFAULT_MIN_GAIN: f64 = 0.05;

struct Args {
    command: String,
    flags: std::collections::BTreeMap<String, String>,
}

fn parse(mut args: Vec<String>) -> anyhow::Result<Args> {
    if args.is_empty() {
        bail!("{USAGE}");
    }
    let command = args.remove(0);
    let mut flags = std::collections::BTreeMap::new();
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        let name = flag
            .strip_prefix("--")
            .with_context(|| format!("unexpected {flag:?}\n{USAGE}"))?;
        let value = it
            .next()
            .with_context(|| format!("--{name} takes a value"))?;
        if !matches!(
            name,
            "run-dir"
                | "config"
                | "profile"
                | "seed"
                | "cache-dir"
                | "brain-endpoint"
                | "brain-bin"
                | "models-dir"
                | "brain-key"
                | "bench"
                | "bench-seed-file"
                | "bench-key"
                | "sven-bin"
                | "min-gain"
        ) {
            bail!("unknown option --{name}\n{USAGE}");
        }
        flags.insert(name.to_string(), value);
    }
    Ok(Args { command, flags })
}

fn sample_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The run seed: the one given, else the one the run's manifest recorded,
/// else a fresh draw (printed).
fn seed(run_dir: &Path, given: Option<u64>) -> anyhow::Result<u64> {
    if given.is_some() {
        return Ok(splinter_sdk::seed::resolve("seed", given));
    }
    let manifest = run_dir.join("manifest.json");
    if let Ok(text) = std::fs::read_to_string(&manifest) {
        let doc: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", manifest.display()))?;
        if let Some(seed) = doc["seed"].as_u64() {
            return Ok(seed);
        }
    }
    Ok(splinter_sdk::seed::resolve("seed", None))
}

/// The revision of the git checkout at `dir`, marked when it has changes.
fn git_revision(dir: &Path) -> String {
    let out = |args: &[&str]| {
        std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
    };
    match out(&["rev-parse", "HEAD"]) {
        Some(head) => {
            let head = String::from_utf8_lossy(&head.stdout).trim().to_string();
            let dirty = out(&["status", "--porcelain", "--untracked-files=no"])
                .is_some_and(|o| !o.stdout.is_empty());
            if dirty {
                format!("{head}+changes")
            } else {
                head
            }
        }
        None => "unknown".into(),
    }
}

/// The sven revision this program was built with, from the lock it was
/// built against.
fn sven_revision() -> String {
    let lock = include_str!("../../../Cargo.lock");
    lock.split("[[package]]")
        .find(|p| p.contains("name = \"sven-sdk\""))
        .and_then(|p| p.lines().find_map(|l| l.strip_prefix("source = \"")))
        .and_then(|s| s.rsplit('#').next())
        .map_or("unknown".into(), |r| r.trim_end_matches('"').to_string())
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("splinter-minimind: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> anyhow::Result<()> {
    let args = parse(std::env::args().skip(1).collect())?;
    let flag = |name: &str| args.flags.get(name).cloned();
    let run_dir = PathBuf::from(flag("run-dir").context("--run-dir is required")?);
    let profile = Profile::parse(&flag("profile").unwrap_or_else(|| "smoke".into()))?;
    let config_path =
        flag("config").map_or_else(|| sample_dir().join("minimind.yaml"), PathBuf::from);
    let config = Config::load(&config_path, profile)?;
    match args.command.as_str() {
        "status" => {
            let ledger = splinter_sdk::store::stage_ledger::StageLedger::open(&run_dir)?;
            print!("{}", splinter_minimind::report::status(&ledger.commits()?));
            Ok(())
        }
        "run" => run(&run_dir, config, &flag),
        other => bail!("unknown command {other:?}\n{USAGE}"),
    }
}

fn run(
    run_dir: &Path,
    config: Config,
    flag: &dyn Fn(&str) -> Option<String>,
) -> anyhow::Result<()> {
    let given_seed = flag("seed")
        .map(|s| s.parse::<u64>())
        .transpose()
        .context("--seed is a whole number")?;
    let seed = seed(run_dir, given_seed)?;
    let context = config.models.values().map(|m| m.context).max().unwrap_or(0);
    let cancel = CancelToken::new();
    // The brain service: one given, or one this run starts on the CPU and
    // stops when it ends.
    let (managed, base_url, key, brain_revision) = match (flag("brain-endpoint"), flag("brain-bin"))
    {
        (Some(url), None) => (
            None,
            url,
            flag("brain-key").unwrap_or_default(),
            "unknown (endpoint given)".to_string(),
        ),
        (None, Some(bin)) => {
            let models =
                PathBuf::from(flag("models-dir").context("--brain-bin needs --models-dir")?);
            let settings = ServiceSettings {
                binary: PathBuf::from(&bin),
                models,
                state: run_dir.join("brain"),
                device: Some("cpu".into()),
                backend: Some("cpu".into()),
                max_batch: None,
            };
            let version = std::process::Command::new(&bin)
                .arg("--version")
                .output()
                .map_or("unknown".into(), |o| {
                    String::from_utf8_lossy(&o.stdout).trim().to_string()
                });
            let brain = ManagedBrain::start(&settings, &cancel)
                .map_err(|e| anyhow::anyhow!("starting brain: {e}"))?;
            let url = brain
                .endpoints()
                .urls()
                .first()
                .cloned()
                .context("the started brain has no endpoint")?;
            let key = brain.endpoints().key().to_string();
            (Some(brain), url, key, version)
        }
        (Some(_), Some(_)) => bail!("give --brain-endpoint or --brain-bin, not both"),
        (None, None) => bail!(
            "needs a brain service: give --brain-endpoint URL or --brain-bin PATH --models-dir DIR"
        ),
    };
    let brain = ServedBrain {
        base_url,
        context_tokens: context,
        key,
    };
    let bench = match flag("bench") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            Some(Bench {
                sven: PathBuf::from(flag("sven-bin").context("--bench needs --sven-bin")?),
                seed_file: PathBuf::from(
                    flag("bench-seed-file").context("--bench needs --bench-seed-file")?,
                ),
                signing_key: PathBuf::from(flag("bench-key").context("--bench needs --bench-key")?),
                min_gain: flag("min-gain")
                    .map(|g| g.parse())
                    .transpose()
                    .context("--min-gain is a number")?
                    .unwrap_or(DEFAULT_MIN_GAIN),
                serving_build: brain_revision.clone(),
                dir,
            })
        }
        None => None,
    };
    let mut revisions = Revisions::default();
    revisions
        .0
        .insert("splinter".into(), git_revision(&sample_dir()));
    revisions.0.insert("sven".into(), sven_revision());
    revisions.0.insert("brain".into(), brain_revision);
    if let Some(b) = &bench {
        revisions.0.insert("bench".into(), git_revision(&b.dir));
    }
    let mut out = std::io::stdout();
    let examiner = bench.as_ref().map(|b| b as &dyn Examiner);
    let mut run = Run::open(run_dir, config, seed, revisions, &brain, examiner, &mut out)?;
    if let Some(cache) = flag("cache-dir") {
        run.cache = PathBuf::from(cache);
    }
    let result = splinter_minimind::stages::run_all(&mut run);
    drop(managed);
    result
}
