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
use splinter_sdk::bench::examine::{Bench, Examiner};
use splinter_sdk::model::endpoint::BrainEndpoints;
use splinter_sdk::model::scratch::ServedBrain;
use splinter_sdk::model::served_train::ArtifactStore;
use splinter_sdk::model::service::ServiceSettings;

const USAGE: &str = "\
usage: splinter-minimind <run|status> [options]

  --run-dir DIR          the run directory (required); everything the run makes is written here
  --config FILE          minimind.yaml (default: the sample's own)
  --profile NAME         smoke | default | 64m, when the file names none (default smoke)
  --seed N               the run seed; drawn and printed on a fresh run, read from the
                         run's manifest afterwards; a different seed is a different run
  --cache-dir DIR        where fetched corpora are kept across runs (default RUN_DIR/cache)
  --brain-endpoint URL   a running `brain serve --training` (http://host:port/v1)
  --brain-artifact-store DIR  that server's artifact store (with --brain-endpoint)
  --brain-bin PATH       or a brain binary to start and own for the run
  --brain-commit REV     the brain commit the binary was built from, recorded in the manifest
  --device gpuN          the CUDA card that brain trains and serves on (required with
                         --brain-bin; there is no CPU fallback)
  --models-dir DIR       the started brain's model store (default RUN_DIR/brain/models)
  --scratch-dir DIR      where files that live for one episode or stage are made, in a private
                         directory removed when the run ends (default /dev/shm when present,
                         else RUN_DIR/scratch); the run directory keeps one pack per stage
  --gpu-lock FILE        held exclusively while a training job runs, shared with other users of
                         the card
  --brain-key KEY        the key a given endpoint takes (default: none needed)
  --bench DIR            a bench checkout: examine with its sealed workspace tool-use exam
  --bench-seed-file F    the exam's private seed file (never read by this program; drawn
                         into it by bench when missing)
  --bench-anchor-seed-file F  the anchor's own private seed file (likewise)
  --bench-exams-dir DIR  where bench generates the exam, relative to its checkout (default
                         .bench-data/exams; bench refuses one that resolves outside the checkout)
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
                | "brain-artifact-store"
                | "device"
                | "brain-commit"
                | "gpu-lock"
                | "scratch-dir"
                | "bench"
                | "bench-seed-file"
                | "bench-anchor-seed-file"
                | "bench-exams-dir"
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

/// The revision of the git checkout at `dir`; a tree with uncommitted
/// changes is marked with the digest of its diff, so other code is another
/// revision.
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
            match out(&["diff", "HEAD"]) {
                Some(diff) if !diff.stdout.is_empty() => {
                    let digest = splinter_sdk::vocabulary::digest::Digest::of(&diff.stdout);
                    format!("{head}+{}", &digest.hex()[..12])
                }
                _ => head,
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
    // The brain service: one given, or one this run starts on the card when
    // a stage first needs it and stops when it ends.
    let (brain, brain_revision) = match (flag("brain-endpoint"), flag("brain-bin")) {
        (Some(url), None) => {
            let store = flag("brain-artifact-store").context(
                "--brain-endpoint needs --brain-artifact-store DIR: the server's artifact store, which this run publishes datasets to",
            )?;
            let store = ArtifactStore::open(Path::new(&store)).map_err(|e| anyhow::anyhow!("{e}"))?;
            let endpoints = BrainEndpoints::new(vec![url], flag("brain-key").unwrap_or_default())
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_artifact_store(store);
            (ServedBrain::given(endpoints, context), "unknown (endpoint given)".to_string())
        }
        (None, Some(bin)) => {
            let device = flag("device").context(
                "--brain-bin needs --device gpuN: training runs on a CUDA card, never the CPU",
            )?;
            if !device
                .strip_prefix("gpu")
                .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
            {
                bail!("--device {device}: name one CUDA card, gpuN; training never runs on the CPU");
            }
            let state = run_dir.join("brain");
            let settings = ServiceSettings {
                binary: PathBuf::from(&bin),
                models: flag("models-dir").map_or_else(|| state.join("models"), PathBuf::from),
                state,
                // Training reserves the card; making a checkpoint from nothing
                // (the tokenizer, the random weights) is host work and needs
                // the CPU schedulable beside it.
                device: Some(format!("{device},cpu")),
                backend: Some("cuda".into()),
                // As many sequences decoded together as episodes run at once.
                max_batch: Some(config.agent.concurrency),
                // A from-scratch run trains whole checkpoints, never adapters.
                serve_adapters: false,
            };
            std::fs::create_dir_all(&settings.models)?;
            // The binary's own digest: a rebuilt brain is other code.
            let file = std::fs::File::open(&bin).with_context(|| format!("reading {bin}"))?;
            let digest = splinter_sdk::vocabulary::digest::Digest::of_reader(file)
                .with_context(|| format!("reading {bin}"))?;
            // Started when a stage first needs it, and only on the card:
            // training runs there or not at all.
            let brain = ServedBrain::managed(settings, context, Some(&device));
            (brain, digest.to_string())
        }
        (Some(_), Some(_)) => bail!("give --brain-endpoint or --brain-bin, not both"),
        (None, None) => bail!(
            "needs a brain service: give --brain-bin PATH, or --brain-endpoint URL --brain-artifact-store DIR"
        ),
    };
    let bench = match flag("bench") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            Some(Bench {
                sven: PathBuf::from(flag("sven-bin").context("--bench needs --sven-bin")?),
                seed_file: PathBuf::from(
                    flag("bench-seed-file").context("--bench needs --bench-seed-file")?,
                ),
                anchor_seed_file: PathBuf::from(
                    flag("bench-anchor-seed-file")
                        .context("--bench needs --bench-anchor-seed-file")?,
                ),
                concurrency: config.agent.concurrency,
                exams: PathBuf::from(
                    flag("bench-exams-dir")
                        .unwrap_or_else(|| splinter_sdk::bench::examine::DEFAULT_EXAMS.into()),
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
    // Provenance only: the binary's digest is what keys depend on.
    if let Some(commit) = flag("brain-commit") {
        revisions.0.insert("brain-commit".into(), commit);
    }
    if let Some(b) = &bench {
        revisions.0.insert("bench".into(), git_revision(&b.dir));
    }
    let mut out = std::io::stdout();
    let examiner = bench.as_ref().map(|b| b as &dyn Examiner);
    let mut run = Run::open(run_dir, config, seed, revisions, &brain, examiner, &mut out)?;
    if let Some(cache) = flag("cache-dir") {
        run.cache = PathBuf::from(cache);
    }
    run.gpu_lock = flag("gpu-lock").map(PathBuf::from);
    if let Some(root) = flag("scratch-dir") {
        run.use_scratch_root(Path::new(&root))?;
    }
    let result = splinter_minimind::stages::run_all(&mut run);
    drop(brain);
    result
}
