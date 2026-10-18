// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible-experiment infrastructure for
// learning systems, for its clients. If your team needs expertise in freezing
// the exact software, models and hardware an evaluation ran on, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Collects the `baseline-v1` record from this process, the files named on
//! the command line and the machine, and prints it.

use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;

use splinter_sdk::vocabulary::baseline::{
    Baseline, BrainBinary, FileDigest, Gpu, Hardware, ModelIdentity, Seeds,
};
use splinter_sdk::vocabulary::build::BuildInfo;
use splinter_sdk::vocabulary::digest::Digest;

use crate::cli::BaselineArgs;
use crate::output::{emit, Report};
use crate::session::Exit;

impl Report for Baseline {
    fn human(&self) -> String {
        let mut out = String::new();
        let revisions = &self.build.revisions;
        for (name, rev) in [
            ("splinter", &revisions.splinter),
            ("sven", &revisions.sven),
            ("brain", &revisions.brain),
        ] {
            let _ = writeln!(out, "{name}: {}", rev.as_deref().unwrap_or("unknown"));
        }
        let _ = writeln!(out, "dirty: {:?}", self.build.dirty);
        for model in &self.models {
            let _ = writeln!(out, "model {}: {} files", model.id, model.files.len());
        }
        let _ = writeln!(out, "gpus: {}", self.hardware.gpus.len());
        out
    }
}

/// Prints the record; with `--require-clean` fails when it is unmet.
pub fn run(build: &BuildInfo, args: &BaselineArgs, json: bool) -> Result<Exit, String> {
    let mut baseline = Baseline::new(build.clone());
    baseline.rustc = option_env!("SPLINTER_RUSTC").map(str::to_owned);
    baseline.seeds = Seeds {
        exam: args.exam_seed,
        training: args.training_seed,
    };
    baseline.brain_binary = args.brain_bin.as_deref().map(brain_binary).transpose()?;
    baseline.models = args
        .models
        .iter()
        .map(|dir| model_identity(dir))
        .collect::<Result<_, _>>()?;
    baseline.hardware = hardware();
    emit(json, &baseline);
    let unmet = baseline.unmet();
    for reason in &unmet {
        eprintln!("splinter: baseline unmet: {reason}");
    }
    Ok(if args.require_clean && !unmet.is_empty() {
        Exit::Failed
    } else {
        Exit::Ok
    })
}

fn digest_file(path: &Path) -> Result<Digest, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Digest::sha256_of_reader(std::io::BufReader::new(file))
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn brain_binary(path: &Path) -> Result<BrainBinary, String> {
    let version = Command::new(path)
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string());
    Ok(BrainBinary {
        path: path.display().to_string(),
        digest: digest_file(path)?,
        version,
    })
}

/// Every regular file of `dir` (not its subdirectories) by digest, in name
/// order, and its tokenizer.
fn model_identity(dir: &Path) -> Result<ModelIdentity, String> {
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let files = names
        .into_iter()
        .map(|name| {
            let digest = digest_file(&dir.join(&name))?;
            Ok(FileDigest { name, digest })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let tokenizer = files
        .iter()
        .find(|file| file.name == "tokenizer.json")
        .map(|file| file.digest.clone());
    let id = dir.components().rev().take(2).collect::<Vec<_>>();
    let id = id
        .iter()
        .rev()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    Ok(ModelIdentity {
        id,
        files,
        tokenizer,
    })
}

fn hardware() -> Hardware {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let cpu_model = cpuinfo
        .lines()
        .find_map(|line| line.strip_prefix("model name"))
        .and_then(|rest| rest.split_once(':'))
        .map(|(_, name)| name.trim().to_string());
    let smi = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
        .unwrap_or_default();
    let (gpus, gpu_driver) = parse_gpus(&smi);
    Hardware {
        kernel: std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .ok()
            .map(|k| k.trim().to_string()),
        cpu_model,
        cpu_threads: std::thread::available_parallelism()
            .ok()
            .and_then(|n| u32::try_from(n.get()).ok()),
        gpu_driver,
        gpus,
    }
}

/// The GPUs and driver of `nvidia-smi --format=csv,noheader,nounits` with the
/// columns name, memory.total, driver_version. A line that does not parse is
/// left out, never turned into a zero.
fn parse_gpus(listing: &str) -> (Vec<Gpu>, Option<String>) {
    let mut driver = None;
    let gpus = listing
        .lines()
        .filter_map(|line| {
            let mut cols = line.split(',').map(str::trim);
            let (name, memory, version) = (cols.next()?, cols.next()?, cols.next()?);
            driver.get_or_insert_with(|| version.to_string());
            Some(Gpu {
                name: name.to_string(),
                memory_mib: memory.parse().ok()?,
            })
        })
        .collect();
    (gpus, driver)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gpus_and_the_driver_are_read_from_the_listing() {
        let (gpus, driver) = parse_gpus(
            "NVIDIA GH200 120GB, 97871, 580.95\nbroken line\nNVIDIA GH200 120GB, 97871, 580.95\n",
        );
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].memory_mib, 97871);
        assert_eq!(driver.as_deref(), Some("580.95"));
        assert_eq!(parse_gpus(""), (Vec::new(), None));
    }

    #[test]
    fn a_model_directory_is_recorded_by_file_with_its_tokenizer() {
        let dir = std::env::temp_dir().join(format!("baseline-{}/org/Model", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"tok").unwrap();
        let model = model_identity(&dir).unwrap();
        assert_eq!(model.id, "org/Model");
        let names: Vec<_> = model.files.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["config.json", "tokenizer.json"]);
        assert_eq!(model.tokenizer, Some(Digest::sha256_of(b"tok")));
        std::fs::remove_dir_all(dir.parent().unwrap().parent().unwrap()).unwrap();
    }
}
