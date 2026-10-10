// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Running a command: its standard output kept whole, its standard error
//! appended to a log and passed on line by line (a supervisor reading the
//! runner's stream sees what the command reports).

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{anyhow, Context as _};

/// What a finished command left.
#[derive(Clone, Debug)]
pub struct Finished {
    /// Everything it printed on standard output.
    pub stdout: String,
    /// Its exit status.
    pub exit: i32,
    /// Seconds it ran.
    pub seconds: f64,
}

/// Runs `command`, appending its standard error to `log`.
///
/// # Errors
/// It cannot start, the log cannot be written, or it was ended by a signal.
pub fn run(command: &mut Command, log: &Path) -> anyhow::Result<Finished> {
    let started = Instant::now();
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("starting {command:?}"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("no stdout pipe"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("no stderr pipe"))?;
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        stdout.read_to_string(&mut text).map(|_| text)
    });
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .with_context(|| format!("opening {}", log.display()))?;
    for line in BufReader::new(stderr).lines() {
        let line = line.context("reading standard error")?;
        eprintln!("{line}");
        writeln!(file, "{line}").with_context(|| format!("writing {}", log.display()))?;
    }
    let status = child.wait().context("waiting for the command")?;
    let stdout = reader
        .join()
        .map_err(|_| anyhow!("the standard output reader panicked"))?
        .context("reading standard output")?;
    let exit = status
        .code()
        .ok_or_else(|| anyhow!("{command:?} was ended by a signal"))?;
    Ok(Finished {
        stdout,
        exit,
        seconds: started.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_output_is_kept_and_standard_error_logged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("log");
        let mut command = Command::new("sh");
        command.args(["-c", "echo out; echo err >&2; exit 3"]);
        let finished = run(&mut command, &log).expect("runs");
        assert_eq!(finished.stdout, "out\n");
        assert_eq!(finished.exit, 3);
        assert_eq!(std::fs::read_to_string(&log).expect("log"), "err\n");
    }
}
