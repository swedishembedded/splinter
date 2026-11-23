// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running untrusted agents and
// their tools confined on any Linux host, container or not, for its clients.
// If your team needs expertise in kernel-level sandboxing then you can
// procure our services by sending an email to info@swedishembedded.com.

//! `splinter-confine`: run one command confined by the kernel, with no
//! privilege and no namespace - the confinement a sandbox adds where the
//! host gives neither containers nor user namespaces (a hardened container).
//!
//! ```text
//! splinter-confine --probe
//! splinter-confine [--read-only PATH]... [--read-write PATH]...
//!                  [--network open|none|tcp:PORT[,PORT...]] -- PROGRAM [ARG]...
//! ```
//!
//! `--probe` prints what this kernel enforces as JSON: `landlock_abi` (or
//! null) and `network` (whether TCP can be confined, Landlock ABI 4).
//! Otherwise the command may read and run what `--read-only` names, write
//! what `--read-write` names, and nothing else of the filesystem
//! ([`splinter_sandbox::landlock`]); `--network none` (the default) gives it
//! no internet socket at all, `tcp:PORT` only TCP `connect` to those ports
//! over IPv4 (no UDP, no listening), `open` the host's network. The
//! confinement is for good: the program is executed in this process after
//! it is enforced, so everything it starts is confined too. A kernel that
//! cannot enforce what was asked refuses (exit 126) rather than run the
//! command less confined; a malformed command line exits 2.

use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

use splinter_sandbox::landlock::{abi_version, Confinement};

const USAGE: &str =
    "usage: splinter-confine --probe | [--read-only PATH]... [--read-write PATH]... \
                     [--network open|none|tcp:PORT[,PORT...]] -- PROGRAM [ARG]...";
/// The Landlock ABI that confines TCP.
const NET_ABI: u32 = 4;

fn usage(problem: &str) -> ExitCode {
    eprintln!("splinter-confine: {problem}\n{USAGE}");
    ExitCode::from(2)
}

/// The confinement and the command a command line asks for.
fn parse(args: &[String]) -> Result<(Confinement, Vec<String>), String> {
    let mut confinement = Confinement::default();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let mut value = || {
            rest.next()
                .cloned()
                .ok_or_else(|| format!("{arg} needs a value"))
        };
        match arg.as_str() {
            "--read-only" => confinement.read_only.push(PathBuf::from(value()?)),
            "--read-write" => confinement.read_write.push(PathBuf::from(value()?)),
            "--network" => match value()?.as_str() {
                "open" => confinement.allow_network = true,
                "none" => confinement.allow_network = false,
                other => {
                    let ports = other.strip_prefix("tcp:").ok_or_else(|| {
                        format!("network {other:?} is not open, none or tcp:PORT")
                    })?;
                    for port in ports.split(',') {
                        confinement.tcp_connect_ports.push(
                            port.parse()
                                .map_err(|_| format!("{port:?} is not a TCP port"))?,
                        );
                    }
                }
            },
            "--" => {
                let command: Vec<String> = rest.cloned().collect();
                if command.is_empty() {
                    return Err("no program after --".into());
                }
                return Ok((confinement, command));
            }
            other => return Err(format!("unknown option {other:?}")),
        }
    }
    Err("no program: give it after --".into())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["--probe"] {
        let abi = abi_version();
        println!(
            "{}",
            serde_json::json!({ "landlock_abi": abi, "network": abi.is_some_and(|a| a >= NET_ABI) })
        );
        return ExitCode::SUCCESS;
    }
    let (confinement, command) = match parse(&args) {
        Ok(parsed) => parsed,
        Err(problem) => return usage(&problem),
    };
    let prepared = match confinement.prepare() {
        Ok(prepared) => prepared,
        Err(e) => {
            eprintln!("splinter-confine: this kernel cannot confine the command as asked: {e}");
            return ExitCode::from(126);
        }
    };
    if let Err(e) = prepared.enforce() {
        eprintln!("splinter-confine: the confinement could not be enforced: {e}");
        return ExitCode::from(126);
    }
    let error = Command::new(&command[0]).args(&command[1..]).exec();
    eprintln!("splinter-confine: {}: {error}", command[0]);
    ExitCode::from(127)
}
