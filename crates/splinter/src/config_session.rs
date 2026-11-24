// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements machine configuration that a team can edit
// from the command line and trust to stay valid, for its clients. If your
// team needs expertise in configuration tooling, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `splinter config`: runs without a session, since it only reads and writes
//! the machine file.

use std::path::Path;

use serde_json::{json, Value};
use splinter_sdk::factory::machine_config::{self as machine, ConfigError, MachineConfig};
use splinter_sdk::Error;

use crate::cli::config::ConfigCommand;
use crate::output;
use crate::session::Exit;

/// Runs `command` against the file at `path`, printing its result.
#[must_use]
pub fn run(path: &Path, json: bool, command: &ConfigCommand) -> Exit {
    match execute(path, json, command) {
        Ok(()) => Exit::Ok,
        Err(e) => {
            output::error(json, &Error::Refused(e.to_string()));
            Exit::Refused
        }
    }
}

/// A value for a person: a string as itself, anything else as JSON.
fn plain(value: &Value) -> String {
    match value {
        Value::String(text) => format!("{text}\n"),
        other => format!("{other:#}\n"),
    }
}

fn execute(path: &Path, as_json: bool, command: &ConfigCommand) -> Result<(), ConfigError> {
    match command {
        ConfigCommand::Show => {
            let config = MachineConfig::load(path)?;
            let exists = path.exists();
            if as_json {
                let shown = json!({ "path": path, "exists": exists, "config": config });
                println!("{shown:#}");
            } else {
                let note = if exists {
                    ""
                } else {
                    " (does not exist: the default)"
                };
                println!("# {}{note}\n{}", path.display(), config.render()?);
            }
        }
        ConfigCommand::Get { key } => {
            let value = machine::get(path, key)?;
            if as_json {
                println!("{value:#}");
            } else {
                print!("{}", plain(&value));
            }
        }
        ConfigCommand::Set { key, value } => {
            machine::set(path, key, value)?;
            println!("{key} set in {}", path.display());
        }
        ConfigCommand::Reset(args) => match &args.key {
            Some(key) if machine::reset(path, key)? => {
                println!("{key} removed from {}", path.display());
            }
            Some(key) => println!("{key} was not set in {}", path.display()),
            None => {
                machine::reset_all(path)?;
                println!("{} returned to the empty default", path.display());
            }
        },
        ConfigCommand::Schema => println!("{:#}", machine::schema()),
        ConfigCommand::Path => println!("{}", path.display()),
        ConfigCommand::Import => {
            let imported = machine::import(path)?;
            if as_json {
                println!("{:#}", json!(imported));
            } else if imported.removed.is_empty() {
                println!("nothing to import beside {}", path.display());
            } else {
                println!(
                    "imported {} connections and {} operators into {}; removed {}",
                    imported.connections.len(),
                    imported.operators.len(),
                    path.display(),
                    imported
                        .removed
                        .iter()
                        .map(|f| f.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
    }
    Ok(())
}
