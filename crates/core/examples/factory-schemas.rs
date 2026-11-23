// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! Writes the JSON Schema of every learning-factory contract into the
//! directory given as the only argument (`make schemas/generate`).

use splinter_core::factory::schemas;

fn main() -> std::io::Result<()> {
    let Some(out) = std::env::args_os().nth(1) else {
        eprintln!("usage: factory-schemas <directory>");
        std::process::exit(2);
    };
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out)?;
    for (name, schema) in schemas::all() {
        let text = serde_json::to_string_pretty(&schema).map_err(std::io::Error::other)?;
        std::fs::write(out.join(schemas::file_name(name)), format!("{text}\n"))?;
    }
    Ok(())
}
