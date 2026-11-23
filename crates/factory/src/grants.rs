// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data factories that run a team's own tools
// with exactly the machine access each is granted, for its clients. If your
// team needs expertise in sandboxed, reproducible pipelines, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What this machine grants a task repository's operators
//! (`~/.config/splinter/operators.yaml`, `splinter.operator-grants/v1`).
//!
//! A repository's operator file is the same on every machine: it says what
//! the operator needs (the network, environment variables by name). What a
//! machine lets it have is the machine's to say, here: paths it may read
//! besides its inputs and the repository (a sealed task set, a model store),
//! the network, and variables of the controller's environment passed through
//! by name (never a value written here). An operator that needs what it is
//! not granted is refused before anything runs.
//!
//! ```yaml
//! schema: splinter.operator-grants/v1
//! grants:
//!   exam.sealed:
//!     read: [/srv/bench/sealed, /srv/models]
//!     network: true
//!     env: [BENCH_BRAIN_URL, BENCH_BRAIN_KEY]
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The schema of a grants file.
pub const GRANTS_SCHEMA: &str = "splinter.operator-grants/v1";

/// What one operator is granted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    /// Absolute paths it may read (never write).
    #[serde(default)]
    pub read: Vec<PathBuf>,
    /// Whether it may reach the network.
    #[serde(default)]
    pub network: bool,
    /// Variables of the controller's environment passed to it, by name.
    #[serde(default)]
    pub env: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GrantsFile {
    schema: String,
    #[serde(default)]
    grants: BTreeMap<String, Grant>,
}

/// Every operator's grant on this machine.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Grants {
    grants: BTreeMap<String, Grant>,
}

/// Whether `name` can name an environment variable.
fn variable_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

impl Grants {
    /// The grants of `text`.
    ///
    /// # Errors
    /// It is not a grants file, a path is not absolute, or a variable name
    /// is not one.
    pub fn parse(text: &str) -> Result<Self, String> {
        let file: GrantsFile = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
        if file.schema != GRANTS_SCHEMA {
            return Err(format!("not a {GRANTS_SCHEMA} file"));
        }
        for (name, grant) in &file.grants {
            if let Some(path) = grant.read.iter().find(|p| !p.is_absolute()) {
                return Err(format!(
                    "{name}: read path {} is not absolute",
                    path.display()
                ));
            }
            if let Some(var) = grant.env.iter().find(|v| !variable_name(v)) {
                return Err(format!("{name}: {var:?} is not a variable name"));
            }
        }
        Ok(Self {
            grants: file.grants,
        })
    }

    /// The grants of the file at `path`; none when there is no file.
    ///
    /// # Errors
    /// The file exists and cannot be read or is refused by [`Self::parse`].
    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// The grant of the operator `name`: nothing when none is written.
    #[must_use]
    pub fn of(&self, name: &str) -> Grant {
        self.grants.get(name).cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grant_names_paths_and_variables_never_values() {
        let grants = Grants::parse(
            "schema: splinter.operator-grants/v1\ngrants:\n  exam:\n    read: [/srv/sealed]\n    network: true\n    env: [BRAIN_KEY]\n",
        )
        .unwrap();
        assert!(grants.of("exam").network);
        assert_eq!(grants.of("other"), Grant::default());
        assert!(Grants::parse(
            "schema: splinter.operator-grants/v1\ngrants:\n  x: { read: [rel] }\n"
        )
        .is_err());
        assert!(Grants::parse(
            "schema: splinter.operator-grants/v1\ngrants:\n  x: { env: [\"KEY=sk\"] }\n"
        )
        .is_err());
    }
}
