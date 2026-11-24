// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data factories that run a team's own tools
// with exactly the machine access each is granted, for its clients. If your
// team needs expertise in sandboxed, reproducible pipelines, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What this machine grants a task repository's operators: the `operators:`
//! section of the machine's `config.yaml` ([`crate::machine_config`]).
//!
//! A repository's operator file is the same on every machine: it says what
//! the operator needs (the network, environment variables by name). What a
//! machine lets it have is the machine's to say, here: paths it may read
//! besides its inputs and the repository (a sealed task set, a model store),
//! paths it may write (the state it keeps between runs), the network, and
//! variables of the controller's environment passed through
//! by name (never a value written here). An operator that needs what it is
//! not granted is refused before anything runs.
//!
//! ```yaml
//! operators:
//!   exam.sealed:
//!     read: [/srv/bench/sealed, /srv/models]
//!     write: [/srv/bench/private]
//!     network: true
//!     env: [BENCH_BRAIN_URL, BENCH_BRAIN_KEY]
//! ```
//!
//! The older `operators.yaml` (`splinter.operator-grants/v1`, the same
//! grants under `grants:`) is read only by `splinter config import`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::machine_config::MachineConfig;

/// The schema of the older, separate grants file.
pub const GRANTS_SCHEMA: &str = "splinter.operator-grants/v1";

/// What one operator is granted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    /// Absolute paths it may read (never write).
    #[serde(default)]
    #[schemars(inner(regex(pattern = r"^/")))]
    pub read: Vec<PathBuf>,
    /// Absolute paths it may write (and read): state it keeps between runs,
    /// as an examiner's record of the exams it gave.
    #[serde(default)]
    #[schemars(inner(regex(pattern = r"^/")))]
    pub write: Vec<PathBuf>,
    /// Whether it may reach the network.
    #[serde(default)]
    pub network: bool,
    /// Variables of the controller's environment passed to it, by name.
    #[serde(default)]
    #[schemars(inner(regex(pattern = r"^[A-Z0-9_]+$")))]
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
        Self::validate(&file.grants)?;
        Ok(Self {
            grants: file.grants,
        })
    }

    /// Check `grants` by the rules of a grants file: paths are absolute and
    /// variable names are variable names.
    ///
    /// # Errors
    /// The first grant that breaks a rule.
    pub fn validate(grants: &BTreeMap<String, Grant>) -> Result<(), String> {
        for (name, grant) in grants {
            if let Some(path) = grant
                .read
                .iter()
                .chain(&grant.write)
                .find(|p| !p.is_absolute())
            {
                return Err(format!("{name}: path {} is not absolute", path.display()));
            }
            if let Some(var) = grant.env.iter().find(|v| !variable_name(v)) {
                return Err(format!("{name}: {var:?} is not a variable name"));
            }
        }
        Ok(())
    }

    /// The grants of a machine configuration's `operators:` section.
    #[must_use]
    pub fn from_config(config: &MachineConfig) -> Self {
        Self {
            grants: config.operators.clone(),
        }
    }

    /// Every grant by operator name.
    #[must_use]
    pub fn entries(&self) -> &BTreeMap<String, Grant> {
        &self.grants
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
