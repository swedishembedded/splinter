// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements machine configuration that a team can edit
// from the command line and trust to stay valid, for its clients. If your
// team needs expertise in configuration tooling, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The one file of this machine's settings,
//! `$XDG_CONFIG_HOME/splinter/config.yaml` (`splinter.config/v1`).
//!
//! ```yaml
//! schema: splinter.config/v1
//! connections:   # how models are reached (see `crate::connections`)
//!   brain: { protocol: openai, url: "http://127.0.0.1:8990/v1", token: "env:BRAIN_KEY" }
//! operators:     # what task repositories' operators may have (see `crate::grants`)
//!   exam.sealed: { read: [/srv/sealed], network: true, env: [BRAIN_KEY] }
//! ```
//!
//! The file is the typed [`MachineConfig`]: unknown keys are refused, and its
//! checks are [`Connections::validate`] and [`Grants::validate`], the rules
//! the sections always had. [`get`], [`set`], [`reset`], [`reset_all`] and
//! [`import`] are what `splinter config` runs; each write is validated as a
//! whole before it replaces the file, atomically, so a refused write leaves
//! the file byte-identical. [`schema`] is the JSON Schema of the file.
//!
//! Writers are not locked against each other: two concurrent `set`s each
//! replace the whole file, and the later one wins.
//!
//! The settings used to live in `connections.yaml` and `operators.yaml`
//! beside it. They are no longer read: with no `config.yaml`, their presence
//! is refused with the way to move them ([`import`]).

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::connections::{Connection, Connections};
use crate::grants::{Grant, Grants};

/// The schema name of the machine file.
pub const SCHEMA: &str = "splinter.config/v1";

/// The file name `$XDG_CONFIG_HOME/splinter/` holds.
pub const FILE_NAME: &str = "config.yaml";

/// The files this one replaces, which [`import`] moves into it.
const LEGACY_CONNECTIONS: &str = "connections.yaml";
const LEGACY_OPERATORS: &str = "operators.yaml";

/// What a file `splinter config` writes starts with.
const HEADER: &str =
    "# Managed by `splinter config`: edit it with `splinter config set`, `reset` and\n\
                      # `import`; every write is checked against `splinter config schema`.\n";

/// Why the machine file cannot be read, changed or written.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ConfigError(String);

fn refuse<T>(message: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError(message.into()))
}

/// The machine's settings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MachineConfig {
    /// Always `splinter.config/v1`.
    #[serde(default = "current_schema")]
    #[schemars(schema_with = "schema_name")]
    pub schema: String,
    /// How models are reached, by connection name.
    #[serde(default)]
    pub connections: BTreeMap<String, Connection>,
    /// What each task-repository operator is granted, by operator name.
    #[serde(default)]
    pub operators: BTreeMap<String, Grant>,
}

fn current_schema() -> String {
    SCHEMA.to_owned()
}

fn schema_name(_: &mut SchemaGenerator) -> Schema {
    json_schema!({ "const": SCHEMA })
}

impl Default for MachineConfig {
    fn default() -> Self {
        Self {
            schema: current_schema(),
            connections: BTreeMap::new(),
            operators: BTreeMap::new(),
        }
    }
}

impl MachineConfig {
    /// The configuration of `text`, checked.
    ///
    /// # Errors
    /// It is not the right shape, or a section's rules refuse it.
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let value: Value = serde_yaml::from_str(text).map_err(|e| ConfigError(e.to_string()))?;
        Self::from_value(value)
    }

    fn from_value(value: Value) -> Result<Self, ConfigError> {
        let config: Self = serde_json::from_value(value).map_err(|e| ConfigError(e.to_string()))?;
        config.check()?;
        Ok(config)
    }

    fn check(&self) -> Result<(), ConfigError> {
        if self.schema != SCHEMA {
            return refuse(format!("schema must be {SCHEMA}, not {}", self.schema));
        }
        Connections::validate(&self.connections).map_err(|e| ConfigError(e.to_string()))?;
        Grants::validate(&self.operators).map_err(|e| ConfigError(format!("operators: {e}")))
    }

    /// The configuration in the file at `path`; `None` when there is no
    /// file.
    ///
    /// # Errors
    /// The file cannot be read or is refused; it is absent while the older
    /// `connections.yaml` or `operators.yaml` beside it exist.
    pub fn read(path: &Path) -> Result<Option<Self>, ConfigError> {
        match read_file(path)? {
            Some(config) => Ok(Some(config)),
            None => match legacy_files(path).as_slice() {
                [] => Ok(None),
                found => refuse(format!(
                    "{} does not exist, but the older {} remain: this machine's settings are now one file; \
                     run `splinter config import` to move them into it",
                    path.display(),
                    found
                        .iter()
                        .map(|f| f.display().to_string())
                        .collect::<Vec<_>>()
                        .join(" and ")
                )),
            },
        }
    }

    /// The configuration in the file at `path`; the empty default when there
    /// is no file.
    ///
    /// # Errors
    /// As [`Self::read`].
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        Ok(Self::read(path)?.unwrap_or_default())
    }

    /// The file as `splinter config show` prints it: the configuration as
    /// YAML, `schema` first, with nothing unset written.
    ///
    /// # Errors
    /// Never in practice: a configuration is plain data.
    pub fn render(&self) -> Result<String, ConfigError> {
        let Value::Object(mut sections) = self.to_value()? else {
            return refuse("the configuration is not a mapping");
        };
        let mut ordered = serde_yaml::Mapping::new();
        for key in ["schema", "connections", "operators"] {
            if let Some(section) = sections.remove(key) {
                ordered.insert(
                    key.into(),
                    serde_yaml::to_value(section).map_err(|e| ConfigError(e.to_string()))?,
                );
            }
        }
        serde_yaml::to_string(&ordered).map_err(|e| ConfigError(e.to_string()))
    }

    /// The configuration as JSON with every unset value left out.
    fn to_value(&self) -> Result<Value, ConfigError> {
        let mut value = serde_json::to_value(self).map_err(|e| ConfigError(e.to_string()))?;
        strip_nulls(&mut value);
        Ok(value)
    }
}

fn strip_nulls(value: &mut Value) {
    if let Value::Object(map) = value {
        map.retain(|_, v| !v.is_null());
        map.values_mut().for_each(strip_nulls);
    }
}

/// The file at `path` parsed and checked; `None` when it does not exist.
fn read_file(path: &Path) -> Result<Option<MachineConfig>, ConfigError> {
    match std::fs::read_to_string(path) {
        Ok(text) => MachineConfig::parse(&text)
            .map(Some)
            .map_err(|e| ConfigError(format!("{}: {e}", path.display()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => refuse(format!("{}: {e}", path.display())),
    }
}

/// The older files beside the default-named `path` that exist.
fn legacy_files(path: &Path) -> Vec<PathBuf> {
    if path.file_name().is_none_or(|name| name != FILE_NAME) {
        return Vec::new();
    }
    let beside = path.parent().unwrap_or_else(|| Path::new("."));
    [LEGACY_CONNECTIONS, LEGACY_OPERATORS]
        .iter()
        .map(|name| beside.join(name))
        .filter(|file| file.exists())
        .collect()
}

/// Replace the file at `path` with `config`, atomically (a temporary file in
/// its directory, renamed over it).
fn write(path: &Path, config: &MachineConfig) -> Result<(), ConfigError> {
    let failed = |e: std::io::Error| ConfigError(format!("{}: {e}", path.display()));
    let directory = path.parent().filter(|d| !d.as_os_str().is_empty());
    let directory = directory.unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(directory).map_err(failed)?;
    let mut temporary = tempfile::NamedTempFile::new_in(directory).map_err(failed)?;
    temporary
        .write_all(format!("{HEADER}{}", config.render()?).as_bytes())
        .map_err(failed)?;
    temporary.persist(path).map_err(|e| failed(e.error))?;
    Ok(())
}

/// A key path split into its names: `connections.hosted.url`, or
/// `operators[exam.sealed].network` for a name that holds dots.
fn parse_key(key: &str) -> Result<Vec<String>, ConfigError> {
    let bad = |why: &str| ConfigError(format!("key {key:?}: {why}"));
    let mut names = Vec::new();
    let mut chars = key.chars().peekable();
    let mut current = String::new();
    // Whether the next character may start a name (the start, or after a dot).
    let mut expect_name = true;
    while let Some(c) = chars.next() {
        match c {
            '.' => {
                if current.is_empty() {
                    return Err(bad("an empty name"));
                }
                names.push(std::mem::take(&mut current));
                expect_name = true;
            }
            '[' => {
                if !current.is_empty() {
                    names.push(std::mem::take(&mut current));
                }
                let mut quoted = String::new();
                loop {
                    match chars.next() {
                        Some(']') => break,
                        Some(q) => quoted.push(q),
                        None => return Err(bad("a `[` without its `]`")),
                    }
                }
                if quoted.is_empty() {
                    return Err(bad("an empty [name]"));
                }
                names.push(quoted);
                if chars.peek().is_some_and(|n| *n != '.' && *n != '[') {
                    return Err(bad("a name after `]` must follow a `.`"));
                }
                if chars.peek() == Some(&'.') {
                    chars.next();
                    expect_name = true;
                    if chars.peek().is_none() {
                        return Err(bad("ends in a `.`"));
                    }
                } else {
                    expect_name = false;
                }
            }
            ']' => return Err(bad("a `]` without its `[`")),
            other => {
                current.push(other);
                expect_name = false;
            }
        }
    }
    if !current.is_empty() {
        names.push(current);
    } else if names.is_empty() || expect_name {
        return Err(bad("ends in a `.` or is empty"));
    }
    Ok(names)
}

/// The value at `key` of the file at `path` (the empty default when there is
/// no file).
///
/// # Errors
/// The file is refused, the key is malformed, or nothing is set there.
pub fn get(path: &Path, key: &str) -> Result<Value, ConfigError> {
    let names = parse_key(key)?;
    let root = MachineConfig::load(path)?.to_value()?;
    names
        .iter()
        .try_fold(&root, |value, name| value.get(name))
        .cloned()
        .map_or_else(
            || refuse(format!("{key} is not set in {}", path.display())),
            Ok,
        )
}

/// Set `key` to `value`, parsed as YAML, creating what lies on the way; the
/// file changes only when the result is a valid configuration.
///
/// # Errors
/// The file is refused, the key or value is malformed, or the result is not
/// a valid configuration.
pub fn set(path: &Path, key: &str, value: &str) -> Result<(), ConfigError> {
    let names = parse_key(key)?;
    let parsed: Value = serde_yaml::from_str(value)
        .map_err(|e| ConfigError(format!("value {value:?} is not YAML: {e}")))?;
    if parsed.is_null() {
        return refuse(format!(
            "{key}: a value is required (`splinter config reset {key}` removes it)"
        ));
    }
    let mut root = MachineConfig::load(path)?.to_value()?;
    let (last, parents) = names
        .split_last()
        .unwrap_or_else(|| unreachable!("a key has a name"));
    let mut at = &mut root;
    for name in parents {
        at = at
            .as_object_mut()
            .ok_or_else(|| ConfigError(format!("{key}: {name} cannot hold values")))?
            .entry(name.clone())
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
    }
    at.as_object_mut()
        .ok_or_else(|| ConfigError(format!("{key}: the value on the way cannot hold {last}")))?
        .insert(last.clone(), parsed);
    let config = MachineConfig::from_value(root).map_err(|e| ConfigError(format!("{key}: {e}")))?;
    write(path, &config)
}

/// Remove `key`; whether it was set. The file changes only when the result is
/// a valid configuration.
///
/// # Errors
/// The file is refused, the key is malformed, or removing it leaves an
/// invalid configuration.
pub fn reset(path: &Path, key: &str) -> Result<bool, ConfigError> {
    let names = parse_key(key)?;
    let mut root = MachineConfig::load(path)?.to_value()?;
    let (last, parents) = names
        .split_last()
        .unwrap_or_else(|| unreachable!("a key has a name"));
    let mut at = Some(&mut root);
    for name in parents {
        at = at.and_then(|value| value.get_mut(name));
    }
    let removed = at
        .and_then(Value::as_object_mut)
        .and_then(|map| map.remove(last))
        .is_some();
    if removed {
        let config =
            MachineConfig::from_value(root).map_err(|e| ConfigError(format!("{key}: {e}")))?;
        write(path, &config)?;
    }
    Ok(removed)
}

/// Return the file at `path` to the empty default.
///
/// # Errors
/// The file cannot be written.
pub fn reset_all(path: &Path) -> Result<(), ConfigError> {
    write(path, &MachineConfig::default())
}

/// The JSON Schema of the machine file.
#[must_use]
pub fn schema() -> Value {
    serde_json::to_value(schemars::schema_for!(MachineConfig))
        .unwrap_or_else(|_| unreachable!("a schema is plain data"))
}

/// What [`import`] moved.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Imported {
    /// The connections found in the older `connections.yaml`.
    pub connections: Vec<String>,
    /// The operators found in the older `operators.yaml`.
    pub operators: Vec<String>,
    /// The older files removed.
    pub removed: Vec<PathBuf>,
}

/// Merge the older `connections.yaml` and `operators.yaml` beside `path`
/// into it, then remove them. A name both files define with different
/// content is refused and nothing changes. The configuration is written
/// before anything is removed, so an interrupted import is repeated safely.
///
/// # Errors
/// An older file or the merge is refused, or a file cannot be written or
/// removed.
pub fn import(path: &Path) -> Result<Imported, ConfigError> {
    let beside = path.parent().unwrap_or_else(|| Path::new("."));
    let (connections_file, operators_file) = (
        beside.join(LEGACY_CONNECTIONS),
        beside.join(LEGACY_OPERATORS),
    );
    let in_file =
        |file: &Path, e: &dyn std::fmt::Display| ConfigError(format!("{}: {e}", file.display()));
    let read = |file: &Path| match std::fs::read_to_string(file) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(in_file(file, &e)),
    };
    let mut config = read_file(path)?.unwrap_or_default();
    let mut imported = Imported::default();
    let mut conflicts = Vec::new();
    if let Some(text) = read(&connections_file)? {
        let found = Connections::parse(&text).map_err(|e| in_file(&connections_file, &e))?;
        for (name, connection) in found.connections {
            merge(&mut config.connections, &name, connection, &mut conflicts);
            imported.connections.push(name);
        }
        imported.removed.push(connections_file);
    }
    if let Some(text) = read(&operators_file)? {
        let found = Grants::parse(&text).map_err(|e| in_file(&operators_file, &e))?;
        for (name, grant) in found.entries() {
            merge(&mut config.operators, name, grant.clone(), &mut conflicts);
            imported.operators.push(name.clone());
        }
        imported.removed.push(operators_file);
    }
    if !conflicts.is_empty() {
        return refuse(format!(
            "{} already defines {} differently from the older files; \
             reset or rename them in one place, then import again",
            path.display(),
            conflicts.join(", ")
        ));
    }
    config.check()?;
    if imported.removed.is_empty() {
        return Ok(imported);
    }
    write(path, &config)?;
    for file in &imported.removed {
        std::fs::remove_file(file).map_err(|e| in_file(file, &e))?;
    }
    Ok(imported)
}

/// Add `item` as `name`, or record a conflict when `name` holds another.
fn merge<T: PartialEq>(
    into: &mut BTreeMap<String, T>,
    name: &str,
    item: T,
    conflicts: &mut Vec<String>,
) {
    match into.get(name) {
        Some(existing) if *existing != item => conflicts.push(name.to_owned()),
        Some(_) => {}
        None => {
            into.insert(name.to_owned(), item);
        }
    }
}
