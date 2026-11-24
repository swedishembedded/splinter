// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data factories that run a team's own tools
// as reproducible, sandboxed pipeline steps, for its clients. If your team
// needs expertise in turning scripts into reliable pipeline operations, you
// can procure our services by sending an email to info@swedishembedded.com.

//! External operators (`splinter.operator/v1`): a task repository's own
//! programs as operations of its recipes.
//!
//! A repository declares one in `operators/NAME.yaml`: its ports, a JSON
//! Schema of its parameters, the command that runs it and the code it is
//! made of (`code`, repository paths, pinned: a change to them is a change
//! of every product it made). The controller runs the command in the process
//! sandbox with the request it is given (`--request FILE`, a
//! `splinter.operator-request/1` naming its parameters, its inputs and the
//! directory it writes into): it reads its inputs, the request, its own file
//! and its code (what its identity pins, so a cached product never depends on
//! a repository file whose change would not invalidate it) and, for a `./`
//! program outside its code, that program's top directory (the machine's
//! environment, as a virtualenv, unpinned like a program on PATH); it writes
//! only its output directory, and reaches the network only when it declares
//! `network: true`. It exits 0 having written its
//! product, 3 to reject its input (the reason on its last line of standard
//! output), anything else when it failed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_core::digest::{digest_of, Digest};
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::operation::{
    OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_sandbox::{CommandCall, Limits, ProcessSandbox, Sandbox};

use crate::execute::{Call, Execute, Outcome};
use crate::grants::{Grant, Grants};
use crate::registry::{OperationSpec, Registry};
use crate::tree::code_digest;

/// The schema of an operator file.
pub const OPERATOR_SCHEMA: &str = "splinter.operator/v1";
/// The schema of the request an operator is given.
pub const REQUEST_SCHEMA: &str = "splinter.operator-request/1";
/// Where a repository keeps its operators.
pub const OPERATORS_DIR: &str = "operators";
/// The exit code of an operator that rejects its input.
pub const REJECTED: i32 = 3;

fn hour() -> u64 {
    3600
}

/// An operator file as written.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorFile {
    /// Always [`OPERATOR_SCHEMA`].
    pub schema: String,
    /// Its name (dotted plain names).
    pub name: String,
    /// Its revision: raise it when what it computes changes.
    pub revision: u32,
    /// What it does, in a line.
    pub summary: String,
    /// Its input ports.
    #[serde(default)]
    pub inputs: Vec<Port>,
    /// Its one output port.
    pub output: Port,
    /// The JSON Schema of its parameters.
    pub parameters: Value,
    /// Whether the same request computes the same product.
    pub reproducibility: Reproducibility,
    /// The program and its arguments; a program written `./PATH` is the
    /// repository's.
    pub command: Vec<String>,
    /// The repository paths it is made of, pinned.
    pub code: Vec<String>,
    /// Whether it reaches the network (a machine grants it).
    #[serde(default)]
    pub network: bool,
    /// The variables of the controller's environment it is given, by name
    /// (a machine grants each).
    #[serde(default)]
    pub env: Vec<String>,
    /// The longest one run may take, in seconds.
    #[serde(default = "hour")]
    pub timeout_seconds: u64,
}

/// An operator of a repository, ready to run.
pub struct External {
    file: OperatorFile,
    repository: PathBuf,
    /// The repository paths it may read: its own file and its code (what its
    /// identity pins), and the directory of a `./` program outside its code
    /// (the machine's environment, as a virtualenv).
    pinned_paths: Vec<PathBuf>,
    /// What this machine grants it.
    grant: Grant,
    /// The digest of the operator file and its code.
    code: Digest,
    validator: jsonschema::Validator,
}

impl External {
    /// The operator file at `relative` in `repository`, checked, with its
    /// code pinned.
    ///
    /// # Errors
    /// It cannot be read, is not an operator file, or names code that is
    /// not there.
    pub fn load(repository: &Path, relative: &str, grants: &Grants) -> Result<Self, String> {
        let path = repository.join(relative);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{relative}: {e}"))?;
        let file: OperatorFile =
            serde_yaml::from_str(&text).map_err(|e| format!("{relative}: {e}"))?;
        if file.schema != OPERATOR_SCHEMA {
            return Err(format!("{relative}: not a {OPERATOR_SCHEMA} file"));
        }
        if file.command.is_empty() {
            return Err(format!("{relative}: names no command"));
        }
        let validator = jsonschema::validator_for(&file.parameters)
            .map_err(|e| format!("{relative}: parameters are not a JSON Schema: {e}"))?;
        let mut pinned = vec![(
            relative.to_owned(),
            splinter_core::digest::Digest::of(text.as_bytes()),
        )];
        // A `./` program outside its code is the machine's (a virtualenv
        // the repository keeps, like a program on PATH): readable whole,
        // never pinned.
        let environment = file.command[0].strip_prefix("./").and_then(|program| {
            let covered = file.code.iter().any(|code| {
                let code = code.trim_end_matches('/');
                program == code || program.starts_with(&format!("{code}/"))
            });
            let top = program
                .split('/')
                .next()
                .filter(|top| !top.is_empty() && *top != "..")?;
            (!covered).then(|| repository.join(top))
        });
        for code in &file.code {
            if code.starts_with('/') || code.split('/').any(|p| p == "..") {
                return Err(format!("{relative}: code {code:?} leaves the repository"));
            }
            let digest = code_digest(&repository.join(code))
                .map_err(|e| format!("{relative}: code {code}: {e}"))?;
            pinned.push((code.clone(), digest));
        }
        let grant = grants.of(&file.name);
        let mut ungranted: Vec<String> = file
            .env
            .iter()
            .filter(|v| !grant.env.contains(v))
            .map(|v| format!("variable {v}"))
            .collect();
        if file.network && !grant.network {
            ungranted.push("the network".into());
        }
        if !ungranted.is_empty() {
            return Err(format!(
                "{relative}: {} needs {}, which this machine does not grant it (see `splinter config`, the operators section)",
                file.name,
                ungranted.join(", ")
            ));
        }
        let pinned_paths = pinned
            .iter()
            .map(|(path, _)| repository.join(path))
            .chain(environment)
            .collect();
        let external = Self {
            code: digest_of(&(pinned, &grant)),
            file,
            repository: repository.to_path_buf(),
            pinned_paths,
            grant,
            validator,
        };
        let problems = external.descriptor().problems();
        if problems.is_empty() {
            Ok(external)
        } else {
            Err(format!("{relative}: {}", problems.join("; ")))
        }
    }

    /// The digest of the operator file and its code, as the lock pins it.
    #[must_use]
    pub fn code(&self) -> &Digest {
        &self.code
    }

    /// The operator's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.file.name
    }

    fn program(&self) -> String {
        match self.file.command[0].strip_prefix("./") {
            Some(own) => self.repository.join(own).to_string_lossy().into_owned(),
            None => self.file.command[0].clone(),
        }
    }
}

/// Every operator of the repository at `repository` (`operators/*.yaml`),
/// added to `registry`.
///
/// # Errors
/// Every operator file that cannot be loaded, each with why.
pub fn with_operators(
    registry: Registry,
    repository: &Path,
    grants: &Grants,
) -> Result<Registry, Vec<String>> {
    let dir = repository.join(OPERATORS_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(registry);
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".yaml"))
        .collect();
    names.sort();
    let (mut registry, mut problems) = (registry, Vec::new());
    for name in names {
        match External::load(repository, &format!("{OPERATORS_DIR}/{name}"), grants) {
            Ok(external) if registry.get(external.name()).is_some() => problems.push(format!(
                "{OPERATORS_DIR}/{name}: {} is an operation already",
                external.name()
            )),
            Ok(external) => registry = registry.with(external),
            Err(why) => problems.push(why),
        }
    }
    if problems.is_empty() {
        Ok(registry)
    } else {
        Err(problems)
    }
}

impl OperationSpec for External {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            schema: OPERATION_SCHEMA.into(),
            name: self.file.name.clone(),
            revision: self.file.revision,
            summary: self.file.summary.clone(),
            inputs: self.file.inputs.clone(),
            outputs: vec![self.file.output.clone()],
            parameters: self.file.parameters.clone(),
            reproducibility: self.file.reproducibility,
            isolation: Isolation::Any,
        }
    }

    fn check(&self, params: &Value) -> Result<(), String> {
        let problems: Vec<String> = self
            .validator
            .iter_errors(params)
            .map(|e| format!("{}: {e}", e.instance_path()))
            .collect();
        if problems.is_empty() {
            Ok(())
        } else {
            Err(problems.join("; "))
        }
    }

    fn pinned(&self) -> Option<(String, Digest)> {
        Some((format!("operator:{}", self.file.name), self.code.clone()))
    }

    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

impl Execute for External {
    fn implementation(&self) -> String {
        format!(
            "external {}@{} {}",
            self.file.name, self.file.revision, self.code
        )
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let work = tempfile::tempdir().map_err(|e| e.to_string())?;
        let request_dir = work.path().join("request");
        std::fs::create_dir(&request_dir).map_err(|e| e.to_string())?;
        let request = request_dir.join("request.json");
        let body = json!({
            "schema": REQUEST_SCHEMA,
            "operator": self.file.name,
            "params": call.params,
            "inputs": call.inputs,
            "out": call.out,
            "repository": self.repository,
        });
        std::fs::write(&request, body.to_string()).map_err(|e| e.to_string())?;
        let readable: Vec<PathBuf> = call
            .inputs
            .values()
            .flatten()
            .cloned()
            .chain([request_dir])
            .chain(self.pinned_paths.iter().cloned())
            .chain(self.grant.read.iter().cloned())
            .collect();
        let variables: BTreeMap<String, String> = self
            .file
            .env
            .iter()
            .filter_map(|name| Some((name.clone(), (call.variables?)(name)?)))
            .collect();

        let limits = Limits {
            wall_time_ms: self.file.timeout_seconds.saturating_mul(1000),
            cpu_seconds: self.file.timeout_seconds,
            ..Limits::default()
        };
        let sandbox = ProcessSandbox::new(work.path().join("calls"), BTreeMap::new(), &[], limits)
            .granted_env(variables)
            .read_write(self.grant.write.clone())
            .read_only(readable)
            .allow_network(self.file.network);
        let mut argv = vec![self.program()];
        argv.extend(self.file.command[1..].iter().cloned());
        argv.push("--request".into());
        argv.push(request.to_string_lossy().into_owned());
        let ran = sandbox
            .run_command(&call.out, &CommandCall { argv, stdin: None })
            .map_err(|e| format!("{}: {e}", self.file.name))?;
        match ran.exit_code {
            Some(0) => Ok(Outcome::Produced),
            Some(REJECTED) => Ok(Outcome::Rejected(
                ran.stdout.lines().last().unwrap_or("rejected").to_owned(),
            )),
            code => {
                let tail: String = ran
                    .stderr
                    .lines()
                    .rev()
                    .take(8)
                    .collect::<Vec<_>>()
                    .join(" | ");
                Err(format!("{} ended with {code:?}: {tail}", self.file.name))
            }
        }
    }
}
