// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible execution environments for
// agent learning, for its clients. If your team needs expertise in
// replayable agent runs, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: a runtime is named in a registry and pinned by what it resolves
//! to - the probed version string and the executable's content digest - and
//! an environment's snapshot is the digest of everything that determines
//! its behaviour, so changing any limit changes it.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use splinter_core::experience::{Digest, Environment};
use splinter_sandbox::{
    Limits, ProcessSandbox, ResolvedEnvironment, RuntimeEnvironment, RuntimeError, RuntimeRegistry,
    RuntimeSpec, SandboxError,
};

fn sandbox(limits: Limits) -> Arc<ProcessSandbox> {
    Arc::new(ProcessSandbox::new(
        std::env::temp_dir(),
        BTreeMap::new(),
        &[],
        limits,
    ))
}

fn python(limits: Limits) -> ResolvedEnvironment {
    ResolvedEnvironment::Runtime(
        RuntimeEnvironment::new(&RuntimeRegistry::builtin(), "python3", sandbox(limits)).unwrap(),
    )
}

#[test]
fn the_builtin_registry_names_the_common_runtimes() {
    let registry = RuntimeRegistry::builtin();
    for name in ["python3", "lua", "node", "sh"] {
        let spec = registry.get(name).unwrap();
        assert_eq!(spec.name, name);
        assert!(!spec.extension.is_empty());
    }
    assert!(matches!(
        registry.get("cobol"),
        Err(RuntimeError::Unknown { .. })
    ));
}

#[test]
fn a_missing_executable_is_a_typed_error() {
    let mut registry = RuntimeRegistry::builtin();
    registry.register(RuntimeSpec {
        name: "ghost".into(),
        program: "splinter-no-such-runtime".into(),
        argv: vec!["{exe}".into(), "{file}".into()],
        extension: "gh".into(),
        version_argv: Some(vec!["{exe}".into(), "--version".into()]),
        completion: "print('{line}')".into(),
        version_prefix: None,
    });
    let missing = RuntimeEnvironment::new(&registry, "ghost", sandbox(Limits::default()));
    assert!(
        matches!(
            missing,
            Err(SandboxError::Runtime(
                RuntimeError::ExecutableNotFound { .. }
            ))
        ),
        "{missing:?}"
    );
}

#[test]
fn version_and_executable_digest_land_in_the_snapshot() {
    let env = python(Limits::default());
    let ResolvedEnvironment::Runtime(runtime) = &env else {
        unreachable!()
    };
    let resolved = runtime.runtime();
    let version = resolved.version.clone().unwrap();
    assert!(version.starts_with("Python 3"), "{version}");
    let executable = PathBuf::from(&resolved.executable);
    let digest = Digest::of(&std::fs::read(&executable).unwrap());
    assert_eq!(resolved.executable_digest.as_ref(), Some(&digest));

    let record = env.record().unwrap();
    assert_eq!(record.kind, "runtime:python3");
    assert_eq!(record.spec["runtime"]["version"], version.as_str());
    assert_eq!(record.spec["runtime"]["executable_digest"], digest.as_str());
    assert_eq!(record.spec["sandbox"]["isolation"], "none");
    assert_eq!(
        record.snapshot,
        Some(Environment::snapshot_of(&record.kind, &record.spec))
    );
    assert_eq!(
        python(Limits::default()).record().unwrap(),
        record,
        "resolving again pins the same environment"
    );
}

#[test]
fn changing_any_enforced_limit_changes_the_snapshot() {
    let base = python(Limits::default()).record().unwrap().snapshot;
    let d = Limits::default();
    let variants = [
        Limits {
            wall_time_ms: d.wall_time_ms + 1,
            ..d
        },
        Limits {
            cpu_seconds: d.cpu_seconds + 1,
            ..d
        },
        Limits {
            memory_bytes: d.memory_bytes + 1,
            ..d
        },
        Limits {
            file_size_bytes: d.file_size_bytes + 1,
            ..d
        },
        Limits {
            output_cap_bytes: d.output_cap_bytes + 1,
            ..d
        },
    ];
    for limits in variants {
        assert_ne!(
            python(limits).record().unwrap().snapshot,
            base,
            "{limits:?}"
        );
    }
    // A process sandbox takes its process budget from `max_processes()`,
    // not from the limits it is built with, so the setting does not move the
    // snapshot (the budget, when given, does: see tests/containment.rs).
    let unenforced = Limits {
        max_processes: Some(d.max_processes.unwrap_or(0) + 1),
        ..d
    };
    assert_eq!(python(unenforced).record().unwrap().snapshot, base);
    assert_ne!(
        ResolvedEnvironment::ClosedBook.record().unwrap().snapshot,
        base
    );
    assert_eq!(
        ResolvedEnvironment::ClosedBook.record().unwrap(),
        Environment::closed_book()
    );
}

/// A script that is an interpreter as far as a version probe goes.
fn interpreter(dir: &std::path::Path, name: &str, version: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\necho '{version}'\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

#[test]
fn the_lua_runtime_is_lua_5_1_and_an_interpreter_that_reports_another_version_is_refused() {
    let dir = std::env::temp_dir().join(format!("splinter-lua-pin-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let resolve = |version: &str, name: &str| {
        let exe = interpreter(&dir, name, version);
        let mut registry = RuntimeRegistry::builtin();
        let mut spec = registry.get("lua").unwrap().clone();
        assert_eq!(spec.version_prefix.as_deref(), Some("Lua 5.1"));
        spec.program = exe.display().to_string();
        registry.register(spec);
        RuntimeEnvironment::new(&registry, "lua", sandbox(Limits::default()))
    };
    let pinned = resolve(
        "Lua 5.1.5  Copyright (C) 1994-2012 Lua.org, PUC-Rio",
        "lua51",
    )
    .unwrap();
    assert!(pinned
        .runtime()
        .version
        .as_deref()
        .unwrap()
        .starts_with("Lua 5.1"));
    let other = resolve(
        "Lua 5.4.6  Copyright (C) 1994-2023 Lua.org, PUC-Rio",
        "lua54",
    );
    assert!(
        matches!(
            other,
            Err(SandboxError::Runtime(RuntimeError::WrongVersion { .. }))
        ),
        "{other:?}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
