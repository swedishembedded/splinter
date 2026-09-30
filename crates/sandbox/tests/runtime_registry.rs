// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible execution environments for
// agent learning, for its clients. If your team needs expertise in
// replayable agent experiments, you can procure our services by sending an
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

use splinter_sandbox::{
    Limits, ProcessSandbox, ResolvedEnvironment, RuntimeEnvironment, RuntimeError, RuntimeRegistry,
    RuntimeSpec, SandboxError,
};
use splinter_store::experience::{Digest, Environment};

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
fn changing_any_limit_changes_the_snapshot() {
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
            max_processes: d.max_processes + 1,
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
    assert_ne!(
        ResolvedEnvironment::ClosedBook.record().unwrap().snapshot,
        base
    );
    assert_eq!(
        ResolvedEnvironment::ClosedBook.record().unwrap(),
        Environment::closed_book()
    );
}
