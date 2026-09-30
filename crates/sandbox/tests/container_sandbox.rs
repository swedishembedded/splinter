// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements container-isolated execution of
// model-written code for its clients. If your team needs expertise in agent
// sandboxing, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the container sandbox pins its image by digest and never by tag,
//! records `isolation: "container"` and the image digest in the snapshot,
//! and runs a call with no network and a read-only root.
//!
//! The tests that run a container need a docker daemon this user can reach;
//! without one they print why they were skipped and pass.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use splinter_sandbox::process::{run, ProcessSpec};
use splinter_sandbox::{
    CodeCall, ContainerSandbox, ImageRef, Isolation, Limits, ResolvedEnvironment,
    RuntimeEnvironment, RuntimeRegistry, Sandbox,
};

const DOCKER: &str = "docker";
/// The image the container tests run, pinned by digest once pulled.
const TEST_IMAGE: &str = "python:3-alpine";

/// The docker client's own environment: enough to find its socket and
/// configuration, nothing else.
fn client_env() -> BTreeMap<String, String> {
    [("PATH", "/usr/local/bin:/usr/bin:/bin"), ("HOME", "/tmp")]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn an_image_is_pinned_by_digest_never_by_tag() {
    let digest = format!("sha256:{}", "a".repeat(64));
    let image = ImageRef::parse(&format!("python@{digest}")).unwrap();
    assert_eq!(image.repository, "python");
    assert_eq!(image.digest.as_str(), digest);
    assert_eq!(image.to_string(), format!("python@{digest}"));
    for refused in [
        "python",
        "python:3.12",
        "python:3.12@sha256:abc",
        "@sha256:",
    ] {
        assert!(ImageRef::parse(refused).is_err(), "{refused}");
    }
}

/// `repo@sha256:<digest>` of `TEST_IMAGE`, pulled when it is not present.
fn pinned_test_image() -> Result<ImageRef, String> {
    let inspect = || {
        docker(&[
            "image",
            "inspect",
            "--format",
            "{{index .RepoDigests 0}}",
            TEST_IMAGE,
        ])
    };
    let reference = match inspect() {
        Ok(r) => r,
        Err(_) => {
            docker(&["pull", "--quiet", TEST_IMAGE])?;
            inspect()?
        }
    };
    ImageRef::parse(reference.trim()).map_err(|e| e.to_string())
}

fn docker(args: &[&str]) -> Result<String, String> {
    let mut argv = vec![DOCKER.to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let spec = ProcessSpec::new(
        argv,
        Path::new("/"),
        client_env(),
        std::time::Duration::from_secs(300),
        1024 * 1024,
    );
    let out = run(&spec).map_err(|e| e.to_string())?;
    if out.exit_code == Some(0) {
        Ok(String::from_utf8_lossy(&out.stdout.bytes).into_owned())
    } else {
        Err(String::from_utf8_lossy(&out.stderr.bytes).into_owned())
    }
}

/// The container sandbox for the test image, or why it cannot run here.
fn container(scratch: &Path) -> Option<ContainerSandbox> {
    if let Err(why) = ContainerSandbox::reachable(Path::new(DOCKER), &client_env()) {
        eprintln!(
            "SKIPPED: container sandbox tests need a docker daemon this user can reach: {why}"
        );
        return None;
    }
    match pinned_test_image() {
        Ok(image) => Some(ContainerSandbox::new(
            PathBuf::from(DOCKER),
            client_env(),
            image,
            scratch,
            Limits::default(),
        )),
        Err(why) => {
            eprintln!("SKIPPED: cannot pin the test image {TEST_IMAGE}: {why}");
            None
        }
    }
}

#[test]
fn a_call_runs_in_a_pinned_container_without_network() {
    let scratch = std::env::temp_dir().join(format!("splinter-container-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let Some(sandbox) = container(&scratch) else {
        return;
    };
    assert_eq!(sandbox.isolation(), Isolation::Container);
    let image = sandbox.image().clone();
    let env = ResolvedEnvironment::Runtime(
        RuntimeEnvironment::new(&RuntimeRegistry::builtin(), "python3", Arc::new(sandbox)).unwrap(),
    );
    let record = env.record().unwrap();
    assert_eq!(record.spec["sandbox"]["isolation"], "container");
    assert_eq!(record.spec["sandbox"]["image"], image.to_string());

    let ResolvedEnvironment::Runtime(runtime) = &env else {
        unreachable!()
    };
    let result = runtime
        .run(&CodeCall {
            code: "import socket, sys\nprint(sys.stdin.read())\n\
                   try:\n    socket.create_connection(('1.1.1.1', 53), timeout=2)\n    \
                   print('online')\nexcept OSError:\n    print('offline')\n\
                   try:\n    open('/etc/x', 'w')\n    print('writable')\nexcept OSError:\n    \
                   print('read-only')\n"
                .into(),
            stdin: Some("hello".into()),
        })
        .unwrap();
    assert_eq!(result.exit_code, Some(0), "{result:?}");
    assert_eq!(result.stdout, "hello\noffline\nread-only\n");
    assert_eq!(std::fs::read_dir(&scratch).unwrap().count(), 0);
    let _ = std::fs::remove_dir_all(&scratch);
}
