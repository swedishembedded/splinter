// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements containment for agents that run untrusted
// model-written code, for its clients. If your team needs expertise in
// keeping a language-model solver away from hidden tests, credentials, other
// candidates and the network, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: what a solver's command must not be able to do is prevented by the
//! kernel, not noticed afterwards. Each fixture is an attack a solver could
//! mount from its shell (directly, through `sh -c`, an interpreter or a
//! binary it brought along) and asserts that the attack is refused and that
//! nothing it aimed at changed or leaked.
//!
//! These run under the real Landlock and seccomp of the host. They are
//! ordinary tests, not skipped ones: on a kernel that cannot confine, the
//! sandbox refuses to run a command at all and every fixture fails, which is
//! the intended behaviour (`the_kernel_can_confine_and_deny_the_network`
//! says why in one place).

#![allow(clippy::unwrap_used)]

use std::net::TcpListener;
use std::path::{Path, PathBuf};

use splinter_sandbox::host::is_secret_name;
use splinter_sandbox::landlock::abi_version;
use splinter_sandbox::{
    CodeCall, CodeResult, CommandCall, Limits, ProcessSandbox, RuntimeRegistry, Sandbox,
};

/// Two candidate trees, a state root holding withheld tests, and a stand-in
/// home directory with credentials, all beside each other under one scratch.
struct Rig {
    scratch: tempfile::TempDir,
    tree_a: PathBuf,
    tree_b: PathBuf,
    state: PathBuf,
    home: PathBuf,
}

impl Rig {
    fn new() -> Self {
        let scratch = tempfile::tempdir().unwrap();
        let base = scratch.path().canonicalize().unwrap();
        let rig = Self {
            tree_a: base.join("candidate-a"),
            tree_b: base.join("candidate-b"),
            state: base.join("state"),
            home: base.join("home"),
            scratch,
        };
        for dir in [
            rig.tree_a.clone(),
            rig.tree_b.clone(),
            rig.state.join("hidden"),
            rig.home.join(".ssh"),
            rig.home.join(".cargo"),
        ] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(rig.tree_a.join("solution.py"), "VISIBLE = 1\n").unwrap();
        std::fs::write(rig.tree_b.join("solution.py"), "OTHER = 1\n").unwrap();
        std::fs::write(
            rig.state.join("hidden/test_hidden.py"),
            "assert HIDDEN_EXPECTATION == 42\n",
        )
        .unwrap();
        std::fs::write(rig.home.join(".ssh/id_ed25519"), "PRIVATE-KEY-MATERIAL\n").unwrap();
        std::fs::write(
            rig.home.join(".cargo/credentials.toml"),
            "token = \"CARGO-TOKEN-MATERIAL\"\n",
        )
        .unwrap();
        rig
    }

    fn sandbox(&self) -> ProcessSandbox {
        ProcessSandbox::new(
            self.scratch.path().join("calls"),
            [],
            &[],
            Limits::default(),
        )
    }

    /// Runs `script` under bash as the candidate in tree A.
    fn sh(&self, sandbox: &ProcessSandbox, script: &str) -> CodeResult {
        let call = CommandCall {
            argv: vec!["bash".into(), "-c".into(), script.into()],
            stdin: None,
        };
        sandbox.run_command(&self.tree_a, &call).unwrap()
    }
}

fn refused(result: &CodeResult) -> bool {
    result.exit_code != Some(0)
}

/// The sandbox's refusal as a program sees it: EACCES or EPERM, which the
/// shell, python and coreutils all print in words.
fn is_permission_error(result: &CodeResult) -> bool {
    let text = format!("{}{}", result.stderr, result.stdout);
    text.contains("Permission denied") || text.contains("Operation not permitted")
}

#[test]
fn the_kernel_can_confine_and_deny_the_network() {
    let abi = abi_version();
    assert!(
        abi.is_some_and(|v| v >= 4),
        "containment needs Landlock ABI 4 (filesystem and TCP rules); this kernel has {abi:?}: \
         every fixture here fails closed until it does"
    );
}

#[test]
fn withheld_tests_cannot_be_read_by_any_route() {
    let rig = Rig::new();
    let sandbox = rig.sandbox();
    let hidden = rig.state.join("hidden/test_hidden.py");
    let state = rig.state.display();
    for script in [
        format!("cat {}", hidden.display()),
        format!("sh -c 'cat {}'", hidden.display()),
        format!("python3 -c 'print(open(\"{}\").read())'", hidden.display()),
        format!("ls {state}/hidden"),
        // A walk from the filesystem root finds nothing of the state root.
        "find / -name 'test_hidden.py' 2>/dev/null".to_string(),
    ] {
        let result = rig.sh(&sandbox, &script);
        assert!(
            !result.stdout.contains("HIDDEN_EXPECTATION") && !result.stdout.contains("hidden"),
            "{script}: leaked: {result:?}"
        );
    }
    let direct = rig.sh(&sandbox, &format!("cat {}", hidden.display()));
    assert!(
        refused(&direct) && is_permission_error(&direct),
        "{direct:?}"
    );
}

#[test]
fn path_traversal_and_symlinks_cannot_leave_the_tree() {
    let rig = Rig::new();
    let sandbox = rig.sandbox();
    let outside = rig.state.join("hidden");
    let result = rig.sh(
        &sandbox,
        &format!(
            "ln -s {} link && cat link/test_hidden.py; \
             cat ../state/hidden/test_hidden.py; \
             cat ./../../{}/hidden/test_hidden.py; \
             python3 -c 'print(open(\"link/test_hidden.py\").read())'",
            outside.display(),
            rig.state.file_name().unwrap().to_string_lossy()
        ),
    );
    assert!(!result.stdout.contains("HIDDEN_EXPECTATION"), "{result:?}");
    assert!(
        result.stderr.matches("Permission denied").count() >= 3,
        "each attempt is refused by the kernel: {result:?}"
    );
}

#[test]
fn host_credentials_are_unreachable() {
    let rig = Rig::new();
    let sandbox = rig.sandbox();
    let mut targets = vec![
        rig.home.join(".ssh/id_ed25519"),
        rig.home.join(".cargo/credentials.toml"),
    ];
    // The real ones too, where this host has them.
    if let Some(home) = std::env::home_dir() {
        targets.extend([
            home.join(".ssh/id_ed25519"),
            home.join(".ssh/id_rsa"),
            home.join(".cargo/credentials"),
            home.join(".cargo/credentials.toml"),
            home.join(".config/gh/hosts.yml"),
        ]);
    }
    for target in targets.iter().filter(|t| t.exists()) {
        let result = rig.sh(&sandbox, &format!("cat {}", target.display()));
        assert!(
            refused(&result) && is_permission_error(&result),
            "{}: {result:?}",
            target.display()
        );
        assert!(!result.stdout.contains("MATERIAL"), "{result:?}");
    }
    let listed = rig.sh(&sandbox, &format!("ls -a {}", rig.home.display()));
    assert!(refused(&listed), "the home directory is not even listed");
}

#[test]
fn a_secret_in_the_callers_environment_never_reaches_a_command() {
    let rig = Rig::new();
    let caller = [
        ("BRAIN_API_KEY", "brain-key-value"),
        ("SSH_AUTH_SOCK", "/run/agent.sock"),
        ("GITHUB_TOKEN", "gh-token-value"),
        ("AWS_SECRET_ACCESS_KEY", "aws-value"),
        ("HF_HOME", "hf-dir"),
        ("CARGO_HOME", "cargo-dir"),
    ]
    .map(|(k, v)| (k.to_string(), v.to_string()));
    // Even an allowlist that names the secrets cannot pass them on.
    let sandbox = ProcessSandbox::new(
        rig.scratch.path().join("calls"),
        caller,
        &[
            "BRAIN_API_KEY",
            "SSH_AUTH_SOCK",
            "GITHUB_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "CARGO_HOME",
        ],
        Limits::default(),
    );
    let seen = rig.sh(&sandbox, "env");
    assert_eq!(seen.exit_code, Some(0), "{seen:?}");
    for leaked in [
        "brain-key-value",
        "gh-token-value",
        "aws-value",
        "agent.sock",
        "BRAIN_API_KEY",
        "SSH_AUTH_SOCK",
    ] {
        assert!(
            !seen.stdout.contains(leaked),
            "{leaked} reached the command"
        );
    }
    assert!(
        seen.stdout.contains("CARGO_HOME=cargo-dir"),
        "a plain allowlisted variable still passes"
    );
    assert!(!sandbox.spec().env.keys().any(|k| is_secret_name(k)));
}

#[test]
fn secret_names_are_recognised() {
    for secret in [
        "BRAIN_API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "SSH_AUTH_SOCK",
        "DB_PASSWORD",
        "AWS_PROFILE",
        "GITHUB_PAT",
    ] {
        assert!(is_secret_name(secret), "{secret}");
    }
    for plain in ["PATH", "LANG", "CARGO_HOME", "RUSTUP_HOME", "TERM"] {
        assert!(!is_secret_name(plain), "{plain}");
    }
}

#[test]
fn a_candidate_cannot_write_into_a_sibling_candidates_tree() {
    let rig = Rig::new();
    let sandbox = rig.sandbox();
    let sibling = &rig.tree_b;
    let result = rig.sh(
        &sandbox,
        &format!(
            "echo pwned > {0}/solution.py; echo pwned > {0}/planted.py; rm {0}/solution.py; \
             python3 -c 'open(\"{0}/p.py\", \"w\").write(\"x\")'",
            sibling.display()
        ),
    );
    assert!(
        result.stderr.matches("Permission denied").count() >= 3,
        "{result:?}"
    );
    assert_eq!(
        std::fs::read_to_string(sibling.join("solution.py")).unwrap(),
        "OTHER = 1\n"
    );
    assert!(!sibling.join("planted.py").exists() && !sibling.join("p.py").exists());
    assert_eq!(
        std::fs::read_to_string(rig.tree_a.join("solution.py")).unwrap(),
        "VISIBLE = 1\n",
        "its own tree is still its own"
    );
}

#[test]
fn a_subprocess_inherits_the_confinement() {
    let rig = Rig::new();
    let sandbox = rig.sandbox();
    let hidden = rig.state.join("hidden/test_hidden.py");
    // A compiled binary the candidate brought along runs from its tree, and
    // is confined like everything else it starts.
    compile_reader(&rig.tree_a.join("helper"));
    for script in [
        format!("sh -c 'sh -c \"cat {}\"'", hidden.display()),
        format!(
            "python3 -c 'import subprocess; print(subprocess.run([\"cat\", \"{}\"], capture_output=True, text=True))'",
            hidden.display()
        ),
        format!("./helper {}", hidden.display()),
        format!("nohup ./helper {} &", hidden.display()),
    ] {
        let result = rig.sh(&sandbox, &script);
        assert!(
            !result.stdout.contains("HIDDEN_EXPECTATION"),
            "{script}: {result:?}"
        );
    }
    let helper = rig.sh(&sandbox, &format!("./helper {}", hidden.display()));
    assert!(
        refused(&helper) && is_permission_error(&helper),
        "{helper:?}"
    );
}

/// Compiles a program that prints the file named by its first argument, the
/// way a candidate's own helper binary would (it makes its own system calls).
fn compile_reader(output: &Path) {
    let source = output.with_extension("c");
    std::fs::write(
        &source,
        "#include <stdio.h>\nint main(int c, char **v) { FILE *f = fopen(v[1], \"r\"); \
         if (!f) { perror(\"helper\"); return 1; } int ch; while ((ch = fgetc(f)) != EOF) \
         putchar(ch); return 0; }\n",
    )
    .unwrap();
    let status = std::process::Command::new("cc")
        .arg(&source)
        .arg("-o")
        .arg(output)
        .status();
    assert!(
        status.is_ok_and(|s| s.success()),
        "a C compiler (cc) is needed to build the helper fixture"
    );
}

/// A listener on the loopback, standing for a brain server a solver must not
/// reach.
fn loopback_listener() -> (TcpListener, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    (listener, port)
}

fn connect_script(port: u16) -> String {
    format!(
        "python3 -c 'import socket; socket.create_connection((\"127.0.0.1\", {port}), timeout=3)'"
    )
}

#[test]
fn the_network_is_denied_to_a_command_by_default() {
    let rig = Rig::new();
    let sandbox = rig.sandbox();
    let (listener, port) = loopback_listener();

    for script in [
        connect_script(port),
        format!("exec 3<>/dev/tcp/127.0.0.1/{port}"),
        // UDP is not covered by Landlock's TCP rules; the socket filter is.
        "python3 -c 'import socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM)'".to_string(),
        "python3 -c 'import socket; socket.socket(socket.AF_INET6, socket.SOCK_STREAM)'"
            .to_string(),
        format!("sh -c \"{}\"", connect_script(port).replace('"', "\\\"")),
    ] {
        let result = rig.sh(&sandbox, &script);
        assert!(
            refused(&result) && is_permission_error(&result),
            "{script}: {result:?}"
        );
    }
    assert!(
        listener.accept().is_err(),
        "nothing ever reached the listener"
    );
    assert!(!sandbox.spec().network_allowed);
}

#[test]
fn an_explicitly_allowed_network_is_open_and_recorded() {
    let rig = Rig::new();
    let sandbox = rig.sandbox().allow_network(true);
    let (listener, port) = loopback_listener();

    let result = rig.sh(&sandbox, &connect_script(port));
    assert_eq!(result.exit_code, Some(0), "{result:?}");
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert!(listener.accept().is_ok(), "the connection arrived");
    assert!(sandbox.spec().network_allowed, "the record says so");
}

#[test]
fn a_fork_bomb_is_stopped_at_the_process_budget_and_the_budget_is_recorded() {
    let rig = Rig::new();
    let sandbox = rig.sandbox().max_processes(Some(48));
    assert_eq!(sandbox.spec().limits.max_processes, Some(48));

    let result = rig.sh(
        &sandbox,
        "python3 - <<'PY'\n\
         import os, time\n\
         started = 0\n\
         for _ in range(600):\n\
        \x20   try:\n\
        \x20       pid = os.fork()\n\
        \x20   except OSError:\n\
        \x20       print('refused after', started)\n\
        \x20       break\n\
        \x20   if pid == 0:\n\
        \x20       time.sleep(30)\n\
        \x20       os._exit(0)\n\
        \x20   started += 1\n\
         else:\n\
        \x20   print('unbounded', started)\n\
         PY",
    );
    assert!(
        result.stdout.contains("refused after"),
        "the kernel stopped the forking: {result:?}"
    );
}

#[test]
fn cpu_memory_and_wall_limits_are_in_the_record() {
    let rig = Rig::new();
    let spec = rig.sandbox().spec();
    let defaults = Limits::default();
    assert_eq!(spec.limits.cpu_seconds, defaults.cpu_seconds);
    assert_eq!(spec.limits.memory_bytes, defaults.memory_bytes);
    assert_eq!(spec.limits.wall_time_ms, defaults.wall_time_ms);
    assert_eq!(
        spec.limits.max_processes, None,
        "unbounded unless the sandbox was given a budget, and the record says so"
    );
    assert_eq!(
        rig.sandbox()
            .max_processes(Some(7))
            .spec()
            .limits
            .max_processes,
        Some(7)
    );
}

/// Runs python `code` as a code call (the `run_code` tool's route).
fn run_code(sandbox: &ProcessSandbox, code: &str) -> CodeResult {
    let runtime = sandbox
        .resolve(RuntimeRegistry::builtin().get("python3").unwrap())
        .unwrap();
    sandbox
        .run(
            &runtime,
            &CodeCall {
                code: code.into(),
                stdin: None,
            },
        )
        .unwrap()
}

#[test]
fn a_code_call_cannot_read_write_or_connect_outside_its_directory() {
    let rig = Rig::new();
    let sandbox = rig.sandbox();
    let (listener, port) = loopback_listener();
    let hidden = rig.state.join("hidden/test_hidden.py");
    let credential = rig.home.join(".ssh/id_ed25519");
    let planted = rig.tree_b.join("planted.py");
    for (what, code) in [
        ("hidden test", format!("print(open({hidden:?}).read())")),
        ("credential", format!("print(open({credential:?}).read())")),
        ("sibling tree", format!("open({planted:?}, 'w').write('x')")),
        (
            "network",
            format!("import socket; socket.create_connection(('127.0.0.1', {port}), timeout=3)"),
        ),
    ] {
        let result = run_code(&sandbox, &code);
        assert!(
            refused(&result) && is_permission_error(&result),
            "{what}: {result:?}"
        );
        assert!(!result.stdout.contains("MATERIAL") && !result.stdout.contains("HIDDEN"));
    }
    assert!(!planted.exists());
    assert!(listener.accept().is_err(), "nothing reached the listener");

    // Its own directory and the system still work.
    let fine = run_code(
        &sandbox,
        "import os, json\nopen('note.txt', 'w').write('ok')\nprint(json.dumps(os.listdir('.')))",
    );
    assert_eq!(fine.exit_code, Some(0), "{fine:?}");
    assert!(fine.stdout.contains("note.txt"));
}

#[test]
fn a_code_call_is_refused_where_the_kernel_cannot_confine_it() {
    let rig = Rig::new();
    let sandbox = rig.sandbox().without_landlock();
    let runtime = sandbox
        .resolve(RuntimeRegistry::builtin().get("python3").unwrap())
        .unwrap();
    let refused = sandbox.run(
        &runtime,
        &CodeCall {
            code: "print(1)".into(),
            stdin: None,
        },
    );
    assert!(matches!(
        refused,
        Err(splinter_sandbox::SandboxError::Unconfined)
    ));
}

/// Starts a process of this user that holds `secret` in its environment, as
/// the supervisor or a model server would, and returns it with its id.
struct Bystander(std::process::Child);

impl Bystander {
    fn holding(secret: &str) -> Self {
        let child = std::process::Command::new("sleep")
            .arg("120")
            .env("BYSTANDER_SECRET", secret)
            .spawn()
            .unwrap();
        Self(child)
    }

    fn pid(&self) -> u32 {
        self.0.id()
    }
}

impl Drop for Bystander {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn another_process_of_the_user_is_invisible_through_proc() {
    let rig = Rig::new();
    let bystander = Bystander::holding("bystander-secret-value");
    let pid = bystander.pid();
    let environ = format!("cat /proc/{pid}/environ");
    let status = format!("cat /proc/{pid}/status");

    // With nothing in the way the secret is there to read, for any process of
    // the user: the fixture attacks something real. (An unconfined command
    // is in a pid namespace of its own too, which only hides it.)
    // The kernel publishes a new program's environment a moment after spawn
    // returns, so an early read is empty.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let open = loop {
        let read = std::fs::read(format!("/proc/{pid}/environ")).unwrap();
        if !read.is_empty() || std::time::Instant::now() > deadline {
            break read;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    assert!(String::from_utf8_lossy(&open).contains("bystander-secret-value"));

    // Landlock alone already refuses the files that need ptrace access
    // (the environment, the memory maps), but leaves the others readable:
    // that another process exists, whose it is and what it is called.
    let landlock_only = rig.sandbox().host_proc(true);
    let refused_by_landlock = rig.sh(&landlock_only, &environ);
    assert!(
        refused(&refused_by_landlock) && is_permission_error(&refused_by_landlock),
        "{refused_by_landlock:?}"
    );
    assert!(rig.sh(&landlock_only, &status).stdout.contains("Name:"));

    // The private view shows none of it.
    let sandbox = rig.sandbox();
    for script in [
        environ,
        status,
        format!("cat /proc/{pid}/cmdline"),
        format!("python3 -c 'print(open(\"/proc/{pid}/environ\").read())'"),
        format!("ls /proc/{pid}"),
    ] {
        let result = rig.sh(&sandbox, &script);
        assert!(refused(&result), "{script}: {result:?}");
        assert!(
            result.stderr.contains("No such file or directory"),
            "{script}: the process does not exist for the command: {result:?}"
        );
    }
    // It sees only its own processes (the shell, ls and what ls lists), not
    // the host's hundreds.
    let listed = rig.sh(&sandbox, "ls /proc | grep -c '^[0-9][0-9]*$'");
    let visible: u32 = listed.stdout.trim().parse().unwrap();
    assert!((1..=10).contains(&visible), "{listed:?}");
    assert!(!sandbox.spec().host_proc_visible);
    assert!(landlock_only.spec().host_proc_visible);
}

#[test]
fn the_private_view_still_reports_how_a_command_ended() {
    let rig = Rig::new();
    let sandbox = rig.sandbox();
    assert_eq!(rig.sh(&sandbox, "exit 7").exit_code, Some(7));
    assert_eq!(rig.sh(&sandbox, "exit 0").exit_code, Some(0));
    let killed = rig.sh(&sandbox, "kill -9 $$");
    assert_eq!(killed.signal, Some(9), "{killed:?}");
    let own = rig.sh(&sandbox, "cat /proc/self/status | head -1; echo $$");
    assert_eq!(own.exit_code, Some(0), "{own:?}");
}
