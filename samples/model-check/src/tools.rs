// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The external programs the checks run: the pinned Lua interpreter and TLC.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;

/// Where the interpreters and the sample's Lua programs are.
#[derive(Debug, Clone)]
pub struct Tools {
    /// The Lua 5.1 interpreter (`make deps/lua`).
    pub lua: PathBuf,
    /// The TLC wrapper script (`make deps/tlaplus`, its `bin/tlc`).
    pub tlc: PathBuf,
    /// The sample's `lua` directory (explorer, trace generator).
    pub lua_dir: PathBuf,
}

/// One finished process: stdout then stderr, and the exit code (`None` when a
/// signal ended it).
#[derive(Debug, Clone)]
pub struct Run {
    /// Standard output followed by standard error.
    pub output: String,
    /// The exit code.
    pub code: Option<i32>,
}

fn finish(command: &mut Command, what: &str) -> anyhow::Result<Run> {
    let out = command
        .output()
        .with_context(|| format!("starting {what}"))?;
    let mut output = String::from_utf8_lossy(&out.stdout).into_owned();
    output.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok(Run {
        output,
        code: out.status.code(),
    })
}

impl Tools {
    /// Runs `lua <script> <args...>` with `script` given as a path.
    pub fn lua_script(&self, script: &Path, args: &[&OsStr]) -> anyhow::Result<Run> {
        finish(
            Command::new(&self.lua).arg(script).args(args),
            &format!("{} {}", self.lua.display(), script.display()),
        )
    }

    /// Runs one of the sample's Lua programs (`lua_dir/<name>`).
    pub fn lua_program(&self, name: &str, args: &[&OsStr]) -> anyhow::Result<Run> {
        self.lua_script(&self.lua_dir.join(name), args)
    }

    /// Runs TLC in `cwd` (where the module and its dependencies are) on `tla`
    /// with `cfg`, keeping its working files under `cwd/tlc-meta`.
    ///
    /// TLC unpacks the standard modules (Naturals, Sequences, ...) into the JVM's
    /// temporary directory, and concurrent runs sharing it corrupt each other's
    /// copies. Every run therefore gets its own, under `cwd/tlc-tmp`.
    pub fn tlc(&self, cwd: &Path, cfg: &str, tla: &str) -> anyhow::Result<Run> {
        let tmp = cwd.join("tlc-tmp");
        std::fs::create_dir_all(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        finish(
            Command::new(&self.tlc)
                .current_dir(cwd)
                .env(
                    "JAVA_TOOL_OPTIONS",
                    format!("-Djava.io.tmpdir={}", tmp.display()),
                )
                .args(["-workers", "4", "-metadir", "tlc-meta", "-config", cfg, tla]),
            &format!("{}", self.tlc.display()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn concurrent_tlc_runs_do_not_share_a_temporary_directory() {
        let dir = tempfile::tempdir().unwrap();
        let fake = dir.path().join("tlc");
        std::fs::write(&fake, "#!/bin/sh\necho \"$JAVA_TOOL_OPTIONS\"\n").unwrap();
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        let tools = Tools {
            lua: PathBuf::new(),
            tlc: fake,
            lua_dir: PathBuf::new(),
        };
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let tmpdir_of = |work: &Path| {
            tools
                .tlc(work, "M.cfg", "M.tla")
                .unwrap()
                .output
                .trim()
                .to_string()
        };
        let (first, second) = (tmpdir_of(a.path()), tmpdir_of(b.path()));
        assert!(first.contains(&a.path().display().to_string()), "{first}");
        assert_ne!(first, second);
    }
}
