// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a model acquire several
// exactly verifiable skills one after another, each checked by an interpreter
// the model cannot influence, for its clients. If your team needs expertise
// in continual acquisition of verifiable task families you can procure our
// services by sending an email to info@swedishembedded.com.

//! A task family of output prediction: one language, the interpreter that is
//! its verifier, the manual the proposer reads, the wording of what the
//! proposer is asked, how a proposed program is admitted, and the generator
//! of the learner's own development programs.
//!
//! A family is a value. Lua 5.1 (what every lineage ran until now) is one
//! family; Python 3 and two counterfactual Lua dialects, whose rules no
//! pretrained model knows, are others. With only the Lua family a lineage's
//! requests, records and journals are what they were before families
//! existed: the Lua family's wording is the original wording, and every
//! family-specific field of a record is left out of its journal line when it
//! has the Lua family's value.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, ensure, Context as _, Result};
use splinter_sdk::sandbox::{
    CodeCall, Limits, ProcessSandbox, RuntimeEnvironment, RuntimeRegistry,
};
use splinter_sdk::vocabulary::digest::Digest;

/// The name of the original family, Lua 5.1; its records carry no family
/// field.
pub const LUA51: &str = "lua51";

/// The Python 3 documentation pages the proposer reads by default, relative
/// to the documentation directory.
pub const PYTHON_PAGES: [&str; 8] = [
    "library/functions.txt",
    "library/stdtypes.txt",
    "library/string.txt",
    "library/math.txt",
    "library/itertools.txt",
    "library/functools.txt",
    "tutorial/controlflow.txt",
    "tutorial/datastructures.txt",
];

/// The pages of a counterfactual dialect's generated manual.
pub const DIALECT_PAGES: [&str; 4] = ["strings.html", "tables.html", "math.html", "operators.html"];

/// What the proposer is told it is for, in Lua 5.1 (the original wording).
const LUA_GOAL: &str = "Write Lua 5.1 programs that exercise the standard library functions the \
                        manual describes, each printing a result a reader has to work out.";

const PYTHON_GOAL: &str =
    "Write Python 3 programs that exercise the built-in functions, types and \
                           library functions the manual describes, each printing a result a reader \
                           has to work out. Every program must print the same output on every run.";

/// The goal of a Lua dialect: its rules differ from standard Lua, so the
/// proposer is pointed at the manual's rules, not at what it knows of Lua.
fn dialect_goal(name: &str) -> String {
    format!(
        "Write programs in the Lua 5.1 dialect `{name}` that follow the rules of the manual \
         below, which differ from standard Lua 5.1 in places; each prints a result a reader has \
         to work out by applying the dialect's rules exactly."
    )
}

/// Angles a Lua request asks its programs to take, so asking again about a
/// section yields different programs.
pub const LUA_ANGLES: [&str; 12] = [
    "Use empty, single-character and long arguments.",
    "Use negative numbers, zero and fractional numbers as arguments.",
    "Print several lines, one result per line, from a loop.",
    "Combine the function with a second standard library function in one expression.",
    "Use return values beyond the first: multiple returns, captures, positions.",
    "Build a table first, then apply the library to it.",
    "Use patterns, escapes or format specifiers with unusual flags.",
    "Make a conditional depend on the function's result and print which branch ran.",
    "Use values near limits: large integers, tiny floats, long strings.",
    "Call it through pcall and print what comes back.",
    "Define a short function that uses it and call that function twice.",
    "Make the output hinge on a subtle rule of the function that a reader may get wrong.",
];

const PYTHON_ANGLES: [&str; 12] = [
    "Use empty, single-character and long arguments.",
    "Use negative numbers, zero and fractional numbers as arguments.",
    "Print several lines, one result per line, from a loop.",
    "Combine the function with a second function or method in one expression.",
    "Use slices, negative indices and step values.",
    "Build a list, dict or tuple first, then apply the function to it.",
    "Use format specifiers, f-strings or str methods with unusual arguments.",
    "Make a conditional depend on the function's result and print which branch ran.",
    "Use values near limits: large integers, tiny floats, long strings.",
    "Catch an exception and print its type and message.",
    "Define a short function that uses it and call that function twice.",
    "Make the output hinge on a subtle rule of the function that a reader may get wrong.",
];

/// The change a Lua dialect makes to the language.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    /// Standard Lua 5.1.
    Stock,
    /// String positions start at 0 and `#s` is the last index.
    ZeroIndex,
    /// `+` on two strings concatenates and `..` does not exist.
    PlusConcat,
}

/// Which language the programs are written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tongue {
    /// Lua 5.1, possibly a dialect of it.
    Lua(Dialect),
    /// Python 3.
    Python,
}

/// How a proposed program is admitted beyond the task generator's own checks
/// (it ran to completion, printed something bounded, printed the same on a
/// second run).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admission {
    /// Further runs that must print what the first printed; a language whose
    /// hash order varies from one process to the next (Python's string sets)
    /// is run more than twice.
    pub reruns: usize,
    /// Whether an output that shows a memory address is refused.
    pub refuse_addresses: bool,
}

/// The longest output a reply can be graded on.
pub const MAX_OUTPUT_BYTES: usize = 400;

/// The interpreter that verifies a family: where it is and which file it
/// must be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interpreter {
    /// The executable.
    pub path: PathBuf,
    /// Its SHA-256 as `sha256:<hex>`; a different file is refused. None
    /// accepts whatever is there (the original `--lua` behaviour).
    pub sha256: Option<String>,
}

/// One task family.
#[derive(Clone, Debug)]
pub struct Language {
    /// The family's name, as lineages and journals know it.
    pub name: String,
    /// The language.
    pub tongue: Tongue,
    /// The interpreter that is the verifier.
    pub interpreter: Interpreter,
    /// The manual pages the proposer reads.
    pub manuals: Vec<PathBuf>,
    /// How proposals are admitted.
    pub admission: Admission,
}

impl Language {
    /// The original family: Lua 5.1, whatever interpreter is given, no
    /// extra admission.
    #[must_use]
    pub fn lua51(interpreter: &Path, manuals: &[PathBuf]) -> Self {
        Self {
            name: LUA51.into(),
            tongue: Tongue::Lua(Dialect::Stock),
            interpreter: Interpreter {
                path: interpreter.to_path_buf(),
                sha256: None,
            },
            manuals: manuals.to_vec(),
            admission: Admission {
                reruns: 0,
                refuse_addresses: false,
            },
        }
    }

    /// Python 3: the interpreter must be the file whose digest is `sha256`.
    #[must_use]
    pub fn python3(interpreter: &Path, sha256: &str, manuals: &[PathBuf]) -> Self {
        Self {
            name: "python3".into(),
            tongue: Tongue::Python,
            interpreter: Interpreter {
                path: interpreter.to_path_buf(),
                sha256: Some(sha256.into()),
            },
            manuals: manuals.to_vec(),
            admission: Admission {
                reruns: 4,
                refuse_addresses: true,
            },
        }
    }

    /// A counterfactual Lua dialect, named as its installation is
    /// (`zero-index`, `plus-concat`).
    ///
    /// # Errors
    /// The name is not a known dialect.
    pub fn dialect(
        name: &str,
        interpreter: &Path,
        sha256: &str,
        manuals: &[PathBuf],
    ) -> Result<Self> {
        let dialect = match name {
            "zero-index" => Dialect::ZeroIndex,
            "plus-concat" => Dialect::PlusConcat,
            other => bail!("{other:?} is not a Lua dialect (zero-index, plus-concat)"),
        };
        Ok(Self {
            name: format!("lua51-{name}"),
            tongue: Tongue::Lua(dialect),
            interpreter: Interpreter {
                path: interpreter.to_path_buf(),
                sha256: Some(sha256.into()),
            },
            manuals: manuals.to_vec(),
            admission: Admission {
                reruns: 1,
                refuse_addresses: true,
            },
        })
    }

    /// Whether this is the original family.
    #[must_use]
    pub fn is_original(&self) -> bool {
        self.name == LUA51
    }

    /// The registry name of the runtime the family's programs run in.
    #[must_use]
    pub fn runtime(&self) -> &'static str {
        match self.tongue {
            Tongue::Lua(_) => "lua",
            Tongue::Python => "python3",
        }
    }

    /// The task kind the proposer is asked for.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self.tongue {
            Tongue::Lua(_) => "predict-lua",
            Tongue::Python => "predict-python",
        }
    }

    /// The language tag of a fenced block of the family's programs.
    #[must_use]
    pub fn fence(&self) -> &'static str {
        match self.tongue {
            Tongue::Lua(_) => "lua",
            Tongue::Python => "python",
        }
    }

    /// What the proposer is told it is for.
    #[must_use]
    pub fn goal(&self) -> String {
        match self.tongue {
            Tongue::Lua(Dialect::Stock) => LUA_GOAL.to_string(),
            Tongue::Lua(_) => dialect_goal(&self.name),
            Tongue::Python => PYTHON_GOAL.to_string(),
        }
    }

    /// The angles a request takes, by visit to a section.
    #[must_use]
    pub fn angles(&self) -> &'static [&'static str] {
        match self.tongue {
            Tongue::Lua(_) => &LUA_ANGLES,
            Tongue::Python => &PYTHON_ANGLES,
        }
    }

    /// The request that shows `program` and asks what it prints.
    #[must_use]
    pub fn request(&self, program: &str) -> String {
        crate::neutral::request_in(self.fence(), program)
    }

    /// The directory to put first on the search path of the runtime so that
    /// it finds the pinned interpreter: the interpreter's own directory when
    /// it is named as the runtime's program, else a directory under `state`
    /// holding a link of that name to it.
    ///
    /// # Errors
    /// The link cannot be made.
    pub fn runtime_dir(&self, state: &Path) -> Result<PathBuf> {
        let program = self.runtime();
        let path = &self.interpreter.path;
        if path.file_name().is_some_and(|n| n == program) {
            return Ok(path
                .parent()
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf));
        }
        let bin = std::path::absolute(state)?
            .join("runtime")
            .join(&self.name)
            .join("bin");
        std::fs::create_dir_all(&bin).with_context(|| format!("creating {}", bin.display()))?;
        let link = bin.join(program);
        match std::fs::remove_file(&link) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(e).with_context(|| format!("replacing {}", link.display()))
            }
            _ => {}
        }
        std::os::unix::fs::symlink(std::path::absolute(path)?, &link)
            .with_context(|| format!("linking {}", link.display()))?;
        Ok(bin)
    }

    /// Refuses an interpreter that is not the file the family is pinned to.
    ///
    /// # Errors
    /// The interpreter cannot be read or its digest is not the pinned one.
    pub fn verify_interpreter(&self) -> Result<()> {
        let Some(pinned) = &self.interpreter.sha256 else {
            return Ok(());
        };
        let path = self.interpreter.path.canonicalize().with_context(|| {
            format!(
                "resolving the interpreter {}",
                self.interpreter.path.display()
            )
        })?;
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let found = Digest::sha256_of(&bytes).to_string();
        ensure!(
            &found == pinned,
            "family {}: the interpreter {} has digest {found}, the family is pinned to {pinned}",
            self.name,
            path.display()
        );
        Ok(())
    }

    /// Whether a program's printed output is admitted for the family: not
    /// longer than a reply can be graded on, and, where the family says so,
    /// free of memory addresses.
    #[must_use]
    pub fn output_admitted(&self, output: &str) -> bool {
        output.len() <= MAX_OUTPUT_BYTES
            && !(self.admission.refuse_addresses && shows_address(output))
    }

    /// Runs `program` `admission.reruns` more times under `runtime` and
    /// accepts it only when each run prints `expected`.
    ///
    /// # Errors
    /// The runtime cannot run the program at all (not a refusal of it).
    pub fn rerun_agrees(
        &self,
        runtime: &RuntimeEnvironment,
        program: &str,
        expected: &str,
    ) -> Result<bool> {
        for _ in 0..self.admission.reruns {
            let run = runtime.run(&CodeCall {
                code: program.to_string(),
                stdin: None,
            })?;
            if run.timed_out
                || run.exit_code != Some(0)
                || run.stdout.trim_end() != expected.trim_end()
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Whether `text` shows a memory address: `0x` and at least six hex digits
/// (`table: 0x5581...`, `<function f at 0x7f...>`).
#[must_use]
pub fn shows_address(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.windows(2).enumerate().any(|(i, w)| {
        w == b"0x"
            && bytes[i + 2..]
                .iter()
                .take_while(|b| b.is_ascii_hexdigit())
                .count()
                >= 6
    })
}

/// Runs a family's programs under the sandbox every task is run under (no
/// network, bounded time, memory and output), with the pinned interpreter.
pub struct Runner {
    runtime: RuntimeEnvironment,
}

impl Runner {
    /// A runner of `language`'s programs; the sandbox keeps its scratch
    /// directories under `scratch`.
    ///
    /// # Errors
    /// The interpreter is not the pinned one or cannot be resolved.
    pub fn new(language: &Language, scratch: &Path) -> Result<Self> {
        language.verify_interpreter()?;
        std::fs::create_dir_all(scratch)
            .with_context(|| format!("creating {}", scratch.display()))?;
        let mut registry = RuntimeRegistry::builtin();
        let mut spec = registry.get(language.runtime())?.clone();
        // An absolute program is run as it is, not looked up on a path.
        spec.program = if language.interpreter.path.components().count() > 1 {
            std::path::absolute(&language.interpreter.path)?
                .display()
                .to_string()
        } else {
            language.interpreter.path.display().to_string()
        };
        registry.register(spec);
        let sandbox = ProcessSandbox::new(
            std::path::absolute(scratch)?,
            std::iter::empty::<(String, String)>(),
            &[],
            Limits::default(),
        );
        let runtime = RuntimeEnvironment::new(&registry, language.runtime(), Arc::new(sandbox))
            .map_err(|e| anyhow::anyhow!("{}: {e}", language.name))?;
        Ok(Self { runtime })
    }

    /// The runtime, for the admission reruns.
    #[must_use]
    pub fn runtime(&self) -> &RuntimeEnvironment {
        &self.runtime
    }

    /// What `source` prints (trailing whitespace trimmed).
    ///
    /// # Errors
    /// The program fails, times out or cannot be run.
    pub fn output(&self, source: &str) -> Result<String> {
        let run = self.runtime.run(&CodeCall {
            code: source.to_string(),
            stdin: None,
        })?;
        ensure!(
            !run.timed_out && run.exit_code == Some(0),
            "the interpreter refused a generated program ({source:?}): {}",
            run.stderr
        );
        Ok(run.stdout.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use splinter_sdk::vocabulary::rng::SplitMix64;

    use super::*;

    fn python() -> Language {
        // Whichever `python3` the host has: these tests are about the
        // family's checks, not about one build of it.
        let mut language = Language::python3(Path::new("python3"), "sha256:unused", &[]);
        language.interpreter.sha256 = None;
        language
    }

    #[test]
    fn an_address_in_the_output_is_refused_and_a_short_hex_number_is_not() {
        assert!(shows_address("table: 0x5581a3c4d2e0"));
        assert!(shows_address("<function f at 0x7f12abcd>"));
        assert!(!shows_address("0xff and 0x10"));
        let language = python();
        assert!(!language.output_admitted("<object at 0x7f12abcd>"));
        assert!(language.output_admitted("42"));
        assert!(!language.output_admitted(&"x".repeat(MAX_OUTPUT_BYTES + 1)));
        let lua = Language::lua51(Path::new("lua"), &[]);
        assert!(
            lua.output_admitted("table: 0x5581a3c4d2e0"),
            "the original family's admission is what it was: the task generator's own checks"
        );
    }

    #[test]
    fn a_digest_other_than_the_pinned_one_is_refused_and_the_pinned_one_accepted() {
        let me = std::env::current_exe().unwrap();
        let mut language = Language::python3(&me, "sha256:00", &[]);
        let error = language.verify_interpreter().unwrap_err();
        assert!(
            format!("{error:#}").contains("is pinned to sha256:00"),
            "{error:#}"
        );
        let digest = Digest::sha256_of(&std::fs::read(&me).unwrap()).to_string();
        language.interpreter.sha256 = Some(digest);
        language.verify_interpreter().unwrap();
    }

    #[test]
    fn an_interpreter_not_named_as_the_runtime_is_reached_through_a_link_of_that_name() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("python3.99");
        std::fs::write(&exe, "").unwrap();
        let language = Language::python3(&exe, "sha256:00", &[]);
        let bin = language.runtime_dir(&dir.path().join("state")).unwrap();
        assert_eq!(std::fs::read_link(bin.join("python3")).unwrap(), exe);
        // Asking again replaces the link.
        assert_eq!(
            language.runtime_dir(&dir.path().join("state")).unwrap(),
            bin
        );
    }

    #[test]
    fn python_programs_of_every_level_run_and_a_run_that_varies_is_caught() {
        let dir = tempfile::tempdir().unwrap();
        let language = python();
        let runner = Runner::new(&language, dir.path()).unwrap();
        let mut rng = SplitMix64::new(splinter_sdk::seed::resolve("seed", None));
        for level in 1..=6 {
            let program = language.tongue.generate(level, &mut rng);
            let printed = runner.output(&program).unwrap();
            assert!(!printed.is_empty(), "level {level}: {program}");
            assert!(language
                .rerun_agrees(runner.runtime(), &program, &printed)
                .unwrap());
        }
        // Hash order of strings changes from one process to the next.
        let varying = "print(set('abcdefghijklmnop'))";
        let once = runner.output(varying).unwrap();
        assert!(!language
            .rerun_agrees(runner.runtime(), varying, &once)
            .unwrap());
    }

    #[test]
    fn dialect_programs_are_generated_in_the_dialects_vocabulary() {
        let mut rng = SplitMix64::new(splinter_sdk::seed::resolve("seed", None));
        let plus: Vec<String> = (1..=6)
            .map(|l| Tongue::Lua(Dialect::PlusConcat).generate(l, &mut rng))
            .collect();
        assert!(plus.iter().all(|p| !p.contains("..")), "{plus:?}");
        let zero: Vec<String> = (1..=6)
            .map(|l| Tongue::Lua(Dialect::ZeroIndex).generate(l, &mut rng))
            .collect();
        assert!(zero.iter().any(|p| p.contains("string.sub")));
    }
}
