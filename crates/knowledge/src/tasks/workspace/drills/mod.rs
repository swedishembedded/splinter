// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements comprehension drills made by code from a
// codebase's own functions and tests, for its clients. If your team needs
// expertise in teaching a model what your code does, not only how to repair
// it, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Comprehension drills: small, machine-verified prerequisite tasks made by
//! code from a repository's clean snapshot, so a policy that misreads part of
//! the code can learn what that code does.
//!
//! A drill shows a short program that imports the repository's own module and
//! calls one of its functions on concrete inputs, and asks what it prints. The
//! reference is what the pinned interpreter prints when the program runs
//! against the clean snapshot - never a broken tree of any task, never a
//! hidden test. No model is asked for anything:
//!
//! * the functions are the module-level functions of the source files (Python
//!   by the parser, Lua by the house layout of one `function M.name(...)` per
//!   module function);
//! * the inputs are the literals the repository's own tests call each
//!   function with, and recombinations of them per parameter, chosen by a
//!   seed. Only the arguments are taken from a test: never an assertion, an
//!   expected value, a test's name or its text, so a drill leaks no hidden
//!   check. A function the tests never call with literals gets no drill;
//! * a candidate is admitted only when its program runs to completion, prints
//!   something bounded and not just `None`, prints the same on a second run
//!   (under another hash seed, for Python), does not read what the language
//!   leaves undefined, and the output is not one an earlier drill of the same
//!   function already has.
//!
//! A drill is a closed-book task of kind `predict-python` or `predict-lua`
//! graded by the formal verifier. Its first concept is the file it exercises
//! (its family, as a repair task's is) and its second `file::function`, so the
//! exam reservation by file applies to it unchanged; its privileged
//! [`DRILL_ORIGIN`] record says where it came from.

mod export;
mod inputs;
mod lua;
mod python;
mod run;
mod scope;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, Privileged, PrivilegedKind, Task};
use splinter_core::kinds::DRILL_ORIGIN;
use splinter_core::rng::Seed;
use splinter_sandbox::TreeSnapshot;

pub use export::DRILL_HARBOR_SCHEMA;
pub use scope::{touched_lines, Touched};

use super::profile::list_files;
use super::{FreshTree, WorkspaceError, WorkspaceGenerator};

/// The generator's name in provenance.
pub const DRILL_GENERATOR: &str = "splinter-knowledge/drills@1";

/// What [`DrillOrigin::schema`] says.
pub const DRILL_SCHEMA: &str = "splinter.drill/1";

/// The most drills of one function, by default.
pub const DEFAULT_PER_FUNCTION: usize = 3;

/// The longest, in seconds, a drill's program may run.
const RUN_LIMIT_SECS: u64 = 10;

/// The question every drill asks. The program follows it, fenced.
const QUESTION: &str =
    "What does this program print? Reply with exactly what it prints, in one fenced code block.";

/// The language of a drill's program, which fixes its kind and interpreter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    /// Python 3, run by `python3`.
    Python,
    /// Lua 5.1, run by `lua`.
    Lua,
}

impl Language {
    /// The language of the source file `path`, by its extension.
    #[must_use]
    pub fn of_path(path: &str) -> Option<Self> {
        match path.rsplit_once('.')?.1 {
            "py" => Some(Self::Python),
            "lua" => Some(Self::Lua),
            _ => None,
        }
    }

    /// The task kind a drill of this language has.
    #[must_use]
    pub fn kind(self) -> &'static str {
        match self {
            Self::Python => "predict-python",
            Self::Lua => "predict-lua",
        }
    }

    /// The fence the program is shown in.
    fn fence(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Lua => "lua",
        }
    }
}

/// What a generation of drills is asked to make.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrillPlan {
    /// The most drills of any one function.
    pub per_function: usize,
    /// The seed that chooses among the inputs; the same seed on the same tree
    /// makes the same drills.
    pub seed: Seed,
    /// The source files to drill, relative to the tree; every source file of
    /// the tree when empty.
    pub files: Vec<String>,
    /// When set, only these functions of these files (file to function
    /// names): what a repair task touches, found by [`touched_lines`].
    pub targets: Option<BTreeMap<String, BTreeSet<String>>>,
}

impl DrillPlan {
    /// A plan for `per_function` drills of every function of every source
    /// file, chosen by `seed`.
    #[must_use]
    pub fn new(per_function: usize, seed: Seed) -> Self {
        Self {
            per_function,
            seed,
            files: Vec::new(),
            targets: None,
        }
    }

    fn validate(&self) -> Result<(), WorkspaceError> {
        if self.per_function == 0 {
            return Err(WorkspaceError::Parameter {
                name: "per_function",
                reason: "at least one drill per function".into(),
            });
        }
        Ok(())
    }
}

/// Why a candidate program, or a function, did not become a drill.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrillReason {
    /// No test of the repository calls the function with literal arguments,
    /// and it needs some: there is no input to draw.
    NoTestInputs,
    /// The program failed to start, exited with an error or was cut off by a
    /// limit.
    DoesNotRun,
    /// The program prints nothing.
    EmptyOutput,
    /// The program prints more than a solver is asked to write out.
    OutputTooLong,
    /// The program prints something else when run again.
    Nondeterministic,
    /// The program reads what the language does not define (the clock,
    /// randomness, the environment).
    UndefinedByTheLanguage,
    /// The program prints only `None`: nothing to predict.
    Uninformative,
    /// An earlier drill of the function already has this output.
    SameOutput,
}

/// How a drill came to be, kept beside it as its privileged
/// [`DRILL_ORIGIN`] record. It holds no text of the program (the instruction
/// shows that), only what identifies it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrillOrigin {
    /// Always [`DRILL_SCHEMA`].
    pub schema: String,
    /// The language of the program.
    pub language: Language,
    /// The file the function is in: the drill's family.
    pub file: String,
    /// The function the program calls.
    pub function: String,
    /// The id of the git tree of the snapshot the program was run against.
    pub tree: String,
    /// What the interpreter that ran it says it is.
    pub interpreter: String,
    /// The digest of the program text.
    pub program: Digest,
}

impl DrillOrigin {
    /// The origin a task records, when it is a drill.
    #[must_use]
    pub fn of_task(task: &Task) -> Option<Self> {
        task.privileged.iter().find_map(|item| match &item.kind {
            PrivilegedKind::Other(name) if name == DRILL_ORIGIN => {
                serde_json::from_str(&item.content).ok()
            }
            _ => None,
        })
    }

    /// The concept naming the function, beside the file: `file::function`.
    #[must_use]
    pub fn function_concept(&self) -> String {
        format!("{}::{}", self.file, self.function)
    }
}

/// A drill admitted, with what identifies it.
#[derive(Clone, Debug)]
pub struct Drill {
    /// The task. Closed-book; its first concept is the file, its second
    /// `file::function`.
    pub task: Task,
    /// The family: the file the function is in.
    pub family: String,
    /// The function.
    pub function: String,
    /// What identifies the drill across runs: a digest over the tree, the
    /// file, the function and the program.
    pub key: Digest,
}

/// What a generation admitted and what it refused, by reason.
#[derive(Clone, Debug, Default)]
pub struct DrillGeneration {
    /// The drills admitted, in file, function and drawing order.
    pub admitted: Vec<Drill>,
    /// The candidates (and functions, for [`DrillReason::NoTestInputs`]) not
    /// admitted, counted by reason.
    pub refused: BTreeMap<DrillReason, usize>,
    /// Functions considered.
    pub functions: usize,
    /// Source or test files the parser could not read.
    pub unreadable: Vec<String>,
}

impl DrillGeneration {
    fn refuse(&mut self, reason: DrillReason) {
        *self.refused.entry(reason).or_insert(0) += 1;
    }
}

/// A function of a source file with what the tests call it with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct FunctionSource {
    pub file: String,
    pub name: String,
    pub language: Language,
    /// First and last line, 1-based, decorators included.
    pub line_start: usize,
    pub line_end: usize,
    pub params: Vec<Param>,
    /// How the file is imported: the directory to put on the path and the
    /// module name (Lua: the directory of `require`'s search and the file's
    /// stem).
    pub import: Option<Import>,
    pub calls: Vec<Call>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Param {
    pub name: String,
    pub variadic: bool,
    pub has_default: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Import {
    pub root: String,
    pub module: String,
}

/// One call: its positional arguments and keyword arguments as source text.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Call {
    pub args: Vec<String>,
    pub kwargs: Vec<(String, String)>,
}

/// A file that is a test, by where it lives and how it is named: its calls
/// are where inputs come from, and it is never itself drilled.
fn is_test(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    path.split('/')
        .any(|part| matches!(part, "tests" | "test" | "spec"))
        || name.starts_with("test_")
        || name.ends_with("_test.py")
        || matches!(name, "conftest.py" | "prelude.lua" | "setup.py")
}

/// Directories no source of the project lives in.
const SKIPPED: [&str; 9] = [
    ".git",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    "docs",
    "examples",
    "venv",
    ".venv",
];

fn is_source(path: &str) -> bool {
    Language::of_path(path).is_some()
        && !is_test(path)
        && !path.split('/').any(|part| SKIPPED.contains(&part))
}

impl WorkspaceGenerator {
    /// Makes the drills `plan` asks for from the clean `snapshot`: see the
    /// module documentation for what a drill is and what it is admitted on.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError`] when the plan is out of range, a file it names is
    /// not a source file of the tree, or the sandbox or the stores fail. A
    /// candidate that does not become a drill is in
    /// [`DrillGeneration::refused`], not an error.
    pub fn drills(
        &self,
        snapshot: &TreeSnapshot,
        plan: &DrillPlan,
    ) -> Result<DrillGeneration, WorkspaceError> {
        plan.validate()?;
        let env = self.environment(snapshot);
        let archive = self.archive_of(snapshot)?;
        let scratch = self.scratch.join("drills");
        let found = {
            let tree = FreshTree::lay(&env, &archive, &scratch)?;
            read_functions(&env, tree.path(), &plan.files, plan.targets.as_ref())?
        };
        let mut generation = DrillGeneration {
            unreadable: found.unreadable,
            ..DrillGeneration::default()
        };
        let mut runner = run::Runner::new(&env, archive, scratch);
        for function in found.functions {
            generation.functions += 1;
            let Some(candidates) = inputs::candidates(&function, plan.seed, plan.per_function)
            else {
                generation.refuse(DrillReason::NoTestInputs);
                continue;
            };
            let mut seen_outputs = BTreeSet::new();
            let mut admitted = 0;
            for call in candidates {
                if admitted == plan.per_function {
                    break;
                }
                match runner.admit(&function, &call)? {
                    Ok(done) if !seen_outputs.insert(done.reference.clone()) => {
                        generation.refuse(DrillReason::SameOutput);
                    }
                    Ok(done) => {
                        generation
                            .admitted
                            .push(make_drill(snapshot, &function, done)?);
                        admitted += 1;
                    }
                    Err(reason) => generation.refuse(reason),
                }
            }
        }
        Ok(generation)
    }

    /// The functions of the source files `touched` names that contain the
    /// changed lines it gives (file to the names of the functions), read
    /// from `snapshot`. A line outside every function (a constant, an
    /// import) names none, and a file with none is left out.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError`] when the snapshot cannot be laid out or a file
    /// named is not a source file of it.
    pub fn enclosing_functions(
        &self,
        snapshot: &TreeSnapshot,
        touched: &BTreeMap<String, Touched>,
    ) -> Result<BTreeMap<String, BTreeSet<String>>, WorkspaceError> {
        let env = self.environment(snapshot);
        let tree = FreshTree::lay(
            &env,
            &self.archive_of(snapshot)?,
            &self.scratch.join("drills"),
        )?;
        let files: Vec<String> = touched.keys().cloned().collect();
        let found = read_functions(&env, tree.path(), &files, None)?;
        let mut within: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for function in &found.functions {
            let Some(change) = touched.get(&function.file) else {
                continue;
            };
            change.check_against(tree.path(), &function.file)?;
            if change
                .lines
                .iter()
                .any(|line| (function.line_start..=function.line_end).contains(line))
            {
                within
                    .entry(function.file.clone())
                    .or_default()
                    .insert(function.name.clone());
            }
        }
        Ok(within)
    }
}

/// The functions found in some files, with the calls the tests make of them,
/// and the files that could not be read.
pub(super) struct Read {
    pub functions: Vec<FunctionSource>,
    pub unreadable: Vec<String>,
}

/// The functions of `files` (every source file of the tree when none is
/// named), or of those `targets` names, with the calls the tests make of them.
fn read_functions(
    env: &splinter_sandbox::WorkspaceEnvironment,
    tree: &std::path::Path,
    files: &[String],
    targets: Option<&BTreeMap<String, BTreeSet<String>>>,
) -> Result<Read, WorkspaceError> {
    let listing = list_files(tree)?;
    let named: BTreeSet<&String> = files.iter().collect();
    for file in &named {
        if !listing.contains(*file) || !is_source(file) {
            return Err(WorkspaceError::Parameter {
                name: "files",
                reason: format!("{file} is not a source file of the tree (a .py or .lua file that is not a test)"),
            });
        }
    }
    let sources: Vec<&String> = listing
        .iter()
        .filter(|f| is_source(f) && (named.is_empty() || named.contains(f)))
        .collect();
    let tests: Vec<&String> = listing
        .iter()
        .filter(|f| is_test(f) && Language::of_path(f).is_some())
        .collect();
    let of = |language: Language, paths: &[&String]| -> Vec<String> {
        paths
            .iter()
            .filter(|p| Language::of_path(p) == Some(language))
            .map(|p| (*p).clone())
            .collect()
    };
    let mut functions = Vec::new();
    let mut unreadable = Vec::new();
    let python_sources = of(Language::Python, &sources);
    if !python_sources.is_empty() {
        let read = python::read(env, tree, &python_sources, &of(Language::Python, &tests))?;
        functions.extend(read.functions);
        unreadable.extend(read.unreadable);
    }
    let lua_sources = of(Language::Lua, &sources);
    if !lua_sources.is_empty() {
        let read = lua::read(
            tree,
            &lua_sources,
            &tests.iter().map(|t| (*t).clone()).collect::<Vec<_>>(),
        )?;
        functions.extend(read.functions);
        unreadable.extend(read.unreadable);
    }
    functions.retain(|function| {
        targets.is_none_or(|targets| {
            targets
                .get(&function.file)
                .is_some_and(|names| names.contains(&function.name))
        })
    });
    functions.sort_by(|a, b| (&a.file, a.line_start).cmp(&(&b.file, b.line_start)));
    Ok(Read {
        functions,
        unreadable,
    })
}

/// The drill a program that was run and admitted makes.
fn make_drill(
    snapshot: &TreeSnapshot,
    function: &FunctionSource,
    done: run::Admitted,
) -> Result<Drill, WorkspaceError> {
    let origin = DrillOrigin {
        schema: DRILL_SCHEMA.to_string(),
        language: function.language,
        file: function.file.clone(),
        function: function.name.clone(),
        tree: snapshot.tree_id.clone(),
        interpreter: done.interpreter,
        program: Digest::of(done.program.as_bytes()),
    };
    let instruction = format!(
        "{QUESTION}\n\n```{}\n{}\n```",
        function.language.fence(),
        done.program.trim_end()
    );
    let privileged = vec![
        Privileged {
            kind: PrivilegedKind::Reference,
            content: done.reference,
            span: None,
        },
        Privileged {
            kind: PrivilegedKind::Other(DRILL_ORIGIN.to_string()),
            content: serde_json::to_string(&origin)?,
            span: None,
        },
    ];
    let task = Task::new(
        function.language.kind(),
        Vec::new(),
        Environment::closed_book(),
        instruction,
        privileged,
    )?
    .with_concepts([function.file.clone(), origin.function_concept()])?;
    let key = splinter_core::digest::digest_of(&(
        &snapshot.tree_id,
        &function.file,
        &function.name,
        &origin.program,
    ));
    Ok(Drill {
        task,
        family: function.file.clone(),
        function: function.name.clone(),
        key,
    })
}
