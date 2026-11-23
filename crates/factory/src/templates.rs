// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning factories a team can start from a
// working example, for its clients. If your team needs expertise in model
// fine-tuning pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Templates: task repositories that work as written, which `splinter init
//! --template NAME DIR` writes.
//!
//! Each template is the files of one: its recipes, the collection, tasks,
//! environment and plans they name, and nothing a run would make. Every
//! recipe of every template validates as written (the templates' spec says
//! so). A template is what a removed command did, as a recipe a team can
//! read and change: `knowledge-learning` is `learn`, `agent-sft` the plainest
//! learning loop, `generations` a campaign of promotions, `harness-search`
//! is `harness optimise` and `repository-practice` is `practice`; `init`
//! makes the latter's example project a git repository and names its commit
//! in the recipe. [`init`] writes a
//! template into a directory and never writes over a file already there:
//! it refuses, naming the files, before writing any.

use std::path::{Path, PathBuf};

/// One file of a template: its path in the repository and its text.
#[derive(Clone, Copy, Debug)]
pub struct TemplateFile {
    /// Its path, relative to the repository.
    pub path: &'static str,
    /// Its text.
    pub text: &'static str,
}

const fn file(path: &'static str, text: &'static str) -> TemplateFile {
    TemplateFile { path, text }
}

/// A task repository that works as written.
#[derive(Clone, Copy, Debug)]
pub struct Template {
    /// Its name.
    pub name: &'static str,
    /// What it does, in one line.
    pub summary: &'static str,
    /// Its own files.
    pub files: &'static [TemplateFile],
    /// What of the files the templates share it holds.
    pub shared: Shared,
    /// A directory of it that `init` makes a git repository of, committing
    /// its files; the commit replaces `REVISION` in the template's recipes.
    pub repository: Option<&'static str>,
}

/// The files templates share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shared {
    /// None of them.
    Nothing,
    /// The environment tasks run in and the plan candidates are evaluated by.
    Environment,
    /// Those, and the example task collection.
    EnvironmentAndTasks,
}

/// The token a template's recipe names its repository's revision by.
const REVISION: &str = "REVISION";

/// The example task collection, environment and evaluation plan the
/// task-based templates share.
const COMMON: &[TemplateFile] = &[
    file(
        "collections/tasks.yaml",
        include_str!("../templates/common/collections/tasks.yaml"),
    ),
    file(
        "environments/default.yaml",
        include_str!("../templates/common/environments/default.yaml"),
    ),
    file(
        "eval/held.yaml",
        include_str!("../templates/common/eval/held.yaml"),
    ),
    file(
        "tasks/eval/001-greet/environment/Dockerfile",
        include_str!("../templates/common/tasks/eval/001-greet/environment/Dockerfile"),
    ),
    file(
        "tasks/eval/001-greet/instruction.md",
        include_str!("../templates/common/tasks/eval/001-greet/instruction.md"),
    ),
    file(
        "tasks/eval/001-greet/task.toml",
        include_str!("../templates/common/tasks/eval/001-greet/task.toml"),
    ),
    file(
        "tasks/eval/001-greet/tests/test.sh",
        include_str!("../templates/common/tasks/eval/001-greet/tests/test.sh"),
    ),
    file(
        "tasks/train/001-hello/environment/Dockerfile",
        include_str!("../templates/common/tasks/train/001-hello/environment/Dockerfile"),
    ),
    file(
        "tasks/train/001-hello/instruction.md",
        include_str!("../templates/common/tasks/train/001-hello/instruction.md"),
    ),
    file(
        "tasks/train/001-hello/task.toml",
        include_str!("../templates/common/tasks/train/001-hello/task.toml"),
    ),
    file(
        "tasks/train/001-hello/tests/test.sh",
        include_str!("../templates/common/tasks/train/001-hello/tests/test.sh"),
    ),
];

/// Every template.
pub const TEMPLATES: &[Template] = &[
    Template {
        name: "agent-sft",
        summary: "a model learns a task collection from its own verified attempts",
        files: &[file(
            "recipes/agent-sft.yaml",
            include_str!("../templates/agent-sft/recipes/agent-sft.yaml"),
        )],
        shared: Shared::EnvironmentAndTasks,
        repository: None,
    },
    Template {
        name: "knowledge-learning",
        summary: "a model learns what documents say from closed-book tasks generated from them",
        files: &[
            file(
                "documents/guide.md",
                include_str!("../templates/knowledge-learning/documents/guide.md"),
            ),
            file(
                "recipes/knowledge-learning.yaml",
                include_str!("../templates/knowledge-learning/recipes/knowledge-learning.yaml"),
            ),
        ],
        shared: Shared::Nothing,
        repository: None,
    },
    Template {
        name: "harness-search",
        summary: "the harness that verifiably beats a baseline, among its mutations",
        files: &[
            file(
                "harness/baseline.yaml",
                include_str!("../templates/harness-search/harness/baseline.yaml"),
            ),
            file(
                "harness/space.yaml",
                include_str!("../templates/harness-search/harness/space.yaml"),
            ),
            file(
                "recipes/harness-search.yaml",
                include_str!("../templates/harness-search/recipes/harness-search.yaml"),
            ),
        ],
        shared: Shared::EnvironmentAndTasks,
        repository: None,
    },
    Template {
        name: "repository-practice",
        summary: "a model learns a repository from its own tests, by faults seeded into it",
        files: &[
            file(
                "project/pyproject.toml",
                include_str!("../templates/repository-practice/project/pyproject.toml"),
            ),
            file(
                "project/shop/__init__.py",
                include_str!("../templates/repository-practice/project/shop/__init__.py"),
            ),
            file(
                "project/shop/pricing.py",
                include_str!("../templates/repository-practice/project/shop/pricing.py"),
            ),
            file(
                "project/shop/status.py",
                include_str!("../templates/repository-practice/project/shop/status.py"),
            ),
            file(
                "project/tests/__init__.py",
                include_str!("../templates/repository-practice/project/tests/__init__.py"),
            ),
            file(
                "project/tests/test_pricing.py",
                include_str!("../templates/repository-practice/project/tests/test_pricing.py"),
            ),
            file(
                "project/tests/test_status.py",
                include_str!("../templates/repository-practice/project/tests/test_status.py"),
            ),
            file(
                "recipes/repository-practice.yaml",
                include_str!("../templates/repository-practice/recipes/repository-practice.yaml"),
            ),
        ],
        shared: Shared::Environment,
        repository: Some("project"),
    },
    Template {
        name: "generations",
        summary: "a generation of a model, promoted only when it improves, transfers and retains",
        files: &[
            file(
                "recipes/generation.yaml",
                include_str!("../templates/generations/recipes/generation.yaml"),
            ),
            file(
                "tasks/retention/001-list/environment/Dockerfile",
                include_str!(
                    "../templates/generations/tasks/retention/001-list/environment/Dockerfile"
                ),
            ),
            file(
                "tasks/retention/001-list/instruction.md",
                include_str!("../templates/generations/tasks/retention/001-list/instruction.md"),
            ),
            file(
                "tasks/retention/001-list/task.toml",
                include_str!("../templates/generations/tasks/retention/001-list/task.toml"),
            ),
            file(
                "tasks/retention/001-list/tests/test.sh",
                include_str!("../templates/generations/tasks/retention/001-list/tests/test.sh"),
            ),
            file(
                "tasks/transfer/001-note/environment/Dockerfile",
                include_str!(
                    "../templates/generations/tasks/transfer/001-note/environment/Dockerfile"
                ),
            ),
            file(
                "tasks/transfer/001-note/instruction.md",
                include_str!("../templates/generations/tasks/transfer/001-note/instruction.md"),
            ),
            file(
                "tasks/transfer/001-note/task.toml",
                include_str!("../templates/generations/tasks/transfer/001-note/task.toml"),
            ),
            file(
                "tasks/transfer/001-note/tests/test.sh",
                include_str!("../templates/generations/tasks/transfer/001-note/tests/test.sh"),
            ),
        ],
        shared: Shared::EnvironmentAndTasks,
        repository: None,
    },
];

/// The template named `name`.
#[must_use]
pub fn template(name: &str) -> Option<&'static Template> {
    TEMPLATES.iter().find(|t| t.name == name)
}

/// Why a template was not written.
#[derive(Debug, thiserror::Error)]
pub enum InitError {
    /// No template has the name.
    #[error("no template {name:?}; there are: {known}")]
    Unknown {
        /// The name asked for.
        name: String,
        /// The templates there are.
        known: String,
    },
    /// Files the template would write are already there.
    #[error("{dir} already holds {files}; nothing was written")]
    Exists {
        /// The directory.
        dir: String,
        /// The files, relative to it.
        files: String,
    },
    /// The template's repository could not be made.
    #[error("making {dir} a git repository: {why}")]
    Git {
        /// The directory.
        dir: String,
        /// Why.
        why: String,
    },
    /// A file could not be written.
    #[error("{path}: {source}")]
    Write {
        /// The file.
        path: String,
        /// Why.
        source: std::io::Error,
    },
}

impl Template {
    /// Every file the template writes, its shared ones first.
    pub fn all_files(&self) -> impl Iterator<Item = &'static TemplateFile> {
        let shared = self.shared;
        COMMON
            .iter()
            .filter(move |f| match shared {
                Shared::Nothing => false,
                Shared::Environment => {
                    f.path.starts_with("environments/") || f.path.starts_with("eval/")
                }
                Shared::EnvironmentAndTasks => true,
            })
            .chain(self.files.iter())
    }
}

/// Writes the template `name` into `dir` (created when missing) and returns
/// the files written. A verifier (`tests/*.sh`) is written executable.
///
/// # Errors
/// There is no such template, a file it writes is already there (then
/// nothing is written), or a file cannot be written.
pub fn init(name: &str, dir: &Path) -> Result<Vec<PathBuf>, InitError> {
    let template = template(name).ok_or_else(|| InitError::Unknown {
        name: name.to_owned(),
        known: TEMPLATES
            .iter()
            .map(|t| t.name)
            .collect::<Vec<_>>()
            .join(", "),
    })?;
    let there: Vec<&str> = template
        .all_files()
        .filter(|f| dir.join(f.path).exists())
        .map(|f| f.path)
        .collect();
    if !there.is_empty() {
        return Err(InitError::Exists {
            dir: dir.display().to_string(),
            files: there.join(", "),
        });
    }
    let mut written = Vec::new();
    for file in template.all_files() {
        let path = dir.join(file.path);
        let fail = |source| InitError::Write {
            path: path.display().to_string(),
            source,
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(fail)?;
        }
        std::fs::write(&path, file.text).map_err(fail)?;
        if file.path.ends_with(".sh") {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .map_err(fail)?;
        }
        written.push(path);
    }
    if let Some(project) = template.repository {
        let commit = commit(&dir.join(project))?;
        for path in written
            .iter()
            .filter(|p| p.starts_with(dir.join("recipes")))
        {
            let text = std::fs::read_to_string(path).map_err(|source| InitError::Write {
                path: path.display().to_string(),
                source,
            })?;
            std::fs::write(path, text.replace(REVISION, &commit)).map_err(|source| {
                InitError::Write {
                    path: path.display().to_string(),
                    source,
                }
            })?;
        }
    }
    Ok(written)
}

/// Makes `dir` a git repository of the files in it, committed, and returns
/// the commit.
fn commit(dir: &Path) -> Result<String, InitError> {
    let git = |args: &[&str]| -> Result<String, InitError> {
        let out = std::process::Command::new("git")
            .args([
                "-c",
                "user.name=splinter init",
                "-c",
                "user.email=init@splinter.invalid",
            ])
            .args(["-c", "commit.gpgsign=false"])
            .args(args)
            .current_dir(dir)
            .output()
            .map_err(|e| InitError::Git {
                dir: dir.display().to_string(),
                why: e.to_string(),
            })?;
        if !out.status.success() {
            return Err(InitError::Git {
                dir: dir.display().to_string(),
                why: String::from_utf8_lossy(&out.stderr).trim().to_owned(),
            });
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    git(&["init", "-q", "-b", "main"])?;
    git(&["add", "-A"])?;
    git(&["commit", "-q", "-m", "The example project"])?;
    git(&["rev-parse", "HEAD"])
}
