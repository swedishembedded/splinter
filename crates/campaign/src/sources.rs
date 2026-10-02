// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source intake that grounds every learned
// fact in exact, traceable bytes, for its clients. If your team needs
// expertise in knowledge acquisition or training-data provenance, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What Splinter learns from, as a command line names it: a file (a
//! document), a directory (a repository), or `cmd:` and a command whose run
//! is captured - never a URL - and, where a stage takes stored sources, a
//! stored source's id.

use std::path::PathBuf;

use serde::Serialize;
use splinter_knowledge::capture::{
    capture_command, capture_document, capture_repository, default_environment, CommandSpec,
    DEFAULT_MAX_FILE_BYTES,
};
use splinter_record::source::{CapturedSource, Origin, Source, SourceId};

use crate::context::Context;
use crate::error::CampaignError;
use crate::ids;

/// The prefix that makes a source a captured command.
pub const COMMAND_PREFIX: &str = "cmd:";

/// A source as a command line names it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceTarget {
    /// A file or directory.
    Path {
        /// The path, as given.
        path: PathBuf,
    },
    /// A command to run and capture.
    Command {
        /// The program and its arguments, run without a shell.
        argv: Vec<String>,
    },
    /// A source already in the store.
    Stored {
        /// Its id, or a unique prefix of it.
        id: String,
    },
}

impl SourceTarget {
    /// The target `source add` names: one path, or `cmd:` followed by a
    /// command (`cmd:make test`, or `cmd:make` `test` as separate
    /// arguments). A URL is refused.
    pub fn from_args(args: &[String]) -> Result<Self, CampaignError> {
        let Some(first) = args.first() else {
            return Err(CampaignError::Refused(
                "name a file, a directory, or cmd: followed by a command".into(),
            ));
        };
        if let Some(program) = first.strip_prefix(COMMAND_PREFIX) {
            let argv: Vec<String> = program
                .split_whitespace()
                .map(str::to_string)
                .chain(args[1..].iter().cloned())
                .collect();
            if argv.is_empty() {
                return Err(CampaignError::Refused(format!(
                    "{COMMAND_PREFIX} names no command to run"
                )));
            }
            return Ok(Self::Command { argv });
        }
        refuse_url(first)?;
        if args.len() > 1 {
            return Err(CampaignError::Refused(format!(
                "one path at a time: {:?} follows {first:?} (a command is {COMMAND_PREFIX}<command>)",
                args[1]
            )));
        }
        Ok(Self::Path {
            path: PathBuf::from(first),
        })
    }

    /// The target one argument of `learn` names: an existing path, `cmd:`
    /// and a whole command, or else a stored source's id.
    pub fn from_learn_arg(arg: &str) -> Result<Self, CampaignError> {
        if let Some(program) = arg.strip_prefix(COMMAND_PREFIX) {
            return Self::from_args(&[format!("{COMMAND_PREFIX}{program}")]);
        }
        refuse_url(arg)?;
        let path = PathBuf::from(arg);
        if path.exists() {
            return Ok(Self::Path { path });
        }
        let hex = arg.strip_prefix("sha256:").unwrap_or(arg);
        if hex.len() >= ids::MIN_PREFIX && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Ok(Self::Stored { id: arg.into() });
        }
        Err(CampaignError::Refused(format!(
            "{arg:?} is not a file, a directory, {COMMAND_PREFIX}<command> or a stored source id"
        )))
    }
}

/// Refuses `arg` when it is a URL: sources are local.
fn refuse_url(arg: &str) -> Result<(), CampaignError> {
    let lower = arg.to_ascii_lowercase();
    if lower.contains("://") || lower.starts_with("www.") {
        return Err(CampaignError::Refused(format!(
            "{arg:?} is a URL; Splinter learns from local files, directories and commands only \
             (download it and add the file)"
        )));
    }
    Ok(())
}

/// One source, as the source commands report it.
#[derive(Clone, Debug, Serialize)]
pub struct SourceSummary {
    /// Its id.
    pub id: SourceId,
    /// `document`, `repository` or `command`.
    pub kind: &'static str,
    /// Where it came from: the path, or the command's argv, working
    /// directory and exit code.
    pub origin: Origin,
    /// Its parts: files, or a command's output streams.
    pub parts: usize,
    /// Its parts' total size in bytes.
    pub bytes: u64,
    /// When it was first captured.
    pub captured_at: String,
}

impl From<&Source> for SourceSummary {
    fn from(source: &Source) -> Self {
        Self {
            id: source.id.clone(),
            kind: source.kind(),
            origin: source.origin.clone(),
            parts: source.parts.len(),
            bytes: source.parts.iter().map(|p| p.bytes).sum(),
            captured_at: source.captured_at.clone(),
        }
    }
}

/// What `source add` reports.
#[derive(Clone, Debug, Serialize)]
pub struct SourceAdded {
    /// The source.
    pub source: SourceSummary,
    /// Whether it was new to the store; unchanged content is the same
    /// source, captured once.
    pub new: bool,
}

/// Captures `target` and stores it. A stored id is looked up instead.
pub fn add(ctx: &Context, target: &SourceTarget) -> Result<SourceAdded, CampaignError> {
    let store = ctx.sources();
    let captured = match target {
        SourceTarget::Stored { id } => {
            let source = store.get_source(&resolve(ctx, id)?)?;
            return Ok(SourceAdded {
                source: SourceSummary::from(&source),
                new: false,
            });
        }
        SourceTarget::Path { path } => capture_path(ctx, path)?,
        SourceTarget::Command { argv } => {
            let env = default_environment(ctx.config().command_env.clone());
            let spec = CommandSpec::new(argv.clone(), &ctx.config().working_dir, env);
            capture_command(&spec, ctx.clock())?
        }
    };
    let new = !store.contains(&captured.source().id);
    let id = store.put_source(&captured)?;
    Ok(SourceAdded {
        source: SourceSummary::from(&store.get_source(&id)?),
        new,
    })
}

/// A file captured as a document, a directory as a repository.
fn capture_path(ctx: &Context, path: &std::path::Path) -> Result<CapturedSource, CampaignError> {
    let metadata = std::fs::metadata(path).map_err(crate::error::io(path))?;
    Ok(if metadata.is_dir() {
        capture_repository(path, DEFAULT_MAX_FILE_BYTES, ctx.clock())?
    } else {
        capture_document(path, DEFAULT_MAX_FILE_BYTES, ctx.clock())?
    })
}

/// What `source list` reports.
#[derive(Clone, Debug, Serialize)]
pub struct SourceList {
    /// Every stored source, in id order.
    pub sources: Vec<SourceSummary>,
}

/// Every stored source.
pub fn list(ctx: &Context) -> Result<SourceList, CampaignError> {
    let store = ctx.sources();
    let sources = store
        .list()?
        .iter()
        .map(|id| store.get_source(id).map(|s| SourceSummary::from(&s)))
        .collect::<Result<_, _>>()?;
    Ok(SourceList { sources })
}

/// The stored source `id` names, whole.
pub fn show(ctx: &Context, id: &str) -> Result<Source, CampaignError> {
    Ok(ctx.sources().get_source(&resolve(ctx, id)?)?)
}

/// The stored source `id` (or a unique prefix of it) names.
pub fn resolve(ctx: &Context, id: &str) -> Result<SourceId, CampaignError> {
    let stored = ctx.sources().list()?.into_iter().map(|s| s.0);
    Ok(SourceId(ids::resolve("source", id, stored)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_target_is_a_path_or_a_command_and_never_a_url() {
        assert_eq!(
            SourceTarget::from_args(&args(&["notes/manual.md"])).ok(),
            Some(SourceTarget::Path {
                path: "notes/manual.md".into()
            })
        );
        let command = SourceTarget::Command {
            argv: args(&["make", "--help"]),
        };
        assert_eq!(
            SourceTarget::from_args(&args(&["cmd:make", "--help"])).ok(),
            Some(command.clone())
        );
        assert_eq!(
            SourceTarget::from_args(&args(&["cmd:make --help"])).ok(),
            Some(command)
        );
        assert!(SourceTarget::from_args(&args(&["cmd:"])).is_err());
        assert!(SourceTarget::from_args(&args(&[])).is_err());
        assert!(SourceTarget::from_args(&args(&["a.md", "b.md"])).is_err());
        for url in ["https://example.com/doc.md", "HTTP://x", "www.example.com"] {
            let refused = SourceTarget::from_args(&args(&[url])).unwrap_err();
            assert!(refused.to_string().contains("URL"), "{refused}");
        }
    }

    #[test]
    fn a_learn_argument_that_is_no_path_is_a_stored_id_or_refused() {
        assert!(matches!(
            SourceTarget::from_learn_arg("sha256:abcd1234"),
            Ok(SourceTarget::Stored { .. })
        ));
        assert!(matches!(
            SourceTarget::from_learn_arg("cmd:ls -la"),
            Ok(SourceTarget::Command { .. })
        ));
        assert!(SourceTarget::from_learn_arg("no/such/file.md").is_err());
    }
}
