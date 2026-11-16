// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a model acquire several
// exactly verifiable skills one after another, each checked by an interpreter
// the model cannot influence, for its clients. If your team needs expertise
// in continual acquisition of verifiable task families you can procure our
// services by sending an email to info@swedishembedded.com.

//! The command line's description of a task family (`--family`): a name and
//! `key=value` settings, parsed into the family's [`Language`], when its
//! frontier is found and how large its development sets are.
//!
//! ```text
//! --family python3,at=2,target=3000,interpreter=/path/python3.14,manual-dir=/path/docs,dev-generator=120
//! --family zero-index,at=3,dir=/path/lua51-dialects/zero-index
//! ```
//!
//! Keys: `at` (generations whose rounds find the family's frontier, joined by
//! `+`; default the family's position, 1-based), `target` (frontier
//! programs), `dir` (a dialect's installation: `bin/lua` and `manual/`),
//! `interpreter`, `sha256` (`sha256:<hex>`; defaults to the build recorded
//! for the family), `manual-dir` (the manual's directory, with the family's
//! default pages), `manual` (one page, repeatable; replaces the default
//! pages), and the sizes `dev-generator`, `keep-generator`, `dev-proposals`,
//! `keep-proposals` (defaults: the lineage's own flags).

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context as _, Result};

use crate::language::{Language, DIALECT_PAGES, LUA51, PYTHON_PAGES};
use crate::plan::FamilyPlan;

/// The Python 3.14.4 interpreter pinned for the family (the distribution's
/// `python3.14` of the node the family was built on).
pub const PYTHON3_SHA256: &str =
    "sha256:d1b87c365d0cf5a985cfedfc227ff613a5bc6ff56e841202eb9330d263bbeee9";

/// The `zero-index` dialect interpreter as built by
/// `scripts/deps/lua51-dialects.sh` on the node the family was built on.
pub const ZERO_INDEX_SHA256: &str =
    "sha256:4c91a00829a6af82d101a017bcb9db3b4e995a3afe216a206a31161254e6e007";

/// The `plus-concat` dialect interpreter, as above.
pub const PLUS_CONCAT_SHA256: &str =
    "sha256:dd32b653e4c6e8f7ef0dd0b9aa6ce98957b5fb7a2e0cd535d7efa6ad0e71b8a7";

/// Sizes of a family's development sets; absent ones are the lineage's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sizes {
    /// Generator programs the dev gate is decided on.
    pub dev_generator: Option<usize>,
    /// Generator programs kept for retention.
    pub keep_generator: Option<usize>,
    /// Proposals of the base held out for the dev gate.
    pub dev_proposals: Option<usize>,
    /// Proposals of the base held out for retention.
    pub keep_proposals: Option<usize>,
}

/// One `--family` argument, parsed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FamilySpec {
    /// The family's name as given (`lua51`, `python3`, `zero-index`,
    /// `plus-concat`).
    pub name: String,
    /// Generations that find its frontier.
    pub at: Option<Vec<u32>>,
    /// Frontier programs per round.
    pub target: Option<usize>,
    dir: Option<PathBuf>,
    interpreter: Option<PathBuf>,
    sha256: Option<String>,
    manual_dir: Option<PathBuf>,
    manuals: Vec<PathBuf>,
    /// Development set sizes.
    pub sizes: Sizes,
}

impl FamilySpec {
    /// The original family, Lua 5.1, with its default sets and schedule.
    #[must_use]
    pub fn original() -> Self {
        Self {
            name: LUA51.to_string(),
            ..Self::default()
        }
    }

    /// Parses `text`.
    ///
    /// # Errors
    /// An unknown key, a value that is not what the key takes, or a key given
    /// twice that takes one value.
    pub fn parse(text: &str) -> Result<Self> {
        let mut parts = text.split(',');
        let name = parts.next().unwrap_or_default().trim().to_string();
        ensure!(
            !name.is_empty() && !name.contains('='),
            "--family {text:?} starts with the family's name"
        );
        let mut spec = Self {
            name,
            ..Self::default()
        };
        for part in parts {
            let (key, value) = part
                .split_once('=')
                .with_context(|| format!("--family {text:?}: {part:?} is not key=value"))?;
            let number = |what: &str| -> Result<usize> {
                value
                    .parse()
                    .with_context(|| format!("--family {text:?}: {what} {value:?} is not a number"))
            };
            let once = |slot: bool| -> Result<()> {
                ensure!(!slot, "--family {text:?}: {key} is given twice");
                Ok(())
            };
            match key {
                "at" => {
                    once(spec.at.is_some())?;
                    spec.at = Some(
                        value
                            .split('+')
                            .map(|g| {
                                g.parse()
                                    .with_context(|| format!("--family {text:?}: at {g:?}"))
                            })
                            .collect::<Result<_>>()?,
                    );
                }
                "target" => {
                    once(spec.target.is_some())?;
                    spec.target = Some(number("target")?);
                }
                "dir" => {
                    once(spec.dir.is_some())?;
                    spec.dir = Some(value.into());
                }
                "interpreter" => {
                    once(spec.interpreter.is_some())?;
                    spec.interpreter = Some(value.into());
                }
                "sha256" => {
                    once(spec.sha256.is_some())?;
                    ensure!(
                        value.starts_with("sha256:"),
                        "--family {text:?}: sha256 is `sha256:<hex>`"
                    );
                    spec.sha256 = Some(value.into());
                }
                "manual-dir" => {
                    once(spec.manual_dir.is_some())?;
                    spec.manual_dir = Some(value.into());
                }
                "manual" => spec.manuals.push(value.into()),
                "dev-generator" => spec.sizes.dev_generator = Some(number(key)?),
                "keep-generator" => spec.sizes.keep_generator = Some(number(key)?),
                "dev-proposals" => spec.sizes.dev_proposals = Some(number(key)?),
                "keep-proposals" => spec.sizes.keep_proposals = Some(number(key)?),
                other => bail!("--family {text:?}: unknown key {other:?}"),
            }
        }
        Ok(spec)
    }

    /// The family's language. `lua` and `manuals` are the lineage's own
    /// `--lua` and `--manual`, which the original family uses.
    ///
    /// # Errors
    /// The name is not a family, or its interpreter or manual is not given.
    pub fn language(&self, lua: &Path, manuals: &[PathBuf]) -> Result<Language> {
        let pages = |default: &[&str], dialect_dir: Option<PathBuf>| -> Result<Vec<PathBuf>> {
            if !self.manuals.is_empty() {
                return Ok(self.manuals.clone());
            }
            let root = self
                .manual_dir
                .clone()
                .or(dialect_dir)
                .with_context(|| format!("family {} needs manual-dir or manual", self.name))?;
            Ok(default.iter().map(|page| root.join(page)).collect())
        };
        let own_dir = self.dir.as_ref();
        let interpreter = |program: &str| -> Result<PathBuf> {
            self.interpreter
                .clone()
                .or_else(|| own_dir.map(|d| d.join("bin").join(program)))
                .with_context(|| format!("family {} needs interpreter or dir", self.name))
        };
        match self.name.as_str() {
            LUA51 => {
                ensure!(
                    self.dir.is_none()
                        && self.interpreter.is_none()
                        && self.sha256.is_none()
                        && self.manual_dir.is_none()
                        && self.manuals.is_empty(),
                    "the lua51 family is the lineage's --lua and --manual: give no interpreter or manual here"
                );
                Ok(Language::lua51(lua, manuals))
            }
            "python3" => Ok(Language::python3(
                &interpreter("python3")?,
                self.sha256.as_deref().unwrap_or(PYTHON3_SHA256),
                &pages(&PYTHON_PAGES, None)?,
            )),
            name @ ("zero-index" | "plus-concat") => {
                let default = if name == "zero-index" {
                    ZERO_INDEX_SHA256
                } else {
                    PLUS_CONCAT_SHA256
                };
                Language::dialect(
                    name,
                    &interpreter("lua")?,
                    self.sha256.as_deref().unwrap_or(default),
                    &pages(&DIALECT_PAGES, own_dir.map(|d| d.join("manual")))?,
                )
            }
            other => bail!("{other:?} is not a family (lua51, python3, zero-index, plus-concat)"),
        }
    }

    /// When the family's frontier is found: its `at`, else its 1-based
    /// position among `count` families.
    #[must_use]
    pub fn plan(&self, language: &Language, position: usize) -> FamilyPlan {
        FamilyPlan {
            name: language.name.clone(),
            at: self
                .at
                .clone()
                .unwrap_or_else(|| vec![u32::try_from(position + 1).unwrap_or(u32::MAX)]),
            target: self.target,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_family_is_a_name_and_settings() {
        let spec = FamilySpec::parse(
            "python3,at=2+4,target=3000,interpreter=/x/python3.14,manual-dir=/docs,dev-generator=120,keep-proposals=7",
        )
        .unwrap();
        assert_eq!(spec.name, "python3");
        assert_eq!(spec.at, Some(vec![2, 4]));
        assert_eq!(spec.target, Some(3000));
        assert_eq!(spec.sizes.dev_generator, Some(120));
        assert_eq!(spec.sizes.keep_proposals, Some(7));
        let language = spec.language(Path::new("lua"), &[]).unwrap();
        assert_eq!(language.interpreter.sha256.as_deref(), Some(PYTHON3_SHA256));
        assert!(language.manuals[0].ends_with("library/functions.txt"));
        assert_eq!(spec.plan(&language, 1).at, vec![2, 4]);
    }

    #[test]
    fn a_dialect_is_found_by_its_installation_directory() {
        let spec = FamilySpec::parse("zero-index,dir=/res/zero-index").unwrap();
        let language = spec.language(Path::new("lua"), &[]).unwrap();
        assert_eq!(language.name, "lua51-zero-index");
        assert_eq!(
            language.interpreter.path,
            Path::new("/res/zero-index/bin/lua")
        );
        assert!(language.manuals[0].ends_with("manual/strings.html"));
        assert_eq!(
            spec.plan(&language, 2).at,
            vec![3],
            "the third family joins in generation 3"
        );
    }

    #[test]
    fn a_malformed_or_unknown_setting_is_refused_by_name() {
        for bad in [
            "python3,target",
            "python3,speed=3",
            "python3,at=x",
            "python3,target=1,target=2",
            "=3",
        ] {
            assert!(FamilySpec::parse(bad).is_err(), "{bad}");
        }
        assert!(FamilySpec::parse("ruby")
            .unwrap()
            .language(Path::new("lua"), &[])
            .is_err());
        assert!(FamilySpec::parse("lua51,interpreter=/x")
            .unwrap()
            .language(Path::new("lua"), &[])
            .is_err());
    }
}
